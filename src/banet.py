"""Optional BANet-2D inference adapter; baseline modules do not import PyTorch.

Official implementation is cached outside tracked source. Two explicit compatibility
patches affect initialization only; strict loading of every tensor remains required.
RGB float32 0..255 follows the author's save_disp.py and StereoDataset.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .model import DisparityResult, RectifiedPair

if TYPE_CHECKING:
    import torch


def environment_dll_handle() -> object | None:
    """Keep the active Windows Conda DLL directory in scope for Pillow/torchvision."""
    directory: Path = Path(sys.prefix)/"Library"/"bin"
    if sys.platform == "win32" and directory.is_dir():
        return os.add_dll_directory(str(directory))
    return None


@dataclass(frozen=True)
class BanetPrediction:
    disparity: DisparityResult
    attention: np.ndarray  # float32 (H,W), learned soft branch weight, not confidence
    timing_ms: tuple[float, ...]
    peak_allocated_mib: float
    peak_reserved_mib: float
    padded_shape: tuple[int, int]


def padding_for_shape(height: int, width: int) -> tuple[int,int,int,int]:
    """Official symmetric padding: left,right,top,bottom, no geometric resizing."""
    if min(height,width) <= 0:
        raise ValueError("Positive dimensions required")
    ph, pw = (-height)%32, (-width)%32
    return pw//2,pw-pw//2,ph//2,ph-ph//2


def valid_disparity(disparity: np.ndarray, max_disparity: int) -> DisparityResult:
    """Finite px output and in-image x_right=x_left-d; no GT-based filtering."""
    if disparity.ndim != 2 or disparity.dtype != np.float32:
        raise ValueError("Expected float32 (H,W) pixel disparities")
    height,width = disparity.shape
    columns: np.ndarray = np.arange(width)[None,:]
    valid: np.ndarray = np.isfinite(disparity) & (disparity>=0) & (disparity<max_disparity) & (columns-disparity>=0)
    values: np.ndarray = np.where(valid,disparity,np.nan).astype(np.float32)
    return DisparityResult(values,valid,(0,0,width,height))


def prepare_vendor(source_root: Path, runtime_root: Path) -> dict:
    """Copy the inference-only official modules and record minimal API patches."""
    runtime_root.mkdir(parents=True,exist_ok=True)
    manifest: dict = {"patches":["pretrained=False: full BANet state replaces encoder weights",
        "act1 fallback to Identity only when timm already fuses activation into bn1"],"files":{}}
    for filename in ("__init__.py","BANet.py","fnet.py","aggregation.py","submodule.py"):
        source: bytes = (source_root/filename).read_bytes()
        text: str = source.decode("utf-8")
        if filename == "fnet.py":
            if "pretrained=True" not in text or "self.act1 = model.act1" not in text:
                raise ValueError("Unexpected upstream encoder source; review compatibility patch")
            text = text.replace("pretrained=True","pretrained=False")
            text = text.replace("self.act1 = model.act1","self.act1 = model.act1 if hasattr(model, 'act1') else nn.Identity()")
        destination: Path = runtime_root/filename
        if filename == "fnet.py":
            destination.write_text(text,encoding="utf-8")
        else:
            destination.write_bytes(source)
        manifest["files"][filename] = {"source_sha256":hashlib.sha256(source).hexdigest(),
            "runtime_sha256":hashlib.sha256(destination.read_bytes()).hexdigest()}
    (runtime_root/"compatibility.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


def load_banet(source_root: Path, runtime_root: Path, checkpoint: Path, expected_sha256: str) -> tuple[torch.nn.Module,dict]:
    """Load official full checkpoint strictly on CUDA; reject missing/random weights."""
    dll_handle: object | None = environment_dll_handle()
    import torch
    import timm

    actual: str = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently substitute CPU")
    compatibility: dict = prepare_vendor(source_root,runtime_root)
    package: str = "_svp_banet_vendor"
    spec = importlib.util.spec_from_file_location(package,runtime_root/"__init__.py",submodule_search_locations=[str(runtime_root)])
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load official BANet package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    model_type = importlib.import_module(package+".BANet").BANet
    model: torch.nn.Module = model_type(SimpleNamespace(max_disp=192))
    state: dict = torch.load(checkpoint,map_location="cpu",weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."):value for key,value in state.items()}
    incompatible = model.load_state_dict(state,strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Incomplete model weights")
    model = model.eval().to("cuda")
    model._svp_dll_handle = dll_handle
    info: dict = {"checkpoint_sha256":actual,"strict_load":True,"parameter_count":sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),"torch":torch.__version__,
        "timm":timm.__version__,"cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
        "gpu":torch.cuda.get_device_name(0),"compute_capability":list(torch.cuda.get_device_capability(0)),
        "compiled_architectures":torch.cuda.get_arch_list(),"compatibility":compatibility}
    return model,info


def predict_banet(model: torch.nn.Module, pair: RectifiedPair, max_disparity: int, warmup_runs: int, timed_runs: int) -> BanetPrediction:
    """Original-size CUDA FP32 inference; synchronized forward-only timing, no GT."""
    import torch
    import torch.nn.functional as functional

    if max_disparity != 192:
        raise ValueError("Official 2D checkpoint has 48 correlation channels: max_disp must be 192")
    if warmup_runs<0 or timed_runs<1:
        raise ValueError("Invalid profiling counts")
    if pair.left_bgr.shape != pair.right_bgr.shape or pair.left_bgr.ndim != 3 or pair.left_bgr.shape[2] != 3:
        raise ValueError("Equal BGR (H,W,3) inputs required")
    if pair.left_bgr.dtype != np.uint8 or pair.right_bgr.dtype != np.uint8:
        raise ValueError("uint8 inputs required")
    height,width = pair.left_bgr.shape[:2]
    padding: tuple[int,int,int,int] = padding_for_shape(height,width)
    left_pad,right_pad,top_pad,bottom_pad = padding
    def tensor(image: np.ndarray) -> torch.Tensor:
        rgb: np.ndarray = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        value: torch.Tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().to("cuda")
        return functional.pad(value,padding,mode="replicate")
    left,right = tensor(pair.left_bgr),tensor(pair.right_bgr)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup_runs):
            output: torch.Tensor = model(left,right,max_disparity)
        torch.cuda.synchronize()
        for _ in range(timed_runs):
            start: float = time.perf_counter()
            output = model(left,right,max_disparity)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        peak_allocated: float = torch.cuda.max_memory_allocated()/1024**2
        peak_reserved: float = torch.cuda.max_memory_reserved()/1024**2
        disparity: np.ndarray = output[0,0,top_pad:top_pad+height,left_pad:left_pad+width].float().cpu().numpy().copy()
        captured: dict[str,torch.Tensor] = {}
        def capture_attention(_module: torch.nn.Module, _inputs: tuple, result: torch.Tensor) -> None:
            captured["attention"] = result.detach()
        handle = model.spa_att.register_forward_hook(capture_attention)
        try:
            model(left,right,max_disparity)
        finally:
            handle.remove()
        attention_tensor: torch.Tensor = functional.interpolate(captured["attention"],left.shape[-2:],mode="bilinear",align_corners=False)
        attention: np.ndarray = attention_tensor[0,0,top_pad:top_pad+height,left_pad:left_pad+width].float().cpu().numpy().copy()
    return BanetPrediction(valid_disparity(disparity.astype(np.float32),max_disparity),attention,tuple(times),peak_allocated,
                           peak_reserved,(height+top_pad+bottom_pad,width+left_pad+right_pad))
