"""Optional official pretrained stereo inference; no training or dataset imports.

RGB uint8 HxWx3 inputs, float32 HxW disparity in input pixels. Torch imports are
lazy; full official checkpoints replace every tensor, including BatchNorm state.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch



def environment_dll_handle() -> object | None:
    """Keep the active Windows Conda DLL directory in scope for Pillow/torchvision."""
    directory: Path = Path(sys.prefix)/"Library"/"bin"
    if sys.platform == "win32" and directory.is_dir():
        return os.add_dll_directory(str(directory))
    return None


def padding_for_shape(height: int, width: int) -> tuple[int,int,int,int]:
    """Official symmetric padding: left,right,top,bottom, no geometric resizing."""
    if min(height,width) <= 0:
        raise ValueError("Positive dimensions required")
    ph, pw = (-height)%32, (-width)%32
    return pw//2,pw-pw//2,ph//2,ph-ph//2


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


def predict_banet_rgb(model: torch.nn.Module, arrays: dict[str, NDArray[np.uint8]], maximum: int) -> NDArray[np.float32]:
    """Official dense prediction, RGB H,W,3 uint8; no GT shape placeholder required."""
    import torch
    import torch.nn.functional as functional

    height, width = arrays["left"].shape[:2]
    padding: tuple[int, int, int, int] = padding_for_shape(height, width)
    tensors: list[torch.Tensor] = [functional.pad(torch.from_numpy(np.ascontiguousarray(arrays[key].transpose(2, 0, 1))[None]).float().to("cuda"), padding, mode="replicate") for key in ("left", "right")]
    model.eval()
    with torch.inference_mode():
        output: torch.Tensor = model(*tensors, maximum)
    values: NDArray[np.float32] = output[0, 0, padding[2]:padding[2] + height, padding[0]:padding[0] + width].float().cpu().numpy().copy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite BANet output")
    return values


def prepare_igev(source_root: Path, runtime_root: Path) -> dict[str, object]:
    """Retain upstream bytes and record narrowly scoped import/timm compatibility edits."""
    files: tuple[str, ...] = ("__init__.py", "igev_stereo.py", "extractor.py", "geometry.py", "submodule.py", "update.py", "utils/__init__.py", "utils/utils.py")
    recorded: dict[str, object] = {}
    for filename in files:
        payload: bytes = (source_root/filename).read_bytes()
        body: str = payload.decode("utf-8").replace("from core.", "from _svp_igev_vendor.")
        if filename == "utils/utils.py":
            body = body.replace("from scipy import interpolate\n", "")
            body = body.replace("def forward_interpolate(flow):", "def forward_interpolate(flow):\n    from scipy import interpolate  # Optional helper, unused by IGEV forward")
        if filename == "extractor.py":
            if "pretrained =  True" not in body or "self.act1 = model.act1" not in body:
                raise ValueError("Unexpected IGEV encoder; review upstream before adapting")
            body = body.replace("pretrained =  True", "pretrained = False")
            body = body.replace("self.act1 = model.act1", "self.act1 = model.act1 if hasattr(model, 'act1') else nn.Identity()")
        target: Path = runtime_root/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8", newline="\n")
        recorded[filename] = {"source_sha256": hashlib.sha256(payload).hexdigest(), "runtime_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    result: dict[str, object] = {"files": recorded, "patches": ["private import namespace", "disable all external encoder weights", "timm fused-activation compatibility fallback", "lazy scipy import in unused forward_interpolate helper"]}
    (runtime_root/"compatibility.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _create_igev(source_root: Path, runtime_root: Path, seed: int, precision: str, maximum: int, normalization: str) -> tuple[torch.nn.Module, dict[str, object]]:
    """Random initialization only; no checkpoint or download argument is accepted."""
    dll_handle: object | None = environment_dll_handle()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for IGEV experiment")
    if precision not in ("float32", "bfloat16") or maximum < 32 or maximum % 32:
        raise ValueError("Require FP32/BF16 and a positive 32-aligned search range")
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")
    torch.manual_seed(seed)
    compatibility: dict[str, object] = prepare_igev(source_root, runtime_root)
    package: str = "_svp_igev_vendor"
    for name in list(sys.modules):
        if name == package or name.startswith(package+"."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(package, runtime_root/"__init__.py", submodule_search_locations=[str(runtime_root)])
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load fixed IGEV source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    model_type: type = importlib.import_module(package+".igev_stereo").IGEVStereo
    args: SimpleNamespace = SimpleNamespace(hidden_dims=[128, 128, 128], n_downsample=2, n_gru_layers=3, corr_levels=2, corr_radius=4,
        max_disp=maximum, mixed_precision=precision == "bfloat16", precision_dtype=precision)
    model: torch.nn.Module = model_type(args)
    if normalization != "running":
        raise ValueError("Pretrained inference requires saved BatchNorm statistics")
    model = model.to("cuda")
    model._svp_dll_handle = dll_handle
    return model, {"parameters": sum(value.numel() for value in model.parameters()), "initialization": "scratch_no_external_weights",
        "batch_norm_policy": normalization, "precision": precision, "gpu": torch.cuda.get_device_name(0), "compatibility": compatibility}


def predict_igev(model: torch.nn.Module, arrays: dict[str, NDArray[np.uint8]], iterations: int) -> NDArray[np.float32]:
    """Return float32 HW disparity in input pixels; no GT values used by model."""
    import torch
    import torch.nn.functional as functional

    if type(iterations) is not int or iterations < 1:
        raise ValueError("Iterations must be a positive integer")
    if arrays["left"].shape != arrays["right"].shape or arrays["left"].ndim != 3 or arrays["left"].shape[2] != 3:
        raise ValueError("Expected equal-size RGB HW3 image pair")
    height, width = arrays["left"].shape[:2]
    padding: tuple[int, int, int, int] = padding_for_shape(height, width)
    tensors: list[torch.Tensor] = [functional.pad(torch.from_numpy(np.ascontiguousarray(arrays[key].transpose(2, 0, 1))[None]).float().to("cuda"), padding, mode="replicate") for key in ("left", "right")]
    model.eval()
    with torch.no_grad():
        output: torch.Tensor = model(*tensors, iters=iterations, test_mode=True)
    result: np.ndarray = output[0, 0, padding[2]:padding[2]+height, padding[0]:padding[0]+width].float().cpu().numpy().copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite IGEV prediction")
    return result


def load_igev(config: Path, raw: dict[str, object]) -> tuple[torch.nn.Module, dict[str, object]]:
    """Strict-load the entire official state, including saved BN statistics."""
    checkpoint: Path = (config.parent/raw["checkpoint"]).resolve()
    actual: str = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != raw["checkpoint_sha256"] or raw["batch_norm_policy"] != "running":
        raise ValueError("Wrong checkpoint identity or pretrained normalization policy")
    dll_handle: object | None = environment_dll_handle()
    import torch

    state: dict[str, torch.Tensor] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if all(name.startswith("module.") for name in state):
        state = {name[7:]: value for name, value in state.items()}
    model, info = _create_igev((config.parent/raw["source_root"]).resolve(), (config.parent/raw["runtime_root"]).resolve(), 0, raw["precision"], raw["max_disparity"], "running")
    if set(state) != set(model.state_dict()):
        raise ValueError(f"Official checkpoint keys mismatch: missing={set(model.state_dict())-set(state)}, unexpected={set(state)-set(model.state_dict())}")
    model.load_state_dict(state, strict=True)
    model.eval()
    model._svp_pretrained_dll_handle = dll_handle
    info.update(initialization="official Scene Flow full checkpoint, every key loaded strictly", checkpoint_sha256=actual,
        torch=torch.__version__, external_weights_loaded=True, provenance_limit="Official file name/source describes Scene Flow; entire historical author training data cannot be independently audited")
    return model, info
