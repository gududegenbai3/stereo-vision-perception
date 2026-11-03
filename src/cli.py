"""Middlebury 单场景离线流程的命令行入口。"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import cv2

from .config import OfflineConfig, load_offline_config
from .pipeline import OfflineRunResult, run_offline_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    """运行离线流程，成功时输出 JSON，运行失败时返回 1。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="使用 Middlebury 双目图像进行离线重建、评估和导出。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/offline.json"),
        help="离线 JSON 配置路径，默认 config/offline.json。",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="仅运行重建与评估，不写出文件。",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="在桌面窗口中查看视差图和深度图。",
    )
    parser.add_argument("--image-root", type=Path, help="覆盖 Middlebury 图像和标定根目录。")
    parser.add_argument("--ground-truth-root", type=Path, help="覆盖训练集真值根目录。")
    parser.add_argument("--split", choices=("trainingQ", "testQ"), help="覆盖数据集划分。")
    parser.add_argument("--scene", help="覆盖场景名，例如 Australia。")
    args: argparse.Namespace = parser.parse_args(argv)
    config_path: Path = args.config
    no_export: bool = args.no_export

    try:
        settings: OfflineConfig = load_offline_config(
            config_path,
            image_root_override=args.image_root,
            ground_truth_root_override=args.ground_truth_root,
            split_override=args.split,
            scene_override=args.scene,
        )
        if no_export:
            settings = replace(settings, output_root=None)
        result: OfflineRunResult = run_offline_pipeline(settings)
    except (OSError, ValueError, cv2.error) as error:
        print(f"离线流程失败（{config_path}）：{error}", file=sys.stderr)
        return 1

    summary: dict[str, object] = {
        "split": settings.split,
        "scene_name": settings.scene_name,
        "metrics": asdict(result.metrics),
        "output_dir": str(result.output_dir) if result.output_dir is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if args.preview:
        try:
            _show_preview(result)
        except (ImportError, RuntimeError, OSError) as error:
            print(f"重建已完成，预览窗口无法打开：{error}", file=sys.stderr)
            return 1
    return 0


def _show_preview(result: OfflineRunResult) -> None:
    """显示原分辨率视差与深度，关闭窗口后返回。"""
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

    fig: Figure = plt.figure(figsize=(11, 4), layout="constrained")
    try:
        if fig.canvas.required_interactive_framework is None:
            raise RuntimeError("请在桌面终端使用交互式 Matplotlib 后端")
        disparity_ax: Axes = fig.add_subplot(1, 2, 1)
        depth_ax: Axes = fig.add_subplot(1, 2, 2)
        disparity: AxesImage = disparity_ax.imshow(
            result.disparity.disparity_px, cmap=plt.get_cmap("turbo").with_extremes(bad="0.25")
        )
        depth: AxesImage = depth_ax.imshow(
            result.reconstruction.depth_m, cmap=plt.get_cmap("viridis").with_extremes(bad="0.25")
        )
        disparity_ax.set_title("Disparity")
        depth_ax.set_title("Depth")
        disparity_ax.set_axis_off()
        depth_ax.set_axis_off()
        fig.colorbar(disparity, ax=disparity_ax, label="Disparity (px)")
        fig.colorbar(depth, ax=depth_ax, label="Depth (m)")
        plt.show(block=True)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
