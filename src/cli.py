"""离线重建与双目标定的命令行入口。"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import cv2

from .calibration_run import CalibrationRunResult, run_calibration
from .config import CalibrationConfig, OfflineConfig, load_calibration_config, load_offline_config
from .pipeline import OfflineRunResult, run_offline_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    """运行离线流程或标定流程，成功时输出 JSON，失败时返回 1。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Middlebury 离线重建或基于棋盘格的双目标定。",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="运行双目标定，默认读取 config/calibration.json。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON 配置路径；离线默认 config/offline.json，标定默认 config/calibration.json。",
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
    parser.add_argument("--square-mm", type=float, help="覆盖棋盘格格长（毫米），仅标定使用。")
    parser.add_argument("--baseline-mm", type=float, help="覆盖基线（毫米），仅标定使用。")
    parser.add_argument("--output-dir", type=Path, help="覆盖标定产物目录。")
    args: argparse.Namespace = parser.parse_args(argv)
    if args.calibrate:
        return _run_calibration(args)
    return _run_offline(args)


def _run_offline(args: argparse.Namespace) -> int:
    """运行 Middlebury 离线流程并输出指标。"""
    config_path: Path = args.config if args.config is not None else Path("config/offline.json")
    try:
        settings: OfflineConfig = load_offline_config(
            config_path,
            image_root_override=args.image_root,
            ground_truth_root_override=args.ground_truth_root,
            split_override=args.split,
            scene_override=args.scene,
        )
        if args.no_export:
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


def _run_calibration(args: argparse.Namespace) -> int:
    """运行双目标定并输出质量摘要。"""
    config_path: Path = args.config if args.config is not None else Path("config/calibration.json")
    try:
        settings: CalibrationConfig = load_calibration_config(
            config_path,
            square_size_mm_override=args.square_mm,
            baseline_mm_override=args.baseline_mm,
            output_dir_override=args.output_dir,
        )
        result: CalibrationRunResult = run_calibration(settings)
    except (OSError, ValueError, cv2.error) as error:
        print(f"标定失败（{config_path}）：{error}", file=sys.stderr)
        return 1

    calibration = result.calibration
    summary: dict[str, object] = {
        "pairs_used": len(calibration.pair_errors),
        "pairs_total": len(result.records),
        "image_size": list(calibration.image_size),
        "targets_used": {
            name: calibration.pair_targets.count(name) for name in calibration.targets
        },
        "baseline_mm": calibration.baseline_m * 1000.0,
        "scale_source": calibration.scale_source,
        "rms_stereo_px": calibration.rms_px,
        "rms_left_px": calibration.left.rms_px,
        "rms_right_px": calibration.right.rms_px,
        "vertical_residual_px": {
            "mean": result.vertical.mean_px,
            "p95": result.vertical.p95_px,
            "max": result.vertical.max_px,
        },
        "rectified_focal_px": result.rectification.focal_px,
        "doffs_px": result.rectification.doffs_px,
        "recommended_ndisp": result.ndisp,
        "output_dir": str(result.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
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
