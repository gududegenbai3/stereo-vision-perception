"""编排已校正的 Middlebury 双目图像的离线重建流程。"""

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import OfflineConfig
from .datasets import MiddleburyGroundTruth, MiddleburyScene, load_middlebury_scene
from .disparity import compute_disparity
from .evaluation import DisparityMetrics, evaluate_disparity
from .model import DisparityResult, ReconstructionResult
from .outputs import export_offline_result
from .reprojection import reproject_disparity


@dataclass(frozen=True, eq=False)
class OfflineRunResult:
    """单场景的匹配、重建和评估结果，未导出时 output_dir 为 None。"""

    disparity: DisparityResult
    reconstruction: ReconstructionResult
    metrics: DisparityMetrics
    output_dir: Path | None


def run_offline_pipeline(settings: OfflineConfig) -> OfflineRunResult:
    """运行单场景离线流程，使用调用方已校验的配置。

    测试场景无公开真值，仍计算匹配率。仅在配置了 output_root 时导出。
    输入缺失、数据无效或写入失败时，向调用方传播对应阶段的异常。
    """
    scene: MiddleburyScene = load_middlebury_scene(
        settings.image_root,
        settings.split,
        settings.scene_name,
        ground_truth_root=settings.ground_truth_root,
    )
    disparity: DisparityResult = compute_disparity(
        scene.rectified_pair, scene.geometry, settings.sgbm
    )
    reconstruction: ReconstructionResult = reproject_disparity(
        scene.rectified_pair, scene.geometry, disparity
    )

    truth: MiddleburyGroundTruth | None = scene.ground_truth
    metrics: DisparityMetrics = evaluate_disparity(
        disparity,
        reference_px=truth.disparity_px if truth is not None else None,
        reference_mask=truth.non_occluded_mask if truth is not None else None,
    )

    output_dir: Path | None = None
    if settings.output_root is not None:
        output_dir = export_offline_result(
            settings.output_root,
            reconstruction,
            asdict(metrics),
            split=scene.split,
            scene_name=scene.scene_name,
            sgbm=settings.sgbm,
            search_roi=disparity.valid_roi,
        )

    return OfflineRunResult(
        disparity=disparity,
        reconstruction=reconstruction,
        metrics=metrics,
        output_dir=output_dir,
    )
