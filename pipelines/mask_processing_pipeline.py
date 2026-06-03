# pipelines/mask_processing_pipeline.py
# Wrapper that re-exports MaskRefinementPipeline as MaskProcessingPipeline

from pipelines.mask_refinement_pipeline import (
    MaskRefinementPipeline as MaskProcessingPipeline,
    MaskRefinementPipelineConfig as MaskProcessingPipelineConfig,
    MaskRefinementPipelineResult as MaskProcessingPipelineResult,
)

__all__ = [
    "MaskProcessingPipeline",
    "MaskProcessingPipelineConfig",
    "MaskProcessingPipelineResult",
]
