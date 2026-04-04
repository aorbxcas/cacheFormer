from renderformer.models.renderformer import RenderFormer

__all__ = ["RenderFormer", "RenderFormerRenderingPipeline"]


def __getattr__(name: str):
    if name == "RenderFormerRenderingPipeline":
        from renderformer.pipelines.rendering_pipeline import RenderFormerRenderingPipeline

        return RenderFormerRenderingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
