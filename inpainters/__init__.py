# inpainters/__init__.py

from inpainters.opencv_inpainter import (
    OpenCVInpaintResult,
    OpenCVInpainter,
    OpenCVInpainterConfig,
    opencv_inpaint,
)

from inpainters.sdxl_inpainter import (
    SDXLInpaintResult,
    SDXLInpainter,
    SDXLInpainterConfig,
    sdxl_inpaint,
)

from inpainters.stable_diffusion_inpainter import (
    StableDiffusionInpaintResult,
    StableDiffusionInpainter,
    StableDiffusionInpainterConfig,
    stable_diffusion_inpaint,
)

from inpainters.flux_inpainter import (
    FluxInpaintResult,
    FluxInpainter,
    FluxInpainterConfig,
    flux_inpaint,
)

from inpainters.lama_inpainter import (
    LaMaInpaintResult,
    LaMaInpainter,
    LaMaInpainterConfig,
    lama_inpaint,
)

__all__ = [
    "OpenCVInpaintResult",
    "OpenCVInpainter",
    "OpenCVInpainterConfig",
    "opencv_inpaint",
    "SDXLInpaintResult",
    "SDXLInpainter",
    "SDXLInpainterConfig",
    "sdxl_inpaint",
    "StableDiffusionInpaintResult",
    "StableDiffusionInpainter",
    "StableDiffusionInpainterConfig",
    "stable_diffusion_inpaint",
    "FluxInpaintResult",
    "FluxInpainter",
    "FluxInpainterConfig",
    "flux_inpaint",
    "LaMaInpaintResult",
    "LaMaInpainter",
    "LaMaInpainterConfig",
    "lama_inpaint",
]
