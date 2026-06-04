# ui/components/inpainter_settings.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ui.control_defaults import DEVICE_CHOICES, inpainter_default


@dataclass(slots=True)
class InpainterSettingsPanel:
    inputs: list[Any]
    input_names: list[str]


def build_inpainter_settings_panel(gr: Any) -> InpainterSettingsPanel:
    gr.Markdown("## Inpainter settings")

    input_components: list[Any] = []
    input_names: list[str] = []

    with gr.Accordion("General inpainter settings", open=False):
        with gr.Row():
            pipeline_enabled = gr.Checkbox(
                label="Enable inpainting",
                value=inpainter_default("pipeline_enabled"),
            )

            fallback_enabled = gr.Checkbox(
                label="Fallback to next backend on error",
                value=inpainter_default("fallback_enabled"),
            )

    add_inputs(
        input_components,
        input_names,
        {
            "pipeline_enabled": pipeline_enabled,
            "fallback_enabled": fallback_enabled,
        },
    )

    with gr.Accordion("OpenCV inpainter settings", open=False):
        opencv_enabled = gr.Checkbox(
            label="Enable OpenCV backend",
            value=inpainter_default("opencv.enabled"),
        )

        with gr.Row():
            opencv_method = gr.Dropdown(
                label="OpenCV method",
                choices=["telea", "ns"],
                value=inpainter_default("opencv.method"),
            )

            opencv_radius = gr.Slider(
                label="OpenCV radius",
                minimum=1,
                maximum=15,
                step=0.5,
                value=inpainter_default("opencv.radius"),
            )

        with gr.Row():
            opencv_dilate_mask_iterations = gr.Slider(
                label="OpenCV mask dilate",
                minimum=0,
                maximum=12,
                step=1,
                value=inpainter_default("opencv.dilate_mask_iterations"),
            )

            opencv_mask_kernel_size = gr.Slider(
                label="OpenCV mask kernel",
                minimum=1,
                maximum=15,
                step=2,
                value=inpainter_default("opencv.mask_kernel_size"),
            )

    add_inputs(
        input_components,
        input_names,
        {
            "opencv.enabled": opencv_enabled,
            "opencv.method": opencv_method,
            "opencv.radius": opencv_radius,
            "opencv.dilate_mask_iterations": opencv_dilate_mask_iterations,
            "opencv.mask_kernel_size": opencv_mask_kernel_size,
        },
    )

    with gr.Accordion("LaMa inpainter settings", open=False):
        with gr.Row():
            lama_enabled = gr.Checkbox(
                label="Enable LaMa backend",
                value=inpainter_default("lama.enabled"),
            )

            lama_check_model_path = gr.Checkbox(
                label="Check LaMa model path",
                value=inpainter_default("lama.check_model_path"),
            )

        lama_model_path = gr.Textbox(
            label="LaMa model path",
            value=inpainter_default("lama.model_path"),
        )

        with gr.Row():
            lama_device = gr.Dropdown(
                label="LaMa device",
                choices=DEVICE_CHOICES,
                value=inpainter_default("lama.device"),
            )

            lama_modulo = gr.Number(
                label="LaMa size modulo",
                value=inpainter_default("lama.modulo"),
                precision=0,
            )

        with gr.Row():
            lama_dilate_mask_iterations = gr.Slider(
                label="LaMa mask dilate",
                minimum=0,
                maximum=12,
                step=1,
                value=inpainter_default("lama.dilate_mask_iterations"),
            )

            lama_mask_kernel_size = gr.Slider(
                label="LaMa mask kernel",
                minimum=1,
                maximum=15,
                step=2,
                value=inpainter_default("lama.mask_kernel_size"),
            )

    add_inputs(
        input_components,
        input_names,
        {
            "lama.enabled": lama_enabled,
            "lama.model_path": lama_model_path,
            "lama.device": lama_device,
            "lama.modulo": lama_modulo,
            "lama.dilate_mask_iterations": lama_dilate_mask_iterations,
            "lama.mask_kernel_size": lama_mask_kernel_size,
            "lama.check_model_path": lama_check_model_path,
        },
    )

    with gr.Accordion("Stable Diffusion inpainter settings", open=False):
        stable_diffusion_enabled = gr.Checkbox(
            label="Enable Stable Diffusion backend",
            value=inpainter_default("stable_diffusion.enabled"),
        )

        stable_diffusion_model_id = gr.Textbox(
            label="Stable Diffusion model ID",
            value=inpainter_default("stable_diffusion.model_id"),
        )

        stable_diffusion_prompt = gr.Textbox(
            label="Stable Diffusion prompt",
            value=inpainter_default("stable_diffusion.prompt"),
            lines=2,
        )

        stable_diffusion_negative_prompt = gr.Textbox(
            label="Stable Diffusion negative prompt",
            value=inpainter_default("stable_diffusion.negative_prompt"),
            lines=2,
        )

        with gr.Row():
            stable_diffusion_device = gr.Dropdown(
                label="Stable Diffusion device",
                choices=DEVICE_CHOICES,
                value=inpainter_default("stable_diffusion.device"),
            )

            stable_diffusion_torch_dtype = gr.Dropdown(
                label="Stable Diffusion torch dtype",
                choices=["float16", "float32", "bfloat16"],
                value=inpainter_default("stable_diffusion.torch_dtype"),
            )

        with gr.Row():
            stable_diffusion_num_inference_steps = gr.Number(
                label="Stable Diffusion steps",
                value=inpainter_default("stable_diffusion.num_inference_steps"),
                precision=0,
            )

            stable_diffusion_guidance_scale = gr.Slider(
                label="Stable Diffusion guidance",
                minimum=0,
                maximum=20,
                step=0.5,
                value=inpainter_default("stable_diffusion.guidance_scale"),
            )

            stable_diffusion_strength = gr.Slider(
                label="Stable Diffusion strength",
                minimum=0,
                maximum=1,
                step=0.01,
                value=inpainter_default("stable_diffusion.strength"),
            )

        with gr.Row():
            stable_diffusion_seed = gr.Number(
                label="Stable Diffusion seed, -1 = random",
                value=inpainter_default("stable_diffusion.seed"),
                precision=0,
            )

            stable_diffusion_resize_to_multiple_of = gr.Number(
                label="Stable Diffusion resize multiple",
                value=inpainter_default("stable_diffusion.resize_to_multiple_of"),
                precision=0,
            )

        with gr.Row():
            stable_diffusion_dilate_mask_iterations = gr.Slider(
                label="Stable Diffusion mask dilate",
                minimum=0,
                maximum=12,
                step=1,
                value=inpainter_default("stable_diffusion.dilate_mask_iterations"),
            )

            stable_diffusion_mask_kernel_size = gr.Slider(
                label="Stable Diffusion mask kernel",
                minimum=1,
                maximum=15,
                step=2,
                value=inpainter_default("stable_diffusion.mask_kernel_size"),
            )

        with gr.Row():
            stable_diffusion_enable_model_cpu_offload = gr.Checkbox(
                label="Stable Diffusion CPU offload",
                value=inpainter_default("stable_diffusion.enable_model_cpu_offload"),
            )

            stable_diffusion_enable_attention_slicing = gr.Checkbox(
                label="Stable Diffusion attention slicing",
                value=inpainter_default("stable_diffusion.enable_attention_slicing"),
            )

    add_inputs(
        input_components,
        input_names,
        {
            "stable_diffusion.enabled": stable_diffusion_enabled,
            "stable_diffusion.model_id": stable_diffusion_model_id,
            "stable_diffusion.device": stable_diffusion_device,
            "stable_diffusion.prompt": stable_diffusion_prompt,
            "stable_diffusion.negative_prompt": stable_diffusion_negative_prompt,
            "stable_diffusion.num_inference_steps": stable_diffusion_num_inference_steps,
            "stable_diffusion.guidance_scale": stable_diffusion_guidance_scale,
            "stable_diffusion.strength": stable_diffusion_strength,
            "stable_diffusion.seed": stable_diffusion_seed,
            "stable_diffusion.resize_to_multiple_of": stable_diffusion_resize_to_multiple_of,
            "stable_diffusion.dilate_mask_iterations": stable_diffusion_dilate_mask_iterations,
            "stable_diffusion.mask_kernel_size": stable_diffusion_mask_kernel_size,
            "stable_diffusion.enable_model_cpu_offload": stable_diffusion_enable_model_cpu_offload,
            "stable_diffusion.enable_attention_slicing": stable_diffusion_enable_attention_slicing,
            "stable_diffusion.torch_dtype": stable_diffusion_torch_dtype,
        },
    )

    with gr.Accordion("SDXL inpainter settings", open=False):
        sdxl_enabled = gr.Checkbox(
            label="Enable SDXL backend",
            value=inpainter_default("sdxl.enabled"),
        )

        sdxl_model_id = gr.Textbox(
            label="SDXL model ID",
            value=inpainter_default("sdxl.model_id"),
        )

        sdxl_prompt = gr.Textbox(
            label="SDXL prompt",
            value=inpainter_default("sdxl.prompt"),
            lines=2,
        )

        sdxl_negative_prompt = gr.Textbox(
            label="SDXL negative prompt",
            value=inpainter_default("sdxl.negative_prompt"),
            lines=2,
        )

        with gr.Row():
            sdxl_device = gr.Dropdown(
                label="SDXL device",
                choices=DEVICE_CHOICES,
                value=inpainter_default("sdxl.device"),
            )

            sdxl_torch_dtype = gr.Dropdown(
                label="SDXL torch dtype",
                choices=["float16", "float32", "bfloat16"],
                value=inpainter_default("sdxl.torch_dtype"),
            )

        with gr.Row():
            sdxl_num_inference_steps = gr.Number(
                label="SDXL steps",
                value=inpainter_default("sdxl.num_inference_steps"),
                precision=0,
            )

            sdxl_guidance_scale = gr.Slider(
                label="SDXL guidance",
                minimum=0,
                maximum=20,
                step=0.5,
                value=inpainter_default("sdxl.guidance_scale"),
            )

            sdxl_strength = gr.Slider(
                label="SDXL strength",
                minimum=0,
                maximum=1,
                step=0.01,
                value=inpainter_default("sdxl.strength"),
            )

        with gr.Row():
            sdxl_seed = gr.Number(
                label="SDXL seed, -1 = random",
                value=inpainter_default("sdxl.seed"),
                precision=0,
            )

            sdxl_resize_to_multiple_of = gr.Number(
                label="SDXL resize multiple",
                value=inpainter_default("sdxl.resize_to_multiple_of"),
                precision=0,
            )

        with gr.Row():
            sdxl_dilate_mask_iterations = gr.Slider(
                label="SDXL mask dilate",
                minimum=0,
                maximum=12,
                step=1,
                value=inpainter_default("sdxl.dilate_mask_iterations"),
            )

            sdxl_mask_kernel_size = gr.Slider(
                label="SDXL mask kernel",
                minimum=1,
                maximum=15,
                step=2,
                value=inpainter_default("sdxl.mask_kernel_size"),
            )

        with gr.Row():
            sdxl_enable_model_cpu_offload = gr.Checkbox(
                label="SDXL CPU offload",
                value=inpainter_default("sdxl.enable_model_cpu_offload"),
            )

            sdxl_enable_attention_slicing = gr.Checkbox(
                label="SDXL attention slicing",
                value=inpainter_default("sdxl.enable_attention_slicing"),
            )

    add_inputs(
        input_components,
        input_names,
        {
            "sdxl.enabled": sdxl_enabled,
            "sdxl.model_id": sdxl_model_id,
            "sdxl.device": sdxl_device,
            "sdxl.prompt": sdxl_prompt,
            "sdxl.negative_prompt": sdxl_negative_prompt,
            "sdxl.num_inference_steps": sdxl_num_inference_steps,
            "sdxl.guidance_scale": sdxl_guidance_scale,
            "sdxl.strength": sdxl_strength,
            "sdxl.seed": sdxl_seed,
            "sdxl.resize_to_multiple_of": sdxl_resize_to_multiple_of,
            "sdxl.dilate_mask_iterations": sdxl_dilate_mask_iterations,
            "sdxl.mask_kernel_size": sdxl_mask_kernel_size,
            "sdxl.enable_model_cpu_offload": sdxl_enable_model_cpu_offload,
            "sdxl.enable_attention_slicing": sdxl_enable_attention_slicing,
            "sdxl.torch_dtype": sdxl_torch_dtype,
        },
    )

    with gr.Accordion("Flux inpainter settings", open=False):
        flux_enabled = gr.Checkbox(
            label="Enable Flux backend",
            value=inpainter_default("flux.enabled"),
        )

        flux_model_id = gr.Textbox(
            label="Flux model ID",
            value=inpainter_default("flux.model_id"),
        )

        flux_prompt = gr.Textbox(
            label="Flux prompt",
            value=inpainter_default("flux.prompt"),
            lines=2,
        )

        flux_negative_prompt = gr.Textbox(
            label="Flux negative prompt",
            value=inpainter_default("flux.negative_prompt"),
            lines=2,
        )

        with gr.Row():
            flux_device = gr.Dropdown(
                label="Flux device",
                choices=DEVICE_CHOICES,
                value=inpainter_default("flux.device"),
            )

            flux_torch_dtype = gr.Dropdown(
                label="Flux torch dtype",
                choices=["float16", "float32", "bfloat16"],
                value=inpainter_default("flux.torch_dtype"),
            )

        with gr.Row():
            flux_num_inference_steps = gr.Number(
                label="Flux steps",
                value=inpainter_default("flux.num_inference_steps"),
                precision=0,
            )

            flux_guidance_scale = gr.Slider(
                label="Flux guidance",
                minimum=0,
                maximum=50,
                step=0.5,
                value=inpainter_default("flux.guidance_scale"),
            )

            flux_seed = gr.Number(
                label="Flux seed, -1 = random",
                value=inpainter_default("flux.seed"),
                precision=0,
            )

        with gr.Row():
            flux_resize_to_multiple_of = gr.Number(
                label="Flux resize multiple",
                value=inpainter_default("flux.resize_to_multiple_of"),
                precision=0,
            )

            flux_dilate_mask_iterations = gr.Slider(
                label="Flux mask dilate",
                minimum=0,
                maximum=12,
                step=1,
                value=inpainter_default("flux.dilate_mask_iterations"),
            )

            flux_mask_kernel_size = gr.Slider(
                label="Flux mask kernel",
                minimum=1,
                maximum=15,
                step=2,
                value=inpainter_default("flux.mask_kernel_size"),
            )

        with gr.Row():
            flux_enable_model_cpu_offload = gr.Checkbox(
                label="Flux CPU offload",
                value=inpainter_default("flux.enable_model_cpu_offload"),
            )

            flux_enable_attention_slicing = gr.Checkbox(
                label="Flux attention slicing",
                value=inpainter_default("flux.enable_attention_slicing"),
            )

    add_inputs(
        input_components,
        input_names,
        {
            "flux.enabled": flux_enabled,
            "flux.model_id": flux_model_id,
            "flux.device": flux_device,
            "flux.prompt": flux_prompt,
            "flux.negative_prompt": flux_negative_prompt,
            "flux.num_inference_steps": flux_num_inference_steps,
            "flux.guidance_scale": flux_guidance_scale,
            "flux.seed": flux_seed,
            "flux.resize_to_multiple_of": flux_resize_to_multiple_of,
            "flux.dilate_mask_iterations": flux_dilate_mask_iterations,
            "flux.mask_kernel_size": flux_mask_kernel_size,
            "flux.enable_model_cpu_offload": flux_enable_model_cpu_offload,
            "flux.enable_attention_slicing": flux_enable_attention_slicing,
            "flux.torch_dtype": flux_torch_dtype,
        },
    )

    return InpainterSettingsPanel(
        inputs=input_components,
        input_names=input_names,
    )


def build_inpainter_tuning_dict(
    input_names: list[str],
    values: list[Any],
) -> dict[str, Any]:
    return {
        name: value
        for name, value in zip(input_names, values)
    }


def add_inputs(
    input_components: list[Any],
    input_names: list[str],
    mapping: dict[str, Any],
) -> None:
    for name, component in mapping.items():
        input_names.append(name)
        input_components.append(component)
