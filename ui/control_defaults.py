# ui/control_defaults.py

from __future__ import annotations


PRESET_CHOICES = ["default", "fast", "balanced", "quality"]
INPAINTER_CHOICES = ["auto", "opencv", "lama", "stable_diffusion", "sdxl", "flux"]
DEVICE_CHOICES = ["auto", "cpu", "cuda", "mps"]
TRACKER_CHOICES = ["none", "optical_flow", "kalman", "xmem", "cotracker"]
AUDIO_BACKEND_CHOICES = ["none", "moviepy", "ffmpeg"]

IMAGE_DEFAULTS = {
    "output_path": "data/outputs/output.png",
    "mask_output_path": "data/masks/mask.png",
    "preset": "balanced",
    "inpainter": "auto",
    "device": "auto",
    "proposal_detectors": ["opencv", "fft", "anomaly"],
    "refiner_detectors": [],
}

VIDEO_DEFAULTS = {
    "output_path": "data/outputs/output.mp4",
    "preset": "fast",
    "inpainter": "opencv",
    "tracker": "optical_flow",
    "device": "auto",
    "audio_backend": "moviepy",
    "proposal_detectors": ["opencv", "fft", "anomaly"],
    "refiner_detectors": [],
    "frame_step": 1,
    "max_gallery_frames": 0,
    "frame_count_status": "No video loaded.",
    "frame_slider_minimum": 0,
    "frame_slider_maximum": 1,
    "frame_slider_value": 0,
    "selected_frame_label": "No frame selected.",
    "pipeline_status": "Pipeline not started.",
}

DETECTION_DEFAULTS = {
    "sensitivity": 50,
    "mask_expand": 2,
    "min_area": 25,
    "fusion_strictness": 50,
}

PREPROCESSING_DEFAULTS = {
    "modules": ["denoise", "gamma", "clahe", "edge_enhance"],
    "resize_width": 0,
    "resize_height": 0,
    "resize_keep_aspect": True,
    "resize_only_downscale": True,
    "denoise_strength": 50,
    "denoise_kernel_size": 5,
    "gamma": 1.0,
    "gain": 1.0,
    "clahe_clip_limit": 2.0,
    "clahe_tile_size": 8,
    "edge_strength": 1.0,
    "fft_strength": 0.7,
    "fft_high_pass_radius": 20,
}

POSTPROCESSING_DEFAULTS = {
    "modules": ["color_matching", "seam_blending", "artifact_removal", "sharpening"],
    "color_match_max_shift": 40,
    "color_match_max_scale": 2.0,
    "seam_strength": 1.0,
    "seam_blur_kernel_size": 21,
    "seam_dilate_iterations": 2,
    "artifact_kernel_size": 3,
    "artifact_dilate_iterations": 1,
    "sharpen_strength": 0.4,
    "sharpen_blur_kernel_size": 5,
    "temporal_alpha": 0.7,
    "temporal_window_size": 3,
}

MASK_PROCESSING_DEFAULTS = {
    "enabled": True,
    "stages": ["box_filter", "fusion", "cleanup"],
    "use_candidate_selection": True,
    "candidate_min_area": 10,
    "candidate_max_area_ratio": 0.35,
    "include_detection_masks": False,
    "include_support_masks": True,
    "fallback_to_detection_mask": True,
    "box_filter_method": "combined",
    "box_filter_support_mode": "none",
    "box_filter_diff_threshold": 0.055,
    "box_filter_adaptive_c": -3.0,
    "box_filter_box_padding": 0,
    "box_filter_min_component_area": 25,
    "box_filter_dilate_iterations": 1,
    "fusion_method": "union",
    "fusion_min_votes": 1,
    "fusion_weighted_threshold": 0.5,
    "cleanup_min_area": 25,
    "cleanup_kernel_size": 3,
    "cleanup_dilate_iterations": 1,
    "cleanup_dilate_kernel_size": 5,
    "cleanup_fill_holes": True,
}

MASK_PROCESSING_STAGE_CHOICES = [
    "box_filter",
    "fusion",
    "cleanup",
]

INPAINTER_DEFAULTS = {
    "pipeline_enabled": True,
    "fallback_enabled": True,
    "opencv.enabled": True,
    "opencv.method": "telea",
    "opencv.radius": 3.0,
    "opencv.dilate_mask_iterations": 1,
    "opencv.mask_kernel_size": 3,
    "lama.enabled": True,
    "lama.model_path": "models/lama/big-lama.pt",
    "lama.device": "auto",
    "lama.modulo": 8,
    "lama.dilate_mask_iterations": 1,
    "lama.mask_kernel_size": 5,
    "lama.check_model_path": True,
    "stable_diffusion.enabled": True,
    "stable_diffusion.model_id": "models/stable-diffusion-inpainting",
    "stable_diffusion.device": "auto",
    "stable_diffusion.prompt": (
        "clean natural image, realistic background, seamless texture, "
        "no watermark, no text, no logo"
    ),
    "stable_diffusion.negative_prompt": (
        "watermark, text, logo, blurry, distorted, artifacts, low quality"
    ),
    "stable_diffusion.num_inference_steps": 30,
    "stable_diffusion.guidance_scale": 7.5,
    "stable_diffusion.strength": 0.85,
    "stable_diffusion.seed": 42,
    "stable_diffusion.resize_to_multiple_of": 8,
    "stable_diffusion.dilate_mask_iterations": 1,
    "stable_diffusion.mask_kernel_size": 5,
    "stable_diffusion.enable_model_cpu_offload": False,
    "stable_diffusion.enable_attention_slicing": True,
    "stable_diffusion.torch_dtype": "float16",
    "sdxl.enabled": True,
    "sdxl.model_id": "models/stable-diffusion-xl-1.0-inpainting-0.1",
    "sdxl.device": "auto",
    "sdxl.prompt": (
        "clean natural image, realistic background, seamless texture, "
        "no watermark, no text, no logo"
    ),
    "sdxl.negative_prompt": (
        "watermark, text, logo, blurry, distorted, artifacts, low quality"
    ),
    "sdxl.num_inference_steps": 30,
    "sdxl.guidance_scale": 7.5,
    "sdxl.strength": 0.85,
    "sdxl.seed": 42,
    "sdxl.resize_to_multiple_of": 8,
    "sdxl.dilate_mask_iterations": 1,
    "sdxl.mask_kernel_size": 5,
    "sdxl.enable_model_cpu_offload": False,
    "sdxl.enable_attention_slicing": True,
    "sdxl.torch_dtype": "float16",
    "flux.enabled": True,
    "flux.model_id": "models/FLUX.1-Fill-dev",
    "flux.device": "auto",
    "flux.prompt": (
        "clean natural background, realistic texture, seamless repair, "
        "no watermark, no text, no logo"
    ),
    "flux.negative_prompt": (
        "watermark, text, logo, artifacts, distortion, blurry, low quality"
    ),
    "flux.num_inference_steps": 40,
    "flux.guidance_scale": 30.0,
    "flux.seed": 42,
    "flux.resize_to_multiple_of": 8,
    "flux.dilate_mask_iterations": 1,
    "flux.mask_kernel_size": 5,
    "flux.enable_model_cpu_offload": False,
    "flux.enable_attention_slicing": True,
    "flux.torch_dtype": "bfloat16",
}

DETECTOR_DEFAULTS = {
    "opencv.mode": "combined",
    "opencv.canny_low": 50,
    "opencv.canny_high": 150,
    "opencv.bright_percentile": 92,
    "opencv.dark_percentile": 8,
    "opencv.min_area": 25,
    "opencv.dilate_iterations": 2,
    "grounding_dino.model_id": "models/grounding-dino-base",
    "grounding_dino.prompt": (
        "watermark . logo . text watermark . transparent watermark . "
        "faint watermark . low opacity watermark"
    ),
    "grounding_dino.box_threshold": 0.25,
    "grounding_dino.text_threshold": 0.20,
    "grounding_dino.min_area": 25,
    "grounding_dino.max_boxes": 20,
    "grounding_dino.strict_loading": False,
    "grounding_dino.fallback_to_empty": True,
    "yolo.confidence": 0.25,
    "yolo.iou": 0.45,
    "yolo.image_size": 640,
    "fft.threshold_percentile": 95,
    "fft.min_area": 25,
    "fft.dilate_iterations": 2,
    "anomaly.threshold": 2.5,
    "anomaly.min_area": 25,
    "anomaly.dilate_iterations": 2,
    "paddle_ocr.confidence_threshold": 0.4,
    "paddle_ocr.min_area": 25,
    "paddle_ocr.dilate_iterations": 2,
    "easy_ocr.confidence_threshold": 0.4,
    "easy_ocr.min_area": 25,
    "easy_ocr.dilate_iterations": 2,
    "sam2.checkpoint_path": "models/sam2/sam2_b.pt",
    "sam2.model_config_path": "models/sam2/sam2_hiera_b+.yaml",
    "sam2.min_area": 50,
    "sam2.mask_threshold": 0.5,
    "sam2.dilate_iterations": 1,
    "sam2.morph_kernel_size": 5,
    "sam2.require_prompts": True,
    "sam2.strict_loading": False,
    "sam2.use_auto_watermark_prompts": True,
    "sam2.use_prompt_text": True,
    "sam2.prompt_text": (
        "dark watermark, light watermark, low opacity watermark, "
        "transparent watermark, text watermark, logo"
    ),
    "sam2.prompt_dark_watermark": True,
    "sam2.prompt_light_watermark": True,
    "sam2.prompt_low_opacity_watermark": True,
    "sam2.prompt_text_watermark": True,
    "sam2.prompt_sensitivity": 50,
    "sam2.prompt_min_area": 25,
    "sam2.prompt_max_boxes": 20,
    "sam2.prompt_max_area_ratio": 0.35,
    "sam2.dark_percentile": 8,
    "sam2.light_percentile": 92,
    "sam2.low_opacity_edge_low": 30,
    "sam2.low_opacity_edge_high": 100,
    "mobile_sam_2.mask_threshold": 0.5,
    "mobile_sam_2.min_area": 25,
    "mobile_sam_2.require_prompts": True,
    "fusion.threshold": 0.5,
    "fusion.min_votes": 2,
}

PREPROCESSING_CHOICES = [
    "resize",
    "denoise",
    "gamma",
    "clahe",
    "edge_enhance",
    "fft_enhance",
]

POSTPROCESSING_CHOICES = [
    "color_matching",
    "seam_blending",
    "artifact_removal",
    "sharpening",
    "temporal_smoothing",
]


def detector_default(name: str) -> object:
    return DETECTOR_DEFAULTS[name]


def inpainter_default(name: str) -> object:
    return INPAINTER_DEFAULTS[name]
