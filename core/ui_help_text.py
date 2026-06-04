# core/ui_help_text.py

from __future__ import annotations

import json
from functools import wraps
from inspect import signature
from typing import Any


HELP_BY_LABEL = {
    "Image": "Open image cleanup controls and previews.",
    "Video": "Open video cleanup controls, frame exploration, and output previews.",
    "Packages": "Check or install packages used by the visual interface.",
    "Config": "Inspect the resolved application configuration.",
    "Preprocessing": "Choose cleanup and enhancement steps that run before watermark detection.",
    "Detection": "Choose detectors and tune how watermark regions are found.",
    "General detection tuning": "Adjust shared detection behavior before detector-specific settings are applied.",
    "Detector settings": "Tune individual proposal, refiner, and fusion detectors.",
    "Mask processing": "Refine detected masks before they are sent to the inpainter.",
    "Inpainter settings": "Tune the backends that repair masked image or video regions.",
    "Postprocessing": "Choose final cleanup steps after inpainting.",
    "Frame explorer": "Preview sampled video frames before running the full pipeline.",
    "Pipeline output": "View the processed video, first output frame, metadata, and status.",
    "Config preview": "View the effective configuration currently used by the app.",
    "Package manager": "Check and install packages needed by the visual UI.",
    "Preprocessing settings": "Tune numeric options for the selected preprocessing stages.",
    "Postprocessing settings": "Tune numeric options for the selected postprocessing stages.",
    "Mask processing settings": "Tune candidate selection, box filtering, fusion, and cleanup for masks.",
    "General inpainter settings": "Control whether inpainting runs and whether failed backends can fall back to another backend.",
    "OpenCV inpainter settings": "Tune the fast local OpenCV repair backend.",
    "LaMa inpainter settings": "Tune the LaMa model-based repair backend.",
    "SDXL inpainter settings": "Tune the SDXL diffusion inpainting backend.",
    "Flux inpainter settings": "Tune the Flux Fill diffusion inpainting backend.",
    "OpenCV proposal detector settings": "Tune classical image-processing detection for likely watermark regions.",
    "GroundingDINO proposal detector settings": "Tune text-prompted object detection for watermark boxes.",
    "YOLO proposal detector settings": "Tune YOLO confidence, overlap filtering, and input size.",
    "FFT proposal detector settings": "Tune frequency-domain detection for repeated or faint watermark patterns.",
    "Anomaly proposal detector settings": "Tune anomaly detection for pixels that differ from the local image structure.",
    "PaddleOCR proposal detector settings": "Tune PaddleOCR text detection for watermark text regions.",
    "EasyOCR proposal detector settings": "Tune EasyOCR text detection for watermark text regions.",
    "SAM2 refiner detector settings": "Tune SAM2 mask refinement from detector prompts.",
    "MobileSAM2 refiner detector settings": "Tune the lightweight SAM2 mask refinement backend.",
    "Fusion settings": "Tune how multiple detector outputs are combined into one mask.",
    "Image input": "Choose the source image that will be processed.",
    "Video input": "Choose the source video that will be processed.",
    "Output image path": "Where the processed image will be saved.",
    "Output video path": "Where the processed video will be saved.",
    "Output mask path": "Where the detected/refined mask preview will be saved.",
    "Preset": "Applies a predefined balance of speed and quality before your manual choices are applied.",
    "Inpainter": "Selects the backend that fills masked watermark regions.",
    "Tracker": "Selects how masks are propagated or stabilized across video frames.",
    "Device": "Selects CPU/GPU acceleration for model-based stages.",
    "Proposal detectors: find possible watermark regions": "First-pass detectors that locate likely watermark, logo, or text regions.",
    "Refiner detectors: improve masks from proposal results": "Second-pass detectors that refine masks using proposal boxes or points.",
    "Detection sensitivity": "Higher values make detectors more willing to keep faint watermark pixels.",
    "Mask expand": "Grows detected masks before later processing and inpainting.",
    "Min area": "Removes detected components smaller than this many pixels.",
    "Fusion strictness": "Controls how much detector agreement is required when combining masks.",
    "Original image": "Preview of the unmodified source image.",
    "After preprocessing": "Preview after preprocessing and before detection.",
    "Detected mask": "Preview of the final mask used for inpainting.",
    "After inpainting": "Preview immediately after the selected inpainter repairs the mask.",
    "After postprocessing": "Preview after final cleanup and smoothing.",
    "Pipeline metadata": "Structured details about selected settings and pipeline results.",
    "Resolved config": "The effective configuration after defaults and config file values are combined.",
    "Package status": "Shows whether optional runtime packages are installed.",
    "Package and model status": "Shows installed and missing runtime packages, model checkpoints, and cached model IDs.",
    "Installed packages": "Shows the tracked Python/system packages that are currently available in this environment.",
    "Installed models and checkpoints": "Shows the tracked local checkpoints and cached Hugging Face models currently available.",
    "Model size estimates": "Shows actual installed model size when detected, or predicted download/cache size for installable models.",
    "Runtime packages": "Select Python or system packages to install or uninstall.",
    "Models and checkpoints": "Select local checkpoints or cached model IDs to install or uninstall.",
    "Package action output": "Shows the result of the latest package or model action.",
    "Audio backend": "Selects how original video audio is copied to the output.",
    "Frame step": "Samples every Nth frame when exploring a video.",
    "Max frames, 0 = all": "Limits how many frames are loaded into the frame explorer.",
    "Frame count / status": "Shows frame extraction progress and status messages.",
    "Video metadata": "Structured metadata discovered while reading the video.",
    "Frame number": "Chooses which extracted frame to preview.",
    "Selected frame": "Preview of the currently selected frame.",
    "Selected frame info": "Details for the currently selected frame.",
    "Output video": "Playable processed video output.",
    "First processed frame": "Preview of the first output frame.",
    "Pipeline status": "Shows current video pipeline state.",
    "Preprocessing stages": "Selects image preparation steps that run before detection.",
    "Resize width, 0 = original": "Resizes input width before detection; zero keeps the original width.",
    "Resize height, 0 = original": "Resizes input height before detection; zero keeps the original height.",
    "Keep aspect ratio": "Preserves source proportions when resizing.",
    "Only downscale": "Prevents resizing from enlarging the image.",
    "Denoise strength": "Controls how strongly image noise is reduced before detection.",
    "Denoise kernel size": "Controls the neighborhood size used by denoising.",
    "Gamma": "Adjusts brightness curve before detection.",
    "Gain": "Multiplies brightness after gamma correction.",
    "CLAHE clip limit": "Limits local contrast amplification.",
    "CLAHE tile size": "Controls the local region size for CLAHE contrast enhancement.",
    "Edge strength": "Controls how strongly edges are emphasized before detection.",
    "FFT strength": "Controls frequency-enhancement intensity for faint patterns.",
    "FFT high-pass radius": "Controls how much low-frequency background is removed.",
    "Mask processing stages": "Selects refinement stages between detection and inpainting.",
    "Enable mask processing": "Turns mask refinement on or off.",
    "Compare candidate masks": "Builds several possible masks and chooses the best scoring candidate.",
    "Candidate min area": "Penalizes candidate masks smaller than this pixel area.",
    "Candidate max area ratio": "Penalizes candidate masks covering more than this fraction of the image.",
    "Include detector masks in fusion": "Allows raw detector masks to participate in final mask fusion.",
    "Include support masks": "Allows auxiliary masks to help mask refinement.",
    "Fallback to detector mask": "Uses the detector mask if refinement cannot produce a mask.",
    "Box filter method": "Chooses how pixels are selected inside detector boxes.",
    "Box support mode": "Controls how support masks interact with box-filter masks.",
    "Box diff threshold": "Higher values keep only stronger local differences.",
    "Box adaptive C": "Adjusts adaptive thresholding inside detector boxes.",
    "Box padding": "Expands detector boxes before filtering pixels inside them.",
    "Box min component area": "Removes small fragments from box-filter output.",
    "Box dilate iterations": "Expands box-filter mask regions.",
    "Mask fusion method": "Chooses how candidate masks are combined.",
    "Fusion min votes": "Minimum mask votes required by vote fusion.",
    "Fusion weighted threshold": "Threshold used by weighted mask fusion.",
    "Cleanup min area": "Removes final mask components smaller than this area.",
    "Cleanup kernel size": "Morphology kernel size for final mask cleanup.",
    "Cleanup dilate iterations": "Expands final mask regions before inpainting.",
    "Cleanup dilate kernel": "Kernel size used when dilating the final mask.",
    "Fill mask holes": "Fills holes inside final mask regions.",
    "Enable inpainting": "Turns the inpainting stage on or off.",
    "Fallback to next backend on error": "Tries another enabled inpainter if the selected one fails.",
    "Enable OpenCV backend": "Allows the fast OpenCV inpainter to run.",
    "OpenCV method": "Chooses the OpenCV inpainting algorithm.",
    "OpenCV radius": "Controls the local neighborhood used by OpenCV inpainting.",
    "OpenCV mask dilate": "Expands the mask before OpenCV inpainting.",
    "OpenCV mask kernel": "Kernel size used for OpenCV mask expansion.",
    "Enable LaMa backend": "Allows the LaMa model inpainter to run.",
    "Check LaMa model path": "Validates the local LaMa checkpoint path before loading.",
    "LaMa model path": "Local path to the LaMa TorchScript checkpoint.",
    "LaMa device": "Device used by the LaMa backend.",
    "LaMa size modulo": "Pads dimensions to a multiple required by LaMa.",
    "LaMa mask dilate": "Expands the mask before LaMa inpainting.",
    "LaMa mask kernel": "Kernel size used for LaMa mask expansion.",
    "Enable SDXL backend": "Allows the SDXL inpainting backend to run.",
    "SDXL model ID": "Diffusers model ID or local path for SDXL inpainting.",
    "SDXL prompt": "Positive prompt describing the desired repaired region.",
    "SDXL negative prompt": "Prompt terms to avoid in the repaired region.",
    "SDXL device": "Device used by the SDXL backend.",
    "SDXL torch dtype": "Tensor precision used by the SDXL backend.",
    "SDXL steps": "Number of denoising steps for SDXL.",
    "SDXL guidance": "How strongly SDXL follows the prompt.",
    "SDXL strength": "How strongly SDXL changes the masked area.",
    "SDXL seed, -1 = random": "Seed for reproducible SDXL output; -1 randomizes it.",
    "SDXL resize multiple": "Pads or resizes dimensions to this multiple for SDXL.",
    "SDXL mask dilate": "Expands the mask before SDXL inpainting.",
    "SDXL mask kernel": "Kernel size used for SDXL mask expansion.",
    "SDXL CPU offload": "Moves SDXL model parts to CPU to reduce GPU memory usage.",
    "SDXL attention slicing": "Reduces SDXL memory usage by slicing attention operations.",
    "Enable Flux backend": "Allows the Flux Fill backend to run.",
    "Flux model ID": "Diffusers model ID or local path for Flux Fill.",
    "Flux prompt": "Positive prompt describing the desired repaired region.",
    "Flux negative prompt": "Prompt terms to avoid in the repaired region.",
    "Flux device": "Device used by the Flux backend.",
    "Flux torch dtype": "Tensor precision used by the Flux backend.",
    "Flux steps": "Number of denoising steps for Flux.",
    "Flux guidance": "How strongly Flux follows the prompt.",
    "Flux seed, -1 = random": "Seed for reproducible Flux output; -1 randomizes it.",
    "Flux resize multiple": "Pads or resizes dimensions to this multiple for Flux.",
    "Flux mask dilate": "Expands the mask before Flux inpainting.",
    "Flux mask kernel": "Kernel size used for Flux mask expansion.",
    "Flux CPU offload": "Moves Flux model parts to CPU to reduce GPU memory usage.",
    "Flux attention slicing": "Reduces Flux memory usage by slicing attention operations.",
    "Postprocessing stages": "Selects cleanup steps that run after inpainting.",
    "Color match max shift": "Limits how far repaired colors can shift toward the original image.",
    "Color match max scale": "Limits color contrast scaling during color matching.",
    "Seam blend strength": "Controls how strongly repaired edges blend into the source image.",
    "Seam blur size": "Controls the softness of mask boundary blending.",
    "Seam dilate iterations": "Expands the seam region before blending.",
    "Artifact kernel size": "Kernel size used for artifact cleanup.",
    "Artifact mask dilate": "Expands the region where artifacts are removed.",
    "Sharpen strength": "Controls final sharpening intensity.",
    "Sharpen blur size": "Blur size used by unsharp masking.",
    "Temporal smoothing alpha": "Controls video frame-to-frame smoothing strength.",
    "Temporal window size": "Window size for temporal averaging mode.",
    "OpenCV mode": "Chooses the OpenCV proposal strategy.",
    "Canny low": "Lower edge threshold for Canny-based OpenCV detection.",
    "Canny high": "Upper edge threshold for Canny-based OpenCV detection.",
    "Bright percentile": "Brightness percentile used to find light watermark regions.",
    "Dark percentile": "Darkness percentile used to find dark watermark or prompt regions.",
    "Model ID": "Model identifier or local path used by this detector.",
    "Prompt": "Text prompt used by text-conditioned detectors.",
    "Box threshold": "Minimum box confidence for GroundingDINO detections.",
    "Text threshold": "Minimum text matching confidence for GroundingDINO.",
    "Max boxes": "Maximum number of boxes kept from this detector.",
    "Strict loading": "Raises errors when model loading fails instead of falling back.",
    "Fallback to empty mask on error": "Returns an empty mask if this detector fails.",
    "Confidence": "Minimum confidence for keeping detections.",
    "IoU": "Overlap threshold used for non-maximum suppression.",
    "Image size": "Detector input image size.",
    "Threshold percentile": "Percentile threshold used for FFT mask extraction.",
    "Threshold": "Generic detector threshold; higher is stricter.",
    "Confidence threshold": "Minimum OCR confidence for keeping text boxes.",
    "Checkpoint path": "Local model checkpoint path.",
    "Model config path": "Local model architecture/config path.",
    "Mask threshold": "Threshold used to binarize model mask probabilities.",
    "Dilate iterations": "Expands detected mask regions.",
    "Morph kernel size": "Kernel size used for mask morphology cleanup.",
    "Require prompts": "Requires boxes or points before running the refiner.",
    "Auto-generate watermark prompts": "Generates prompts from likely watermark cues.",
    "Use editable prompt text": "Uses the editable prompt text field during prompt generation.",
    "SAM2 watermark prompt": "Text describing watermark types to guide prompt generation.",
    "Look for dark watermark": "Adds dark watermark cues to automatic prompt generation.",
    "Look for light watermark": "Adds light watermark cues to automatic prompt generation.",
    "Look for low-opacity watermark": "Adds transparent/faint watermark cues.",
    "Look for text-like watermark": "Adds text-shaped watermark cues.",
    "Prompt sensitivity": "Controls how aggressively automatic prompts are generated.",
    "Prompt min area": "Minimum area for auto-generated prompt regions.",
    "Max prompt boxes": "Maximum number of prompt boxes generated.",
    "Prompt max area ratio": "Rejects prompt boxes larger than this image fraction.",
    "Light percentile": "Percentile used to locate light prompt regions.",
    "Low opacity edge low": "Lower edge threshold for faint watermark prompts.",
    "Low opacity edge high": "Upper edge threshold for faint watermark prompts.",
    "Fusion threshold": "Threshold for detector fusion output.",
    "Min votes": "Minimum detector votes required for fusion.",
    "Run image pipeline": "Processes the selected image with the current detection, mask, inpainting, and postprocessing settings.",
    "Stop image pipeline": "Requests cancellation of the running image pipeline after the current stage checks for stop requests.",
    "Image pipeline status": "Shows current image pipeline state.",
    "Extract frames": "Loads preview frames from the selected video for inspection before running the full pipeline.",
    "Stop frame extraction": "Requests cancellation of the current frame extraction job.",
    "Run video pipeline": "Processes the selected video with the current detection, tracking, inpainting, audio, and output settings.",
    "Stop pipeline": "Requests cancellation of the running video pipeline after the current stage checks for stop requests.",
    "Check visual UI packages": "Checks whether packages needed by the visual interface are installed.",
    "Install missing visual UI packages": "Installs missing packages required by the visual interface.",
    "Refresh package and model status": "Refreshes installed and missing status for packages and models.",
    "Install selected packages": "Installs the selected pip-installable packages.",
    "Uninstall selected packages": "Uninstalls the selected pip packages from the current Python environment.",
    "Install selected models": "Downloads selected Hugging Face models when supported, and reports local checkpoint paths that must be added manually.",
    "Uninstall selected models": "Removes selected local checkpoint files or cached Hugging Face model revisions.",
    "Show resolved config": "Reloads and displays the effective config after defaults and config file values are combined.",
}


EFFECT_HELP_BY_LABEL = {
    "Image": "Use this when you want to tune watermark removal on one still image and inspect every intermediate result.",
    "Video": "Use this when the watermark moves or changes over time; video settings add frame sampling, tracking, audio copy, and stop controls.",
    "Packages": "Use this to see which optional tools are available before choosing heavy detectors, OCR, trackers, or generative inpainters.",
    "Config": "Use this to confirm which defaults and config-file values will actually influence the next run.",
    "Preprocessing": "Preprocessing changes the image before detection so faint text, transparent logos, edges, and contrast differences are easier to notice.",
    "Detection": "Detection decides where the watermark probably is; stronger detection catches more marks but can also mark normal image details.",
    "General detection tuning": "These settings affect all detectors together, so use them first when masks are generally too small, too noisy, or too strict.",
    "Detector settings": "Detector-specific controls help tune one detection method without changing the whole pipeline behavior.",
    "Mask processing": "Mask processing turns rough detector output into a cleaner inpainting mask, reducing missed pixels and false painted areas.",
    "Inpainter settings": "Inpainter settings decide how the masked region is reconstructed; speed, realism, memory use, and consistency all change here.",
    "Postprocessing": "Postprocessing helps repaired pixels blend back into the source by smoothing seams, matching color, and reducing artifacts.",
    "Frame explorer": "Use frame previews to check whether the watermark is visible and stable before spending time on full video processing.",
    "Pipeline output": "Use these outputs to judge whether the chosen settings produced a clean final video or need another tuning pass.",
    "Config preview": "Shows the final resolved settings so you can spot defaults or saved config values that may be changing your result.",
    "Package manager": "Shows missing tools before you select features that depend on them, avoiding pipeline failures halfway through a run.",
    "Preprocessing settings": "Tune how much the image is reshaped, cleaned, brightened, or sharpened before watermark detection.",
    "Postprocessing settings": "Tune how aggressively the repaired result is blended and cleaned after inpainting.",
    "Mask processing settings": "Tune how rough detector masks become precise masks that cover the watermark without covering too much background.",
    "General inpainter settings": "Control whether repair happens at all and whether the app can try another backend if the chosen one fails.",
    "OpenCV inpainter settings": "OpenCV is fast and predictable for small masks, but may look blurry on complex backgrounds.",
    "LaMa inpainter settings": "LaMa can repair larger natural areas better than OpenCV when the local checkpoint is available.",
    "SDXL inpainter settings": "SDXL can create realistic texture in difficult regions but needs more VRAM/time and prompt tuning.",
    "Flux inpainter settings": "Flux can generate high quality fills for hard masks but is slower and heavier than classical methods.",
    "Stable Diffusion inpainter settings": "Stable Diffusion is lighter than SDXL/Flux and can generate textured repairs when OpenCV is too simple.",
    "OpenCV proposal detector settings": "OpenCV settings affect edge, bright, and dark pixel heuristics; useful for simple logos or text without ML models.",
    "GroundingDINO proposal detector settings": "GroundingDINO uses text prompts to find watermark-like regions, useful when classical thresholds miss semantic logos.",
    "YOLO proposal detector settings": "YOLO settings affect learned watermark boxes; useful if you have a checkpoint trained for this type of mark.",
    "FFT proposal detector settings": "FFT settings help find repeated, low-opacity, or frequency-pattern watermarks that are hard to see directly.",
    "Anomaly proposal detector settings": "Anomaly settings look for pixels that differ from local surroundings, useful for subtle overlays but sensitive to texture.",
    "PaddleOCR proposal detector settings": "PaddleOCR settings target text-shaped watermarks; higher confidence reduces false text boxes.",
    "EasyOCR proposal detector settings": "EasyOCR settings target text-shaped watermarks with a different OCR backend; useful for comparing OCR results.",
    "SAM2 refiner detector settings": "SAM2 turns boxes or prompt regions into cleaner object-like masks, improving edge accuracy before inpainting.",
    "MobileSAM2 refiner detector settings": "MobileSAM2 refines masks with a lighter model, trading some accuracy for lower resource use.",
    "Fusion settings": "Fusion controls how multiple detector opinions become one mask; stricter fusion reduces false positives but can miss weak marks.",
    "Image input": "The source image determines what all detectors inspect and what the final repair will be based on.",
    "Video input": "The source video determines the frames, motion, audio source, and tracking behavior used by the video pipeline.",
    "Output image path": "The final repaired image is written here, so choose a path that will not overwrite a result you want to keep.",
    "Output video path": "The rebuilt video is written here, including processed frames and copied audio when enabled.",
    "Output mask path": "Saving the mask lets you inspect exactly what area was removed; use it to tune detection before judging inpainting.",
    "Preset": "Presets quickly shift speed/quality defaults; manual controls still let you override details after choosing one.",
    "Inpainter": "Choose fast local repair for simple marks or heavier generative repair when background texture needs to be recreated.",
    "Tracker": "Tracking reuses or predicts masks across frames; it can reduce flicker but a bad track can drag the mask away from the watermark.",
    "Device": "GPU devices can make model detectors and generative inpainters much faster; CPU is safer but slower.",
    "Proposal detectors: find possible watermark regions": "Proposal detectors create the first rough areas to inspect; selecting more can catch more watermarks but may add noisy masks.",
    "Refiner detectors: improve masks from proposal results": "Refiners improve mask shape from rough proposals, usually making inpainting cleaner around watermark edges.",
    "Detection sensitivity": "Increase this when faint watermarks are missed; lower it when normal image details are being masked.",
    "Mask expand": "Increase this when watermark edges remain after repair; lower it when too much surrounding background gets inpainted.",
    "Min area": "Raise this to remove tiny noisy detections; lower it when small watermark letters or logo fragments disappear.",
    "Fusion strictness": "Raise this when detectors disagree and you want fewer false positives; lower it when weak marks are being missed.",
    "Original image": "Compare against this to see whether later stages removed only the watermark and preserved the source content.",
    "After preprocessing": "Use this preview to see whether preprocessing made watermark patterns easier or accidentally damaged useful details.",
    "Detected mask": "White mask areas are what will be repaired; tune detection until this covers the watermark and avoids clean background.",
    "After inpainting": "This shows the raw repair before final cleanup, helping you decide whether the inpainter or mask is the problem.",
    "After postprocessing": "This is the blended final result; compare it with the inpainted preview to see whether cleanup helped or over-smoothed.",
    "Pipeline metadata": "Metadata shows detector choices, mask sizes, and stage outputs so you can diagnose why a result changed.",
    "Resolved config": "Use this to confirm hidden defaults, saved values, and UI choices before blaming a detector or model.",
    "Package status": "Missing packages explain why certain detectors, OCR engines, trackers, or inpainters cannot run.",
    "Package and model status": "Installed/missing status helps you choose available features and avoid starting a pipeline that cannot load a model.",
    "Installed packages": "This quick list shows which package-backed features can run now without reading the full status table.",
    "Installed models and checkpoints": "This quick list shows which model-backed detectors and inpainters can load assets now.",
    "Model size estimates": "Use this to estimate disk space before installing a model, or to see how much installed model cache/checkpoints occupy.",
    "Runtime packages": "Select packages that enable specific features such as OCR, diffusion inpainting, tracking, or video/audio handling.",
    "Models and checkpoints": "Select model assets used by ML detectors and inpainters; missing checkpoints mean those backends cannot run.",
    "Package action output": "Read this after install/uninstall to see what changed or why a package/model action failed.",
    "Audio backend": "Choose whether and how audio is copied; disabling audio is faster, while moviepy/ffmpeg preserves sound in the result.",
    "Frame step": "Higher values preview fewer frames faster; lower values catch short-lived watermarks and motion changes more accurately.",
    "Max frames, 0 = all": "Limit frames to preview faster on long videos; use all frames when you need a complete inspection.",
    "Frame count / status": "Shows whether frame extraction is still running, finished, or stopped, so you know if previews are complete.",
    "Video metadata": "Frame rate, size, duration, and frame counts help estimate processing time and diagnose output issues.",
    "Frame number": "Move through extracted frames to check whether the watermark changes position, opacity, or shape over time.",
    "Selected frame": "Use this frame to judge detector settings before running the full video pipeline.",
    "Selected frame info": "Shows which sampled frame you are viewing so you can connect preview behavior to video timing.",
    "Output video": "Review this to check final motion, audio, flicker, and whether the watermark stays removed across frames.",
    "First processed frame": "A quick first-frame preview helps catch obvious mask or inpainting problems without opening the full video.",
    "Pipeline status": "Shows whether the video run is idle, running, stopped, or finished.",
    "Preprocessing stages": "Choose stages that make watermark signals clearer before detection; too many stages can also create false anomalies.",
    "Resize width, 0 = original": "Changing width affects detector detail: larger images expose small watermark pixels, while smaller images run faster but can hide tiny marks.",
    "Resize height, 0 = original": "Changing height affects mask precision: higher resolution can reveal faint marks, while lower resolution speeds detection with less detail.",
    "Keep aspect ratio": "Keep this on to avoid stretching; distorted images can make boxes, OCR, and masks less accurate.",
    "Only downscale": "Keep this on when you want speed without inventing pixels; turn it off only if upscaling helps tiny watermark detection.",
    "Denoise strength": "Increase to remove grain that causes false detections; lower it if denoising erases faint watermark strokes.",
    "Denoise kernel size": "Larger kernels smooth wider noise patterns but can blur small text or logo edges detectors need.",
    "Gamma": "Adjust gamma to make dark or light transparent watermarks stand out; extreme values can distort normal image contrast.",
    "Gain": "Gain brightens or darkens after gamma, helping detectors notice low-contrast marks but risking blown-out highlights.",
    "CLAHE clip limit": "Higher values boost local contrast and reveal faint marks; too high can amplify texture into false masks.",
    "CLAHE tile size": "Smaller tiles enhance local watermark contrast; larger tiles produce smoother contrast changes with fewer artifacts.",
    "Edge strength": "Increase to make text/logo borders easier to detect; lower it if normal object edges become false watermarks.",
    "FFT strength": "Increase to emphasize repeating or transparent watermark patterns; lower it if frequency enhancement creates noisy masks.",
    "FFT high-pass radius": "Higher radius removes more broad background variation; tune it when faint overlays blend into lighting gradients.",
    "Mask processing stages": "Choose how rough masks are filtered, combined, and cleaned before repair; this directly affects what gets inpainted.",
    "Enable mask processing": "Turn this on for cleaner masks; turn it off only when detector masks are already accurate or debugging raw detection.",
    "Compare candidate masks": "Builds several mask versions and keeps the best-looking one, reducing bad masks from one aggressive setting.",
    "Candidate min area": "Raise to reject tiny accidental masks; lower when the real watermark is very small.",
    "Candidate max area ratio": "Lower to prevent huge masks from damaging the image; raise only if the watermark covers a large area.",
    "Include detector masks in fusion": "Including raw detector masks can recover missed pixels, but may reintroduce detector noise.",
    "Include support masks": "Support masks can guide refinement toward likely watermark pixels, improving coverage when proposals are rough.",
    "Fallback to detector mask": "Keeps the pipeline useful if refinement fails, but the fallback mask may be rougher.",
    "Box filter method": "Choose how pixels inside detector boxes become masks; residual finds local differences, adaptive handles uneven lighting.",
    "Box support mode": "Intersect makes masks stricter, union makes them broader, and none ignores support masks.",
    "Box diff threshold": "Raise to keep only strong watermark differences; lower to catch faint low-opacity marks.",
    "Box adaptive C": "Controls adaptive threshold bias inside boxes; tune when text is missed on uneven backgrounds.",
    "Box padding": "Increase to include watermark edges outside detector boxes; too much padding can capture clean background.",
    "Box min component area": "Raise to remove specks; lower to keep small letters, dots, or logo fragments.",
    "Box dilate iterations": "Increase to connect broken mask fragments; too much dilation expands repair into clean areas.",
    "Mask fusion method": "Union catches more pixels, intersection is stricter, vote/weighted balance detector agreement.",
    "Fusion min votes": "Higher vote counts reduce false positives but may miss watermarks found by only one detector.",
    "Fusion weighted threshold": "Higher threshold requires stronger combined evidence; lower threshold keeps weaker mask pixels.",
    "Cleanup min area": "Raise to remove leftover mask noise; lower when small watermark pieces are being removed by cleanup.",
    "Cleanup kernel size": "Larger kernels smooth mask shape but can erase small details or merge nearby regions.",
    "Cleanup dilate iterations": "Increase when watermark borders remain visible; lower when repair covers too much nearby content.",
    "Cleanup dilate kernel": "Larger kernels expand masks more broadly around detected pixels, useful for soft watermark edges.",
    "Fill mask holes": "Fill holes so the whole watermark area is repaired instead of leaving untouched islands inside letters or logos.",
    "Enable inpainting": "Turn this on to actually repair masked pixels; off is useful for testing detection and mask quality only.",
    "Fallback to next backend on error": "Keeps a run from failing if a heavy model cannot load, but the fallback may have different quality.",
    "Enable OpenCV backend": "Enable fast local repair for small/simple masks; it is usually less realistic for large textured regions.",
    "OpenCV method": "Telea is often smooth and fast; Navier-Stokes can preserve linear structures differently.",
    "OpenCV radius": "Larger radius samples more surrounding pixels for repair; too large can smear texture.",
    "OpenCV mask dilate": "Expands the mask before OpenCV repair so watermark edges do not remain visible.",
    "OpenCV mask kernel": "Larger kernel expands mask edges more strongly before OpenCV inpainting.",
    "Enable LaMa backend": "Enable LaMa for stronger natural image repair when the local checkpoint exists.",
    "Check LaMa model path": "Keep on to fail early if the checkpoint is missing instead of silently falling back later.",
    "LaMa model path": "This checkpoint controls whether LaMa can run; missing files disable this higher-quality repair option.",
    "LaMa device": "GPU makes LaMa faster; CPU works with less setup but can be slow.",
    "LaMa size modulo": "Pads image dimensions for model compatibility; wrong values can cause model shape errors.",
    "LaMa mask dilate": "Expands the mask before LaMa so soft watermark borders are also reconstructed.",
    "LaMa mask kernel": "Larger kernels make LaMa repair a wider area around detected watermark pixels.",
    "Enable SDXL backend": "Enable SDXL for generative repairs where simple pixel filling cannot recreate the background.",
    "Enable Stable Diffusion backend": "Enable classic Stable Diffusion inpainting for generative repair with lower resource use than SDXL.",
    "Stable Diffusion model ID": "The selected model controls repair style, quality, download size, and memory use.",
    "Stable Diffusion prompt": "Describe the clean background you want so the model fills the watermark area naturally.",
    "Stable Diffusion negative prompt": "List text, logos, artifacts, and blur so the model is less likely to recreate watermark-like content.",
    "Stable Diffusion device": "GPU is recommended; CPU is safer but much slower for Stable Diffusion.",
    "Stable Diffusion torch dtype": "Lower precision saves VRAM and can speed up generation, while float32 is more compatible.",
    "Stable Diffusion steps": "More steps can improve repair detail but take longer; too few steps may leave muddy patches.",
    "Stable Diffusion guidance": "Higher guidance follows the prompt more strongly; too high can make the patch look artificial.",
    "Stable Diffusion strength": "Higher strength changes the masked area more; lower strength preserves more original texture.",
    "Stable Diffusion seed, -1 = random": "Fixed seeds make repair results repeatable while tuning; random seeds explore different fills.",
    "Stable Diffusion resize multiple": "Keeps image dimensions compatible with Stable Diffusion and reduces shape-related errors.",
    "Stable Diffusion mask dilate": "Expands the mask before Stable Diffusion so watermark borders and halos are regenerated.",
    "Stable Diffusion mask kernel": "Larger kernels widen the Stable Diffusion repair region around detected mask pixels.",
    "Stable Diffusion CPU offload": "Uses less VRAM by moving model parts to CPU, usually making generation slower.",
    "Stable Diffusion attention slicing": "Reduces memory pressure during generation, useful for smaller GPUs.",
    "SDXL model ID": "The selected model controls repair style, quality, memory use, and whether downloads/login are needed.",
    "SDXL prompt": "Describe the clean background you want; better prompts reduce logo/text artifacts in generated fills.",
    "SDXL negative prompt": "List things to avoid so the model is less likely to recreate text, logos, artifacts, or blur.",
    "SDXL device": "GPU is strongly recommended; CPU is usually very slow for SDXL.",
    "SDXL torch dtype": "Lower precision saves VRAM and can speed up inference, while float32 is more compatible.",
    "SDXL steps": "More steps can improve detail but take longer; too few steps may leave muddy repairs.",
    "SDXL guidance": "Higher guidance follows the prompt more strongly; too high can create unnatural patches.",
    "SDXL strength": "Higher strength changes the masked area more; lower strength preserves more original texture.",
    "SDXL seed, -1 = random": "Fixed seeds make results repeatable for tuning; random seeds can find a better-looking repair.",
    "SDXL resize multiple": "Keeps dimensions model-friendly; wrong multiples can cause artifacts or shape errors.",
    "SDXL mask dilate": "Expands the mask before SDXL so it replaces watermark borders instead of leaving halos.",
    "SDXL mask kernel": "Larger kernels widen the SDXL repair region around detected watermark pixels.",
    "SDXL CPU offload": "Reduces VRAM use by moving model parts to CPU, usually making generation slower.",
    "SDXL attention slicing": "Reduces VRAM use during attention, often slower but safer on smaller GPUs.",
    "Enable Flux backend": "Enable Flux for high-quality generative repairs when the model and hardware are available.",
    "Flux model ID": "The selected Flux model controls quality, access requirements, download size, and memory use.",
    "Flux prompt": "Describe the clean replacement content so Flux fills the watermark area naturally.",
    "Flux negative prompt": "Tell Flux to avoid text, logos, artifacts, blur, or other unwanted repair patterns.",
    "Flux device": "GPU is strongly recommended; CPU processing will be very slow.",
    "Flux torch dtype": "Lower precision reduces memory use; choose a dtype your device supports.",
    "Flux steps": "More steps can improve repair quality but increase processing time.",
    "Flux guidance": "Higher guidance follows the prompt more strongly; too high can look artificial.",
    "Flux seed, -1 = random": "Fixed seeds repeat a repair for comparison; random seeds explore new fills.",
    "Flux resize multiple": "Keeps dimensions compatible with Flux and reduces shape-related artifacts.",
    "Flux mask dilate": "Expands the mask before Flux so watermark edges are fully regenerated.",
    "Flux mask kernel": "Larger kernels widen the Flux repair area around the detected mask.",
    "Flux CPU offload": "Uses less VRAM by moving model parts to CPU, usually at the cost of speed.",
    "Flux attention slicing": "Reduces memory pressure during generation, useful for smaller GPUs.",
    "Postprocessing stages": "Select final cleanup steps that make repaired pixels match the source and reduce visible seams.",
    "Color match max shift": "Higher values allow repaired colors to move closer to the original area; too high may change color unnaturally.",
    "Color match max scale": "Controls contrast scaling during color matching; higher can fix mismatch but may create harsh contrast.",
    "Seam blend strength": "Increase to hide repair borders; lower it if blending makes the area look washed out.",
    "Seam blur size": "Larger blur softens mask boundaries more; too large can smear nearby details.",
    "Seam dilate iterations": "Expands the seam blending region, useful when a halo remains around the removed watermark.",
    "Artifact kernel size": "Larger kernels remove broader artifacts but can soften real image detail.",
    "Artifact mask dilate": "Expands where artifact cleanup applies; increase when artifacts surround the mask edge.",
    "Sharpen strength": "Increase to restore crispness after inpainting; too much creates halos or noise.",
    "Sharpen blur size": "Controls the scale of sharpening; larger sizes affect broader texture and edges.",
    "Temporal smoothing alpha": "Higher smoothing reduces video flicker between frames but may lag behind motion.",
    "Temporal window size": "Larger windows smooth more frames together, reducing flicker but possibly blurring motion changes.",
    "OpenCV mode": "Auto chooses a heuristic; specific modes let you target edge, bright, dark, or combined watermark types.",
    "Canny low": "Lower values detect weaker edges and faint text; too low adds many false edges.",
    "Canny high": "Higher values keep only strong edges; lower values help with faint watermark outlines.",
    "Bright percentile": "Higher values target only the brightest overlays; lower values catch dimmer light watermarks but add noise.",
    "Dark percentile": "Lower values target only very dark overlays; higher values catch softer dark watermarks but add noise.",
    "Model ID": "Changing the model changes what patterns it can recognize, its download size, speed, and hardware needs.",
    "Prompt": "Better prompts focus text-conditioned detectors on watermark/logo/text regions instead of normal objects.",
    "Box threshold": "Raise to keep only confident boxes; lower to catch weak watermark boxes at the risk of false positives.",
    "Text threshold": "Raise to require stronger prompt matching; lower if the detector misses relevant watermark wording.",
    "Max boxes": "Limit boxes to reduce noisy detections and processing cost; raise when multiple watermarks exist.",
    "Strict loading": "Turn on when you want missing model problems to stop the run instead of falling back silently.",
    "Fallback to empty mask on error": "Keeps the pipeline running after detector failure, but that detector contributes nothing to the mask.",
    "Confidence": "Raise to keep only strong detections; lower when trained models miss faint or small watermarks.",
    "IoU": "Controls overlap filtering; lower removes duplicate boxes aggressively, higher keeps overlapping candidates.",
    "Image size": "Larger input size helps detect small marks but costs memory and time; smaller is faster with less detail.",
    "Threshold percentile": "Higher keeps only strongest FFT mask pixels; lower catches faint patterns but adds noise.",
    "Threshold": "Higher makes this detector stricter; lower makes it more sensitive to faint anomalies.",
    "Confidence threshold": "Raise to avoid false OCR text boxes; lower when real watermark text is faint or stylized.",
    "Checkpoint path": "The checkpoint determines whether this local model can run and what it has learned to segment.",
    "Model config path": "The config must match the checkpoint architecture or the model may fail or produce bad masks.",
    "Mask threshold": "Higher keeps only confident mask pixels; lower fills more uncertain regions around the watermark.",
    "Dilate iterations": "Increase to cover watermark borders and broken mask gaps; too much expands into clean content.",
    "Morph kernel size": "Larger morphology smooths and joins masks, but can remove fine text details.",
    "Require prompts": "Keeps refiners from guessing without boxes/points; disable only if the refiner can safely auto-prompt.",
    "Auto-generate watermark prompts": "Generates prompt regions automatically, helping SAM-style refiners when proposals are weak.",
    "Use editable prompt text": "Lets your prompt text guide automatic prompt generation toward the watermark type you expect.",
    "SAM2 watermark prompt": "Describe the watermark appearance so prompt generation searches for matching dark, light, text, or logo regions.",
    "Look for dark watermark": "Adds dark overlay cues, useful for black or shadow-like text and logos.",
    "Look for light watermark": "Adds light overlay cues, useful for white or bright transparent logos.",
    "Look for low-opacity watermark": "Adds faint edge cues, useful when the watermark is transparent and blends into texture.",
    "Look for text-like watermark": "Adds text-shape cues so prompt generation pays attention to letters and word-like regions.",
    "Prompt sensitivity": "Higher generates more prompt boxes for faint marks; lower reduces false prompt regions.",
    "Prompt min area": "Raise to ignore tiny prompt noise; lower to keep small letters or logo pieces.",
    "Max prompt boxes": "Higher allows more candidate watermark regions; lower keeps SAM refinement faster and less noisy.",
    "Prompt max area ratio": "Lower rejects huge prompt boxes that would damage large background areas.",
    "Light percentile": "Higher targets only very light marks; lower catches softer light overlays but can add false positives.",
    "Low opacity edge low": "Lower values detect weaker edges in transparent marks; too low adds background texture.",
    "Low opacity edge high": "Higher values require stronger faint-mark edges; lower keeps more soft outlines.",
    "Fusion threshold": "Higher requires stronger fused mask confidence; lower keeps more uncertain watermark pixels.",
    "Min votes": "More votes means more detector agreement; fewer votes lets one detector preserve faint marks.",
    "Run image pipeline": "Starts image cleanup using current settings; run after the mask preview settings look reasonable.",
    "Stop image pipeline": "Stops after the current stage check, useful when a model is slow or settings are clearly wrong.",
    "Image pipeline status": "Watch this to know whether the image run finished, stopped, or is waiting.",
    "Extract frames": "Builds previews so you can tune detection on representative frames before processing the whole video.",
    "Stop frame extraction": "Stops preview loading when enough frames are visible or the video is too long.",
    "Run video pipeline": "Starts full video cleanup; use frame preview first to avoid wasting time on bad masks.",
    "Stop pipeline": "Stops after the current video stage check, preserving any partial metadata/results available.",
    "Refresh package and model status": "Refresh after installing or deleting tools so the UI reflects what can run now.",
    "Install selected packages": "Installs selected Python packages so their detectors, inpainters, trackers, or model tools become available.",
    "Uninstall selected packages": "Removes selected packages; features depending on them will stop working until reinstalled.",
    "Install selected models": "Downloads cacheable Hugging Face models; local checkpoints still need to be placed at their expected paths.",
    "Uninstall selected models": "Frees disk space by removing selected local checkpoints or cached Hugging Face model revisions.",
    "Show resolved config": "Refresh this after changing config files to see what values the next run will use.",
}

HELP_BY_LABEL.update(EFFECT_HELP_BY_LABEL)


def add_editable_object_effects() -> None:
    for label, help_text in list(HELP_BY_LABEL.items()):
        object_effect = editable_object_effect_for_label(label)

        if object_effect and "Effect on editable object:" not in help_text:
            HELP_BY_LABEL[label] = (
                f"{help_text}\n"
                f"Effect on editable object: {object_effect}"
            )


def editable_object_effect_for_label(label: str) -> str:
    if label in {
        "Image input",
        "Original image",
        "Video input",
        "Selected frame",
    }:
        return "This is the source content that all later masks, repairs, and previews are based on."

    if label in {
        "Output image path",
        "Output video path",
        "Output mask path",
        "Output video",
        "First processed frame",
        "After postprocessing",
    }:
        return "This does not change detection itself, but it controls where or how the edited result can be reviewed."

    if label in {
        "Preprocessing",
        "Preprocessing settings",
        "Preprocessing stages",
        "Resize width, 0 = original",
        "Resize height, 0 = original",
        "Keep aspect ratio",
        "Only downscale",
        "Denoise strength",
        "Denoise kernel size",
        "Gamma",
        "Gain",
        "CLAHE clip limit",
        "CLAHE tile size",
        "Edge strength",
        "FFT strength",
        "FFT high-pass radius",
        "After preprocessing",
    }:
        return "It changes the temporary analysis image/video frames before detection, which can make watermark pixels easier or harder to find."

    if label in {
        "Detection",
        "General detection tuning",
        "Detector settings",
        "Proposal detectors: find possible watermark regions",
        "Refiner detectors: improve masks from proposal results",
        "Detection sensitivity",
        "Mask expand",
        "Min area",
        "Fusion strictness",
        "OpenCV mode",
        "Canny low",
        "Canny high",
        "Bright percentile",
        "Dark percentile",
        "Model ID",
        "Prompt",
        "Box threshold",
        "Text threshold",
        "Max boxes",
        "Strict loading",
        "Fallback to empty mask on error",
        "Confidence",
        "IoU",
        "Image size",
        "Threshold percentile",
        "Threshold",
        "Confidence threshold",
        "Checkpoint path",
        "Model config path",
        "Mask threshold",
        "Require prompts",
        "Auto-generate watermark prompts",
        "Use editable prompt text",
        "SAM2 watermark prompt",
        "Look for dark watermark",
        "Look for light watermark",
        "Look for low-opacity watermark",
        "Look for text-like watermark",
        "Prompt sensitivity",
        "Prompt min area",
        "Max prompt boxes",
        "Prompt max area ratio",
        "Light percentile",
        "Low opacity edge low",
        "Low opacity edge high",
        "Fusion threshold",
        "Min votes",
        "Detected mask",
    }:
        return "It changes which pixels are considered watermark, directly controlling the mask that will be removed from the editable image/video."

    if label in {
        "Mask processing",
        "Mask processing settings",
        "Mask processing stages",
        "Enable mask processing",
        "Compare candidate masks",
        "Candidate min area",
        "Candidate max area ratio",
        "Include detector masks in fusion",
        "Include support masks",
        "Fallback to detector mask",
        "Box filter method",
        "Box support mode",
        "Box diff threshold",
        "Box adaptive C",
        "Box padding",
        "Box min component area",
        "Box dilate iterations",
        "Mask fusion method",
        "Fusion min votes",
        "Fusion weighted threshold",
        "Cleanup min area",
        "Cleanup kernel size",
        "Cleanup dilate iterations",
        "Cleanup dilate kernel",
        "Fill mask holes",
        "Dilate iterations",
        "Morph kernel size",
    }:
        return "It reshapes the removal mask before repair, affecting how much real content is protected or overwritten."

    if label in {
        "Inpainter",
        "Inpainter settings",
        "General inpainter settings",
        "Enable inpainting",
        "Fallback to next backend on error",
        "OpenCV inpainter settings",
        "Enable OpenCV backend",
        "OpenCV method",
        "OpenCV radius",
        "OpenCV mask dilate",
        "OpenCV mask kernel",
        "LaMa inpainter settings",
        "Enable LaMa backend",
        "Check LaMa model path",
        "LaMa model path",
        "LaMa device",
        "LaMa size modulo",
        "LaMa mask dilate",
        "LaMa mask kernel",
        "Stable Diffusion inpainter settings",
        "Enable Stable Diffusion backend",
        "Stable Diffusion model ID",
        "Stable Diffusion prompt",
        "Stable Diffusion negative prompt",
        "Stable Diffusion device",
        "Stable Diffusion torch dtype",
        "Stable Diffusion steps",
        "Stable Diffusion guidance",
        "Stable Diffusion strength",
        "Stable Diffusion seed, -1 = random",
        "Stable Diffusion resize multiple",
        "Stable Diffusion mask dilate",
        "Stable Diffusion mask kernel",
        "Stable Diffusion CPU offload",
        "Stable Diffusion attention slicing",
        "SDXL inpainter settings",
        "Enable SDXL backend",
        "SDXL model ID",
        "SDXL prompt",
        "SDXL negative prompt",
        "SDXL device",
        "SDXL torch dtype",
        "SDXL steps",
        "SDXL guidance",
        "SDXL strength",
        "SDXL seed, -1 = random",
        "SDXL resize multiple",
        "SDXL mask dilate",
        "SDXL mask kernel",
        "SDXL CPU offload",
        "SDXL attention slicing",
        "Flux inpainter settings",
        "Enable Flux backend",
        "Flux model ID",
        "Flux prompt",
        "Flux negative prompt",
        "Flux device",
        "Flux torch dtype",
        "Flux steps",
        "Flux guidance",
        "Flux seed, -1 = random",
        "Flux resize multiple",
        "Flux mask dilate",
        "Flux mask kernel",
        "Flux CPU offload",
        "Flux attention slicing",
        "After inpainting",
    }:
        return "It changes how masked pixels are rebuilt, affecting texture, realism, speed, memory use, and whether watermark edges remain."

    if label in {
        "Postprocessing",
        "Postprocessing settings",
        "Postprocessing stages",
        "Color match max shift",
        "Color match max scale",
        "Seam blend strength",
        "Seam blur size",
        "Seam dilate iterations",
        "Artifact kernel size",
        "Artifact mask dilate",
        "Sharpen strength",
        "Sharpen blur size",
        "Temporal smoothing alpha",
        "Temporal window size",
    }:
        return "It changes the repaired pixels after inpainting so the edit blends better with surrounding image/video content."

    if label in {
        "Tracker",
        "Frame explorer",
        "Frame step",
        "Max frames, 0 = all",
        "Frame number",
        "Audio backend",
    }:
        return "It affects video editing behavior across frames rather than one still mask, changing consistency, speed, or output media."

    if label in {
        "Preset",
        "Device",
        "Run image pipeline",
        "Run video pipeline",
        "Stop image pipeline",
        "Stop pipeline",
        "Stop frame extraction",
    }:
        return "It changes how the edit is executed, which can affect processing time, available models, and final quality."

    return "This does not directly repaint pixels, but it affects setup, diagnosis, preview, or which editing path can run."


add_editable_object_effects()


def apply_ui_help_text(gr: Any) -> None:
    blocks = getattr(gr, "Blocks", None)

    if blocks is not None:
        patch_blocks_help_script(blocks)

    for component_name in [
        "File",
        "Textbox",
        "Dropdown",
        "Checkbox",
        "CheckboxGroup",
        "Slider",
        "Number",
        "Image",
        "Video",
        "JSON",
    ]:
        component = getattr(gr, component_name, None)

        if component is not None:
            patch_component_info(component)

    button = getattr(gr, "Button", None)

    if button is not None:
        patch_button_tooltip(button)


def patch_component_info(component: Any) -> None:
    if getattr(component, "_watermwark_help_patched", False):
        return

    try:
        accepts_info = "info" in signature(component.__init__).parameters
    except (TypeError, ValueError):
        accepts_info = True

    if not accepts_info:
        return

    original_init = component.__init__

    @wraps(original_init)
    def init_with_help(self: Any, *args: Any, **kwargs: Any) -> None:
        label = kwargs.get("label")

        if label is None and args:
            label = args[0]

        if "info" not in kwargs and isinstance(label, str):
            kwargs["info"] = help_text_for_label(label)

        original_init(self, *args, **kwargs)

    component.__init__ = init_with_help
    component._watermwark_help_patched = True


def patch_button_tooltip(component: Any) -> None:
    if getattr(component, "_watermwark_help_patched", False):
        return

    try:
        parameters = signature(component.__init__).parameters
    except (TypeError, ValueError):
        parameters = {}

    help_parameter = first_supported_parameter(
        parameters,
        [
            "tooltip",
            "info",
        ],
    )

    if help_parameter is None:
        return

    original_init = component.__init__

    @wraps(original_init)
    def init_with_help(self: Any, *args: Any, **kwargs: Any) -> None:
        label = kwargs.get("value")

        if label is None and args:
            label = args[0]

        if help_parameter not in kwargs and isinstance(label, str):
            kwargs[help_parameter] = help_text_for_label(label)

        original_init(self, *args, **kwargs)

    component.__init__ = init_with_help
    component._watermwark_help_patched = True


def patch_blocks_help_script(component: Any) -> None:
    if getattr(component, "_watermwark_help_patched", False):
        return

    try:
        parameters = signature(component.__init__).parameters
    except (TypeError, ValueError):
        parameters = {}

    if parameters and "js" not in parameters:
        return

    original_init = component.__init__

    @wraps(original_init)
    def init_with_help(self: Any, *args: Any, **kwargs: Any) -> None:
        if "js" not in kwargs:
            kwargs["js"] = ui_help_script()

        original_init(self, *args, **kwargs)

    component.__init__ = init_with_help
    component._watermwark_help_patched = True


def component_help(component: Any, label: str) -> dict[str, str]:
    """
    Build explicit help kwargs for components created outside the auto patch.
    """
    return help_kwargs_for_component(component, label, ["info"])


def button_help(component: Any, label: str) -> dict[str, str]:
    """
    Build explicit help kwargs for Gradio buttons when the installed version
    supports a button help parameter.
    """
    return help_kwargs_for_component(component, label, ["tooltip", "info"])


def help_kwargs_for_component(
    component: Any,
    label: str,
    help_parameters: list[str],
) -> dict[str, str]:
    try:
        parameters = signature(component.__init__).parameters
    except (TypeError, ValueError):
        parameters = {}

    help_parameter = first_supported_parameter(parameters, help_parameters)

    if help_parameter is None:
        return {}

    return {
        help_parameter: help_text_for_label(label),
    }


def first_supported_parameter(
    parameters: Any,
    names: list[str],
) -> str | None:
    if not parameters:
        return names[0]

    for name in names:
        if name in parameters:
            return name

    return None


def ui_help_script() -> str:
    help_by_label = json.dumps(HELP_BY_LABEL, sort_keys=True)

    return f"""
() => {{
  const helpByLabel = {help_by_label};

  const normalizeText = (value) => (value || "").replace(/\\s+/g, " ").trim();

  const applyTitles = () => {{
    const selector = [
      "button",
      "label",
      ".label-wrap",
      ".tab-nav button",
      ".prose h2",
      ".prose h3"
    ].join(",");

    document.querySelectorAll(selector).forEach((element) => {{
      const text = normalizeText(element.innerText || element.textContent);
      const helpText = helpByLabel[text];

      if (helpText && element.title !== helpText) {{
        element.title = helpText;
      }}
    }});
  }};

  applyTitles();

  new MutationObserver(applyTitles).observe(document.body, {{
    childList: true,
    subtree: true
  }});
}}
"""


def help_text_for_label(label: str) -> str:
    if label in HELP_BY_LABEL:
        return HELP_BY_LABEL[label]

    normalized = label.rstrip(".:")
    return f"Configures {normalized} for this pipeline step."


__all__ = [
    "HELP_BY_LABEL",
    "apply_ui_help_text",
    "button_help",
    "component_help",
    "help_text_for_label",
    "help_kwargs_for_component",
    "patch_blocks_help_script",
    "patch_component_info",
    "patch_button_tooltip",
    "ui_help_script",
]
