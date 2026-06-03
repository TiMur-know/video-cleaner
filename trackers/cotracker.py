# trackers/cotracker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import importlib
import numpy as np


_MODULE_CACHE: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Missing optional dependency '{module_name}'. "
                f"Install it before using CoTracker."
            ) from exc

    return _MODULE_CACHE[module_name]


@dataclass(slots=True)
class CoTrackerConfig:
    enabled: bool = True

    checkpoint_path: str | None = None
    device: str = "auto"

    # Number of points sampled inside the initial watermark mask.
    num_points: int = 128

    # Radius used to rasterize tracked points back into a mask.
    point_radius: int = 5

    # If True, also draw a convex hull around tracked points.
    fill_hull: bool = True

    # External CoTracker model builder.
    model_factory: Callable[["CoTrackerConfig"], Any] | None = None


class CoTracker:
    """
    CoTracker adapter.

    Purpose:
        Track watermark mask using point tracking.

    Important:
        This is an adapter. The real CoTracker API depends on the package/repo
        you install. Provide either:

            CoTracker(model=your_model)

        or:

            CoTrackerConfig(model_factory=build_model)

    Expected model interfaces:
        Option 1:
            model.track(video, queries)

        Option 2:
            model(video, queries=queries)

    Output:
        A list of binary masks, one per frame.
    """

    name = "cotracker"

    def __init__(
        self,
        config: CoTrackerConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or CoTrackerConfig()
        self.model = model

    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Track initial mask across a frame sequence.

        frames:
            List of frames as NumPy arrays.

        initial_mask:
            Binary mask for frame 0.

        Returns:
            List of masks, same length as frames.
        """
        if not frames:
            return []

        if not self.config.enabled:
            mask = self._to_mask_uint8(initial_mask)
            return [mask.copy() for _ in frames]

        initial_mask = self._to_mask_uint8(initial_mask)

        points = sample_points_from_mask(
            initial_mask,
            num_points=self.config.num_points,
        )

        if len(points) == 0:
            empty = np.zeros(frames[0].shape[:2], dtype=np.uint8)
            return [empty.copy() for _ in frames]

        if self.model is None:
            self.model = self._build_model()

        tracks = self._run_model(frames, points)
        tracks_np = self._to_numpy_tracks(tracks)

        masks = self._tracks_to_masks(
            tracks=tracks_np,
            frame_shape=frames[0].shape[:2],
        )

        return masks

    def _build_model(self) -> Any:
        if self.config.model_factory is not None:
            return self.config.model_factory(self.config)

        raise NotImplementedError(
            "No CoTracker model_factory was provided. "
            "Pass an initialized model to CoTracker(model=...) or provide "
            "CoTrackerConfig(model_factory=...)."
        )

    def _run_model(
        self,
        frames: list[np.ndarray],
        points: np.ndarray,
    ) -> Any:
        """
        Run external CoTracker model.

        points shape:
            N x 2, each point is [x, y]

        Many CoTracker implementations expect queries in this format:
            [frame_index, x, y]
        """
        video = self._prepare_video(frames)
        queries = self._prepare_queries(points)

        if hasattr(self.model, "track"):
            return self.model.track(video, queries)

        if callable(self.model):
            try:
                return self.model(video, queries=queries)
            except TypeError:
                return self.model(video, queries)

        raise TypeError(
            "Unsupported CoTracker model interface. Expected "
            "track(video, queries) or callable model(video, queries)."
        )

    def _prepare_video(self, frames: list[np.ndarray]) -> Any:
        """
        Convert frames into Torch video tensor if torch is available.

        Output shape:
            1 x T x C x H x W

        If torch is unavailable, returns NumPy array:
            1 x T x H x W x C
        """
        video_np = np.stack(frames, axis=0)

        try:
            torch = lazy_import("torch")
        except ImportError:
            return video_np[None, ...]

        if video_np.ndim == 3:
            video_np = video_np[..., None]

        video_tensor = torch.from_numpy(video_np).float()

        # T H W C -> 1 T C H W
        video_tensor = video_tensor.permute(0, 3, 1, 2)[None, ...]

        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0

        device = self._resolve_device(torch)
        video_tensor = video_tensor.to(device)

        return video_tensor

    def _prepare_queries(self, points: np.ndarray) -> Any:
        """
        CoTracker-style queries:
            N x 3 = [start_frame_index, x, y]
        """
        queries_np = np.zeros((points.shape[0], 3), dtype=np.float32)
        queries_np[:, 0] = 0
        queries_np[:, 1:] = points.astype(np.float32)

        try:
            torch = lazy_import("torch")
        except ImportError:
            return queries_np

        queries = torch.from_numpy(queries_np)[None, ...]

        device = self._resolve_device(torch)
        return queries.to(device)

    def _to_numpy_tracks(self, tracks: Any) -> np.ndarray:
        """
        Convert external model output to NumPy array.

        Expected final shape:
            T x N x 2
        """
        torch = None

        try:
            torch = lazy_import("torch")
        except ImportError:
            pass

        if isinstance(tracks, dict):
            for key in ("tracks", "pred_tracks", "points"):
                if key in tracks:
                    tracks = tracks[key]
                    break

        if isinstance(tracks, tuple) or isinstance(tracks, list):
            tracks = tracks[0]

        if torch is not None and hasattr(torch, "Tensor") and isinstance(tracks, torch.Tensor):
            tracks = tracks.detach().cpu().numpy()

        tracks = np.asarray(tracks)

        # Common output: B x T x N x 2
        if tracks.ndim == 4:
            tracks = tracks[0]

        # Common output: T x N x 2
        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise ValueError(
                "Unsupported CoTracker output shape. Expected T x N x 2 "
                f"or B x T x N x 2, got {tracks.shape}"
            )

        return tracks.astype(np.float32)

    def _tracks_to_masks(
        self,
        tracks: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        cv2 = lazy_import("cv2")

        height, width = frame_shape
        masks: list[np.ndarray] = []

        for frame_points in tracks:
            mask = np.zeros((height, width), dtype=np.uint8)

            valid_points = []

            for x, y in frame_points:
                x_int = int(round(float(x)))
                y_int = int(round(float(y)))

                if 0 <= x_int < width and 0 <= y_int < height:
                    valid_points.append([x_int, y_int])

                    cv2.circle(
                        mask,
                        center=(x_int, y_int),
                        radius=self.config.point_radius,
                        color=255,
                        thickness=-1,
                    )

            if self.config.fill_hull and len(valid_points) >= 3:
                points_np = np.array(valid_points, dtype=np.int32)
                hull = cv2.convexHull(points_np)

                cv2.fillConvexPoly(mask, hull, 255)

            masks.append(mask)

        return masks

    def _resolve_device(self, torch: Any) -> str:
        if self.config.device != "auto":
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _to_mask_uint8(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if mask.ndim != 2:
            raise ValueError(f"Mask must be HxW or HxWx1, got {mask.shape}")

        if mask.dtype == np.bool_:
            return mask.astype(np.uint8) * 255

        if mask.dtype == np.uint8:
            if mask.max() <= 1:
                return mask * 255
            return mask

        if mask.max() <= 1.0:
            return (mask > 0.5).astype(np.uint8) * 255

        return (mask > 127).astype(np.uint8) * 255


def sample_points_from_mask(
    mask: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Sample points from inside a binary mask.

    Returns:
        N x 2 array of [x, y]
    """
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    total = len(xs)

    if total <= num_points:
        indices = np.arange(total)
    else:
        indices = np.linspace(0, total - 1, num_points).round().astype(int)

    sampled_xs = xs[indices]
    sampled_ys = ys[indices]

    points = np.stack([sampled_xs, sampled_ys], axis=-1)

    return points.astype(np.float32)