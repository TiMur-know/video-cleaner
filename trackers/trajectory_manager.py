# trackers/trajectory_manager.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class TrajectoryManagerConfig:
    enabled: bool = True

    # Minimum mask area to become a valid track.
    min_area: int = 20

    # IoU needed to associate a new mask with an existing trajectory.
    iou_threshold: float = 0.2

    # How many frames a track can disappear before removal.
    max_missing: int = 5

    # Exponential smoothing for bbox.
    # Higher = more stable but slower to react.
    smoothing_alpha: float = 0.7

    # Return tracks that are currently visible only.
    visible_only: bool = True


@dataclass(slots=True)
class TrajectoryState:
    track_id: int
    bbox: BBox
    mask: np.ndarray
    confidence: float
    last_frame_index: int

    age: int = 1
    missing_count: int = 0
    history: list[BBox] = field(default_factory=list)

    @property
    def is_visible(self) -> bool:
        return self.missing_count == 0


class TrajectoryManager:
    """
    Manages watermark mask trajectories across video frames.

    Purpose:
        - assign stable track IDs
        - smooth bounding boxes
        - tolerate missing detections
        - choose active masks for inpainting

    Typical use:
        detections on current frame
            ↓
        trajectory_manager.update(frame_index, masks)
            ↓
        stable trajectories
    """

    name = "trajectory_manager"

    def __init__(
        self,
        config: TrajectoryManagerConfig | None = None,
    ) -> None:
        self.config = config or TrajectoryManagerConfig()
        self.tracks: dict[int, TrajectoryState] = {}
        self._next_track_id = 1

    def update(
        self,
        frame_index: int,
        masks: list[np.ndarray],
        confidences: list[float] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[TrajectoryState]:
        if not self.config.enabled:
            return []

        confidences = confidences or [1.0] * len(masks)

        detections = self._prepare_detections(masks, confidences)

        matched_track_ids: set[int] = set()
        matched_detection_ids: set[int] = set()

        matches = self._match_detections_to_tracks(detections)

        for track_id, detection_id in matches:
            detection = detections[detection_id]
            self._update_track(
                track_id=track_id,
                bbox=detection["bbox"],
                mask=detection["mask"],
                confidence=detection["confidence"],
                frame_index=frame_index,
            )

            matched_track_ids.add(track_id)
            matched_detection_ids.add(detection_id)

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                self.tracks[track_id].missing_count += 1

        for detection_id, detection in enumerate(detections):
            if detection_id not in matched_detection_ids:
                self._create_track(
                    bbox=detection["bbox"],
                    mask=detection["mask"],
                    confidence=detection["confidence"],
                    frame_index=frame_index,
                )

        self._remove_dead_tracks()

        return self.get_tracks()

    def get_tracks(self) -> list[TrajectoryState]:
        tracks = list(self.tracks.values())

        if self.config.visible_only:
            tracks = [track for track in tracks if track.is_visible]

        return sorted(tracks, key=lambda track: track.track_id)

    def get_active_masks(self) -> list[np.ndarray]:
        return [track.mask for track in self.get_tracks()]

    def get_primary_track(self) -> TrajectoryState | None:
        tracks = self.get_tracks()

        if not tracks:
            return None

        return max(tracks, key=lambda track: self._bbox_area(track.bbox))

    def reset(self) -> None:
        self.tracks.clear()
        self._next_track_id = 1

    def _prepare_detections(
        self,
        masks: list[np.ndarray],
        confidences: list[float],
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []

        for mask, confidence in zip(masks, confidences):
            mask_uint8 = self._to_mask_uint8(mask)
            bbox = mask_to_bbox(mask_uint8)

            if bbox is None:
                continue

            if self._bbox_area(bbox) < self.config.min_area:
                continue

            detections.append(
                {
                    "mask": mask_uint8,
                    "bbox": bbox,
                    "confidence": float(confidence),
                }
            )

        return detections

    def _match_detections_to_tracks(
        self,
        detections: list[dict[str, Any]],
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            for detection_id, detection in enumerate(detections):
                score = bbox_iou(track.bbox, detection["bbox"])

                if score >= self.config.iou_threshold:
                    candidates.append((score, track_id, detection_id))

        candidates.sort(reverse=True, key=lambda item: item[0])

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: list[tuple[int, int]] = []

        for score, track_id, detection_id in candidates:
            if track_id in used_tracks:
                continue

            if detection_id in used_detections:
                continue

            matches.append((track_id, detection_id))
            used_tracks.add(track_id)
            used_detections.add(detection_id)

        return matches

    def _create_track(
        self,
        bbox: BBox,
        mask: np.ndarray,
        confidence: float,
        frame_index: int,
    ) -> None:
        track_id = self._next_track_id
        self._next_track_id += 1

        self.tracks[track_id] = TrajectoryState(
            track_id=track_id,
            bbox=bbox,
            mask=mask,
            confidence=confidence,
            last_frame_index=frame_index,
            history=[bbox],
        )

    def _update_track(
        self,
        track_id: int,
        bbox: BBox,
        mask: np.ndarray,
        confidence: float,
        frame_index: int,
    ) -> None:
        track = self.tracks[track_id]

        smoothed_bbox = self._smooth_bbox(track.bbox, bbox)

        track.bbox = smoothed_bbox
        track.mask = mask
        track.confidence = confidence
        track.last_frame_index = frame_index
        track.age += 1
        track.missing_count = 0
        track.history.append(smoothed_bbox)

    def _smooth_bbox(self, old_bbox: BBox, new_bbox: BBox) -> BBox:
        alpha = self.config.smoothing_alpha

        x1 = round(alpha * old_bbox[0] + (1.0 - alpha) * new_bbox[0])
        y1 = round(alpha * old_bbox[1] + (1.0 - alpha) * new_bbox[1])
        x2 = round(alpha * old_bbox[2] + (1.0 - alpha) * new_bbox[2])
        y2 = round(alpha * old_bbox[3] + (1.0 - alpha) * new_bbox[3])

        return int(x1), int(y1), int(x2), int(y2)

    def _remove_dead_tracks(self) -> None:
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].missing_count > self.config.max_missing:
                del self.tracks[track_id]

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

        return (mask > 0).astype(np.uint8) * 255

    @staticmethod
    def _bbox_area(bbox: BBox) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def mask_to_bbox(mask: np.ndarray) -> BBox | None:
    """
    Convert binary mask to bbox.

    Returns:
        x1, y1, x2, y2

    x2 and y2 are exclusive.
    """
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union