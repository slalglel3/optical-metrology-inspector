"""Data models and enumerations for the Optical Metrology Outlier Inspector."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class DefectType(str, Enum):
    """요구사항 2.5 및 2.6에 정의된 오계측 및 광학 이상 유형."""
    POSITION_MISMATCH = "계측 위치 불일치"
    LINE_TRUNCATION = "선분 단절 및 결손"
    LINE_BLOATING = "선폭 팽창 및 뭉침"
    INTERNAL_FRAGMENTATION = "내부 파편화"
    ADJACENT_INTRUSION = "인접 영역 침범"
    PATTERN_DISTORTION = "패턴 왜곡"
    PATTERN_SHRINKAGE_DOT = "패턴 축소 및 점 찍힘"
    OPTICAL_DEFOCUS = "광학 초점 불량"
    UNMEASURABLE_RESCUED = "계측 요소 소실(영역 구제)"


@dataclass
class BoundingBox:
    """2D Bounding Box (x1, y1, x2, y2) inclusive-exclusive in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def pad(self, pad_x: int, pad_y: int, max_w: int, max_h: int) -> BoundingBox:
        """Return a padded bounding box clamped to image boundaries [0, max_w] x [0, max_h]."""
        return BoundingBox(
            x1=max(0, self.x1 - pad_x),
            y1=max(0, self.y1 - pad_y),
            x2=min(max_w, self.x2 + pad_x),
            y2=min(max_h, self.y2 + pad_y),
        )

    def union(self, other: BoundingBox) -> BoundingBox:
        """Return the minimal bounding box containing both boxes (anti-clipping guard)."""
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def iou(self, other: BoundingBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union_area = self.area + other.area - inter
        return inter / float(max(1, union_area))

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class FocusResult:
    """광학 초점 불량(Defocus) 독립 병렬 검정 결과 (요구사항 2.5)."""
    sharpness_score: float
    tenengrad_score: float
    laplacian_score: float
    relative_to_cluster: float = 1.0
    robust_z_score: float = 0.0
    is_defocused: bool = False
    explanation: str = "정상 초점"


@dataclass
class SegmentFeature:
    """단일 계측 세그먼트(선분, 패드, 곡선 호 조각 등)의 기하·위상·광학 정합 피처."""
    segment_id: int
    bbox: BoundingBox
    rel_cx: float
    rel_cy: float
    area: float
    span_length: float
    skeleton_length: float
    mean_thickness: float
    max_thickness: float
    aspect_ratio: float
    orientation_deg: float
    sinuosity_std: float
    max_axis_deviation: float
    orthogonal_intrusion_ratio: float
    fragment_count_in_corridor: int
    underlying_wafer_intensity: float
    underlying_wafer_gradient: float
    is_curved_arc: bool = False


@dataclass
class SiteFeature:
    """단일 계측 위치(Site / Sub-ROI)의 정규화 영역 및 종합 피처."""
    site_id: int
    raw_bbox: Optional[BoundingBox]
    normalized_bbox: BoundingBox
    is_rescued: bool
    anchor_cx: float
    anchor_cy: float
    anchor_w: float
    anchor_h: float
    total_red_area: float
    total_cyan_area: float
    total_skeleton_length: float
    mean_thickness: float
    max_thickness: float
    span_x: float
    span_y: float
    rel_centroid_x: float
    rel_centroid_y: float
    angular_sector_areas: List[float] = field(default_factory=list)
    quadrant_areas: List[float] = field(default_factory=list)
    symmetry_score: float = 1.0
    intrinsic_integrity_score: float = 1.0
    segments: List[SegmentFeature] = field(default_factory=list)


@dataclass
class DefectFinding:
    """검출된 개별 오계측 결함 상세 항목."""
    defect_type: DefectType
    severity_score: float
    site_id: int
    region_bbox: Optional[BoundingBox]
    description: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ImageInspectionResult:
    """단일 이미지에 대한 최종 오계측 검정 및 진단 결과 (요구사항 3)."""
    image_path: str
    image_name: str
    cluster_id: str
    width: int
    height: int
    is_outlier: bool
    status_label: str
    focus_result: FocusResult
    sites: List[SiteFeature] = field(default_factory=list)
    defects: List[DefectFinding] = field(default_factory=list)
    primary_defect: Optional[str] = None
    defect_types: List[str] = field(default_factory=list)
    summary_explanation: str = "정상 계측 (이상 없음)"
    reference_image_name: Optional[str] = None
    reference_image_path: Optional[str] = None
    diagnostic_image_path: Optional[str] = None
    comparison_image_path: Optional[str] = None
