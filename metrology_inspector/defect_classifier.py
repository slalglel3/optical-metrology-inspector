"""Cluster-Relative Defect Classifier and Outlier Judge (요구사항 2.6 & 2.7).

동일 조건의 이미지 군집 내 정상 기준(Golden Baseline)을 강건 통계(Median/MAD) 및 내재적 무결성 점수로 수립하고,
7대 오계측 결함 유형 및 광학 초점 불량을 정밀 분류하여 설명 가능한 판정 결과를 생성합니다.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from .models import (
    BoundingBox,
    DefectFinding,
    DefectType,
    FocusResult,
    ImageInspectionResult,
    SegmentFeature,
    SiteFeature,
)


class ClusterDefectClassifier:
    """군집 비교 및 기하·광학 정합 분석을 통해 7대 오계측 결함을 분류합니다."""

    @staticmethod
    def select_golden_reference_index(
        sites_per_image: List[List[SiteFeature]],
        focus_results: List[FocusResult],
    ) -> int:
        """군집 내에서 가장 완전하고 대칭적이며 초점이 선명한 대표 정상 이미지(Golden Reference)를 선정합니다."""
        num_images = len(sites_per_image)
        if num_images == 0:
            return 0
        if num_images == 1:
            return 0

        # 각 이미지의 종합 품질 점수 계산:
        # 1) 비구제 정상 검출 여부, 2) 광학 초점 정상 여부, 3) 사이트 평균 내재적 무결성 점수, 4) 총 계측 면적 중앙값 근접도
        total_areas = np.array([
            sum(s.total_red_area for s in sites)
            for sites in sites_per_image
        ], dtype=np.float64)

        target_area = float(np.max(total_areas)) if num_images <= 3 else float(np.median(total_areas))

        best_idx = 0
        best_score = -1e9

        for idx in range(num_images):
            sites = sites_per_image[idx]
            focus = focus_results[idx]
            if not sites:
                continue

            rescued_penalty = -2.0 if any(s.is_rescued for s in sites) else 0.0
            defocus_penalty = -1.5 if focus.is_defocused else 0.0
            mean_integrity = float(np.mean([s.intrinsic_integrity_score for s in sites]))

            # 총 검출 면적이 결손되지 않고 완전한지 평가
            cur_area = total_areas[idx]
            area_score = min(1.0, cur_area / max(1.0, target_area))
            if num_images >= 4 and cur_area > target_area * 1.45:
                # 과도한 팽창/노이즈 이미지는 대표 정상에서 배제
                area_score -= 0.5

            total_score = (
                rescued_penalty
                + defocus_penalty
                + 0.72 * mean_integrity
                + 0.20 * area_score
                + 0.08 * min(1.0, focus.relative_to_cluster)
            )
            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        return best_idx

    @classmethod
    def _match_segments(
        cls,
        ref_segments: List[SegmentFeature],
        cur_segments: List[SegmentFeature],
        ref_site: SiteFeature,
        cur_site: SiteFeature,
    ) -> Tuple[List[Tuple[SegmentFeature, Optional[SegmentFeature]]], List[SegmentFeature]]:
        """정상 기준 사이트의 세그먼트들과 현재 사이트의 세그먼트들을 최소 비용 이분 매칭(Minimum Distance)으로 1:1 매칭합니다."""
        if not ref_segments:
            return [], list(cur_segments)

        # 모든 (r_idx, c_idx) 쌍의 매칭 비용(위치 거리 + 방향 차이) 계산
        candidate_pairs: List[Tuple[float, int, int]] = []
        for i, r_seg in enumerate(ref_segments):
            r_cx, r_cy = r_seg.bbox.center
            r_is_vert = r_seg.bbox.height >= r_seg.bbox.width
            for j, c_seg in enumerate(cur_segments):
                c_cx, c_cy = c_seg.bbox.center
                c_is_vert = c_seg.bbox.height >= c_seg.bbox.width

                # 1) 앵커 기준 상대 거리와 2) 정규화 이미지 좌표 거리 중 더 잘 정렬된 거리 사용
                dist_rel = float(np.hypot(c_seg.rel_cx - r_seg.rel_cx, c_seg.rel_cy - r_seg.rel_cy))
                dist_abs = float(np.hypot(c_cx - r_cx, c_cy - r_cy)) / max(1.0, max(ref_site.anchor_w, ref_site.anchor_h))
                pos_dist = min(dist_rel, dist_abs)

                # 수직 선분과 수평 선분이 서로 잘못 매칭되는 현상 방지 (단, 점 형태로 축소된 세그먼트는 예외 허용)
                orient_penalty = 0.0
                if (r_is_vert != c_is_vert) and min(r_seg.aspect_ratio, c_seg.aspect_ratio) > 2.2:
                    orient_penalty = 0.50

                cost = pos_dist + orient_penalty
                if pos_dist < 0.40:
                    candidate_pairs.append((cost, i, j))

        candidate_pairs.sort(key=lambda x: x[0])
        matched_r: Dict[int, SegmentFeature] = {}
        used_c = set()

        for _, i, j in candidate_pairs:
            if i not in matched_r and j not in used_c:
                matched_r[i] = cur_segments[j]
                used_c.add(j)

        matched_pairs: List[Tuple[SegmentFeature, Optional[SegmentFeature]]] = [
            (r_seg, matched_r.get(i, None))
            for i, r_seg in enumerate(ref_segments)
        ]
        unmatched_cur = [c_seg for j, c_seg in enumerate(cur_segments) if j not in used_c]
        return matched_pairs, unmatched_cur

    @classmethod
    def evaluate_site_defects(
        cls,
        cur_site: SiteFeature,
        ref_site: SiteFeature,
        cluster_sites_at_k: List[SiteFeature],
    ) -> List[DefectFinding]:
        """단일 계측 사이트(Site k)에 대해 기준 사이트 및 군집 분포와 대조하여 7대 결함을 판별합니다."""
        findings: List[DefectFinding] = []
        site_id = cur_site.site_id

        # 0. 초점 상실 / 패턴 완전 소실로 인해 영역이 강제 구제된 경우 (요구사항 2.4)
        if cur_site.is_rescued or cur_site.total_red_area < max(10.0, 0.08 * ref_site.total_red_area):
            findings.append(
                DefectFinding(
                    defect_type=DefectType.UNMEASURABLE_RESCUED,
                    severity_score=10.0,
                    site_id=site_id,
                    region_bbox=cur_site.normalized_bbox,
                    description=(
                        f"Site #{site_id + 1}: 계측 패턴/마커 소실로 자체 영역 추출 불가 — "
                        f"군집 기준 영역으로 강제 구제됨 (검출 면적 {cur_site.total_red_area:.0f}px² vs 기준 {ref_site.total_red_area:.0f}px²)"
                    ),
                    metrics={
                        "detected_area": cur_site.total_red_area,
                        "reference_area": ref_site.total_red_area,
                    },
                )
            )
            findings.append(
                DefectFinding(
                    defect_type=DefectType.LINE_TRUNCATION,
                    severity_score=9.5,
                    site_id=site_id,
                    region_bbox=cur_site.normalized_bbox,
                    description=f"Site #{site_id + 1}: 계측 선분/패턴 전체 결손",
                    metrics={"area_ratio": cur_site.total_red_area / max(1.0, ref_site.total_red_area)},
                )
            )
            return findings

        # 1. 방위각/사분면 구간 결손 검사 (특히 원형/타원형 Hole 및 다각형 변 통째 결손에 강력 동작)
        quad_names = ["우측(Right)", "하단(Bottom)", "좌측(Left)", "상단(Top)"]
        missing_quads = []
        missing_quad_indices = []
        for q_idx in range(4):
            ref_q = ref_site.quadrant_areas[q_idx]
            cur_q = cur_site.quadrant_areas[q_idx]
            # 기준 사이트에서 해당 방위에 충분한 계측 요소(> 12% 비중 및 > 150px)가 존재할 때
            if ref_q > max(140.0, 0.12 * ref_site.total_red_area):
                q_ratio = cur_q / ref_q
                if q_ratio < 0.38:
                    missing_quads.append(f"{quad_names[q_idx]} 구간({q_ratio * 100:.0f}%만 검출)")
                    missing_quad_indices.append(q_idx)

        if missing_quads:
            # 결손된 방위의 바운딩 영역 추정 (시각화 핀포인트용)
            nb = cur_site.normalized_bbox
            cx, cy = int(cur_site.anchor_cx), int(cur_site.anchor_cy)
            cx = min(max(cx, nb.x1 + 10), nb.x2 - 10)
            cy = min(max(cy, nb.y1 + 10), nb.y2 - 10)
            if 0 in missing_quad_indices:  # Right
                miss_box = BoundingBox(cx, nb.y1 + 10, nb.x2 - 10, nb.y2 - 10)
            elif 2 in missing_quad_indices:  # Left
                miss_box = BoundingBox(nb.x1 + 10, nb.y1 + 10, cx, nb.y2 - 10)
            elif 1 in missing_quad_indices:  # Bottom
                miss_box = BoundingBox(nb.x1 + 10, cy, nb.x2 - 10, nb.y2 - 10)
            else:  # Top
                miss_box = BoundingBox(nb.x1 + 10, nb.y1 + 10, nb.x2 - 10, cy)

            len_ratio = cur_site.total_skeleton_length / max(1.0, ref_site.total_skeleton_length)
            findings.append(
                DefectFinding(
                    defect_type=DefectType.LINE_TRUNCATION,
                    severity_score=round(8.5 + (1.0 - len_ratio) * 3.0, 2),
                    site_id=site_id,
                    region_bbox=miss_box,
                    description=(
                        f"Site #{site_id + 1} 선분/호 단절 및 결손: {', '.join(missing_quads)} 결손 "
                        f"(전체 계측 길이 정상 대비 {len_ratio * 100:.1f}%)"
                    ),
                    metrics={
                        "length_ratio": round(len_ratio, 3),
                        "missing_quadrant_count": float(len(missing_quads)),
                    },
                )
            )

        # 2. 세그먼트 단위 1:1 매칭 및 정밀 기하·위상·광학 결함 판별
        matched_pairs, _ = cls._match_segments(
            ref_site.segments, cur_site.segments, ref_site, cur_site
        )

        for r_seg, c_seg in matched_pairs:
            if c_seg is None:
                # 해당 위치의 세그먼트가 통째로 결손된 경우 (위 사분면 결손과 중복되지 않을 때 추가)
                if not missing_quads:
                    findings.append(
                        DefectFinding(
                            defect_type=DefectType.LINE_TRUNCATION,
                            severity_score=8.0,
                            site_id=site_id,
                            region_bbox=r_seg.bbox,
                            description=(
                                f"Site #{site_id + 1} 선분 결손: 기준 세그먼트(길이 {r_seg.span_length:.0f}px) 미검출"
                            ),
                            metrics={"reference_span": r_seg.span_length, "detected_span": 0.0},
                        )
                    )
                continue

            span_ratio = c_seg.span_length / max(1.0, r_seg.span_length)
            area_ratio = c_seg.area / max(1.0, r_seg.area)
            thk_ratio = c_seg.mean_thickness / max(1.0, r_seg.mean_thickness)

            # 앵커 기준 변위와 직접 픽셀 변위 중 더 잘 정렬된 기준의 오프셋 산출
            px_offset_rel = float(np.hypot(
                (c_seg.rel_cx - r_seg.rel_cx) * cur_site.anchor_w,
                (c_seg.rel_cy - r_seg.rel_cy) * cur_site.anchor_h,
            ))
            r_cx, r_cy = r_seg.bbox.center
            c_cx, c_cy = c_seg.bbox.center
            px_offset_abs = float(np.hypot(c_cx - r_cx, c_cy - r_cy))
            px_offset = min(px_offset_rel, px_offset_abs)
            rel_offset = px_offset / max(1.0, max(cur_site.anchor_w, cur_site.anchor_h))

            # (A) 패턴 축소 및 점 찍힘 (PATTERN_SHRINKAGE_DOT):
            # 정상 선분(span >= 28px)이 길이 45% 미만, 면적 45% 미만으로 급격히 오그라들고 점/단편 형태(span <= 24px 또는 aspect_ratio 급감)가 된 경우
            if (
                r_seg.span_length >= 25.0
                and span_ratio < 0.45
                and area_ratio < 0.45
                and (c_seg.span_length <= 24.0 or c_seg.aspect_ratio < r_seg.aspect_ratio * 0.5)
            ):
                findings.append(
                    DefectFinding(
                        defect_type=DefectType.PATTERN_SHRINKAGE_DOT,
                        severity_score=round(8.8 + (1.0 - span_ratio) * 2.5, 2),
                        site_id=site_id,
                        region_bbox=c_seg.bbox.pad(8, 8, 10000, 10000),
                        description=(
                            f"Site #{site_id + 1} 패턴 축소 및 점 찍힘: 세그먼트 길이가 정상 {r_seg.span_length:.0f}px에서 "
                            f"{c_seg.span_length:.0f}px({span_ratio * 100:.1f}%) 점 형태로 극단 축소됨"
                        ),
                        metrics={
                            "span_ratio": round(span_ratio, 3),
                            "area_ratio": round(area_ratio, 3),
                            "detected_span": c_seg.span_length,
                            "reference_span": r_seg.span_length,
                        },
                    )
                )

            # (B) 선분 단절 및 결손 (LINE_TRUNCATION - 개별 세그먼트 부분 단절):
            elif r_seg.span_length >= 25.0 and span_ratio < 0.72 and not missing_quads:
                findings.append(
                    DefectFinding(
                        defect_type=DefectType.LINE_TRUNCATION,
                        severity_score=round(7.0 + (1.0 - span_ratio) * 3.0, 2),
                        site_id=site_id,
                        region_bbox=c_seg.bbox.union(r_seg.bbox),
                        description=(
                            f"Site #{site_id + 1} 선분 단절 및 결손: 세그먼트 길이가 정상 {r_seg.span_length:.0f}px 대비 "
                            f"{c_seg.span_length:.0f}px({span_ratio * 100:.1f}%)로 짧게 끊김"
                        ),
                        metrics={"span_ratio": round(span_ratio, 3)},
                    )
                )

            # (C) 내부 파편화 (INTERNAL_FRAGMENTATION):
            # 단일 연속 선분이어야 할 코리도어 내부가 2개 이상의 조각으로 끊어지고 틈(Gap)이 발생한 경우
            if c_seg.fragment_count_in_corridor > max(1, r_seg.fragment_count_in_corridor):
                findings.append(
                    DefectFinding(
                        defect_type=DefectType.INTERNAL_FRAGMENTATION,
                        severity_score=round(8.2 + 0.8 * (c_seg.fragment_count_in_corridor - 1), 2),
                        site_id=site_id,
                        region_bbox=c_seg.bbox.pad(6, 6, 10000, 10000),
                        description=(
                            f"Site #{site_id + 1} 내부 파편화: 단일 계측 선분 내부에 끊김/틈이 발생하여 "
                            f"{c_seg.fragment_count_in_corridor}개 조각으로 분리됨 (정상 {r_seg.fragment_count_in_corridor}개)"
                        ),
                        metrics={
                            "fragment_count": float(c_seg.fragment_count_in_corridor),
                            "reference_fragment_count": float(r_seg.fragment_count_in_corridor),
                        },
                    )
                )

            # (D) 패턴 왜곡 (PATTERN_DISTORTION - 선분 비틀림/구불거림):
            # 정상 직선 세그먼트(r_seg.sinuosity_std < 1.0)가 구불구불하게 휘어진 경우 (예: G1_anom 내측 좌측 변)
            c_minor_span = float(min(c_seg.bbox.width, c_seg.bbox.height))
            r_minor_span = float(max(1, min(r_seg.bbox.width, r_seg.bbox.height)))
            if (
                not r_seg.is_curved_arc
                and c_seg.sinuosity_std > max(1.15, r_seg.sinuosity_std * 2.5)
                and (
                    c_seg.max_axis_deviation > max(2.5, r_seg.max_axis_deviation * 1.8)
                    or c_minor_span > r_minor_span * 1.8
                )
            ):
                findings.append(
                    DefectFinding(
                        defect_type=DefectType.PATTERN_DISTORTION,
                        severity_score=round(8.9 + 0.5 * c_seg.sinuosity_std, 2),
                        site_id=site_id,
                        region_bbox=c_seg.bbox.pad(8, 8, 10000, 10000),
                        description=(
                            f"Site #{site_id + 1} 패턴 왜곡: 계측 선분이 직선 형태를 유지하지 못하고 비정상적으로 뒤틀림 "
                            f"(축 휨 편차 {c_seg.sinuosity_std:.2f}px, 최대 이탈 {c_seg.max_axis_deviation:.1f}px vs 정상 {r_seg.sinuosity_std:.2f}px)"
                        ),
                        metrics={
                            "sinuosity_std": round(c_seg.sinuosity_std, 3),
                            "max_axis_deviation": round(c_seg.max_axis_deviation, 2),
                            "reference_sinuosity_std": round(r_seg.sinuosity_std, 3),
                        },
                    )
                )

            # (E) 계측 위치 불일치 (POSITION_MISMATCH):
            # 케이스 E-1: 기저 광학 패턴(밝은 라인/엣지) 위가 아닌 엉뚱한 배경/허공에 검출되고 위치가 이동한 경우 (예: G3_anom)
            wafer_intensity_drop = (
                r_seg.underlying_wafer_intensity > 85.0
                and c_seg.underlying_wafer_intensity < 0.45 * r_seg.underlying_wafer_intensity
            )
            # 케이스 E-2: 마커/앵커 기준 상대 좌표가 정상 위치에서 유의미하게 이탈한 경우 (단, 홀 반원 결손으로 인한 무게중심 편향은 제외)
            significant_pos_shift = (
                rel_offset > 0.038
                and px_offset > 12.0
                and not (r_seg.is_curved_arc and len(missing_quads) > 0)
            )

            if wafer_intensity_drop or significant_pos_shift:
                reason_parts = []
                if significant_pos_shift:
                    reason_parts.append(f"정상 계측 위치 대비 {px_offset:.1f}px(상대편차 {rel_offset * 100:.1f}%) 이탈")
                if wafer_intensity_drop:
                    reason_parts.append(
                        f"실제 패턴 라인(밝기 {r_seg.underlying_wafer_intensity:.0f})을 벗어나 배경 허공(밝기 {c_seg.underlying_wafer_intensity:.0f})에 오검출"
                    )
                sev = 9.0 if (wafer_intensity_drop and significant_pos_shift) else 8.1
                findings.append(
                    DefectFinding(
                        defect_type=DefectType.POSITION_MISMATCH,
                        severity_score=round(sev + 0.05 * px_offset, 2),
                        site_id=site_id,
                        region_bbox=c_seg.bbox.union(r_seg.bbox).pad(8, 8, 10000, 10000),
                        description=f"Site #{site_id + 1} 계측 위치 불일치: {' 및 '.join(reason_parts)}",
                        metrics={
                            "pixel_offset": round(px_offset, 2),
                            "relative_offset": round(rel_offset, 4),
                            "underlying_intensity": round(c_seg.underlying_wafer_intensity, 1),
                            "reference_intensity": round(r_seg.underlying_wafer_intensity, 1),
                        },
                    )
                )

            # (F) 선폭 팽창 및 뭉침 (LINE_BLOATING):
            if (
                thk_ratio > 1.45
                and (c_seg.mean_thickness - r_seg.mean_thickness) > 3.2
            ) or (
                c_seg.max_thickness > max(r_seg.max_thickness * 1.6, r_seg.max_thickness + 5.0)
                and area_ratio > 1.35
            ):
                findings.append(
                    DefectFinding(
                        defect_type=DefectType.LINE_BLOATING,
                        severity_score=round(7.8 + (thk_ratio - 1.0) * 2.5, 2),
                        site_id=site_id,
                        region_bbox=c_seg.bbox.pad(6, 6, 10000, 10000),
                        description=(
                            f"Site #{site_id + 1} 선폭 팽창 및 뭉침: 계측 선폭이 정상 {r_seg.mean_thickness:.1f}px에서 "
                            f"{c_seg.mean_thickness:.1f}px({thk_ratio * 100:.0f}%)로 비정상 팽창/뭉침 (최대 두께 {c_seg.max_thickness:.1f}px)"
                        ),
                        metrics={
                            "mean_thickness": round(c_seg.mean_thickness, 2),
                            "reference_thickness": round(r_seg.mean_thickness, 2),
                            "thickness_ratio": round(thk_ratio, 3),
                        },
                    )
                )

            # (G) 인접 영역 침범 (ADJACENT_INTRUSION):
            # 모서리나 끝단에서 직교 방향으로 꺾여 들어가 직교 방향 폭(min(w, h))이 정상 두께의 2.2배 이상 커지고 침범 비율이 높은 경우
            c_minor_span = float(min(c_seg.bbox.width, c_seg.bbox.height))
            r_minor_span = float(max(1, min(r_seg.bbox.width, r_seg.bbox.height)))
            if (
                not r_seg.is_curved_arc
                and c_minor_span > max(r_minor_span * 2.1, r_minor_span + 10.0)
                and c_seg.orthogonal_intrusion_ratio > max(0.12, r_seg.orthogonal_intrusion_ratio + 0.10)
                and c_seg.span_length >= r_seg.span_length * 0.75
            ):
                findings.append(
                    DefectFinding(
                        defect_type=DefectType.ADJACENT_INTRUSION,
                        severity_score=round(8.3 + c_seg.orthogonal_intrusion_ratio * 3.0, 2),
                        site_id=site_id,
                        region_bbox=c_seg.bbox.pad(8, 8, 10000, 10000),
                        description=(
                            f"Site #{site_id + 1} 인접 영역 침범: 계측 선분이 인접한 직교 변/영역으로 잘못 꺾여 침범함 "
                            f"(직교 방향 확장폭 {c_minor_span:.0f}px vs 정상 {r_minor_span:.0f}px)"
                        ),
                        metrics={
                            "orthogonal_span": c_minor_span,
                            "reference_orthogonal_span": r_minor_span,
                            "intrusion_ratio": round(c_seg.orthogonal_intrusion_ratio, 3),
                        },
                    )
                )

        # 3. 사이트 전역 간격/대칭 왜곡 보완 검사 (개별 세그먼트 결함으로 설명되지 않는 전역 대칭/간격 붕괴 검출)
        span_y_ratio = cur_site.span_y / max(1.0, ref_site.span_y)
        span_x_ratio = cur_site.span_x / max(1.0, ref_site.span_x)

        if not findings:
            if abs(1.0 - span_y_ratio) > 0.25 or abs(1.0 - span_x_ratio) > 0.25:
                findings.append(
                    DefectFinding(
                        defect_type=DefectType.PATTERN_DISTORTION,
                        severity_score=7.5,
                        site_id=site_id,
                        region_bbox=cur_site.normalized_bbox,
                        description=(
                            f"Site #{site_id + 1} 패턴 간격 왜곡: 전체 계측 스팬({cur_site.span_x:.0f}×{cur_site.span_y:.0f}px)이 "
                            f"정상 기준({ref_site.span_x:.0f}×{ref_site.span_y:.0f}px) 대비 비정상 왜곡됨"
                        ),
                        metrics={"span_x_ratio": round(span_x_ratio, 3), "span_y_ratio": round(span_y_ratio, 3)},
                    )
                )

        # 심각도(severity_score) 내림차순 정렬
        findings.sort(key=lambda f: f.severity_score, reverse=True)
        return findings

    @classmethod
    def classify_cluster(
        cls,
        image_paths: List[str],
        image_names: List[str],
        cluster_id: str,
        image_shapes: List[Tuple[int, int]],
        focus_results: List[FocusResult],
        sites_per_image: List[List[SiteFeature]],
    ) -> List[ImageInspectionResult]:
        """동일 군집 내 모든 이미지에 대해 정상 기준을 수립하고 최종 오계측 판정 및 원인 분류를 수행합니다."""
        num_images = len(image_paths)
        if num_images == 0:
            return []

        golden_idx = cls.select_golden_reference_index(sites_per_image, focus_results)
        golden_sites = sites_per_image[golden_idx]
        golden_name = image_names[golden_idx]
        golden_path = image_paths[golden_idx]

        results: List[ImageInspectionResult] = []

        for idx in range(num_images):
            H, W = image_shapes[idx]
            focus_res = focus_results[idx]
            cur_sites = sites_per_image[idx]
            all_findings: List[DefectFinding] = []

            # 1. 독립 병렬 검정된 광학 초점 불량(Defocus) 반영 (요구사항 2.5)
            if focus_res.is_defocused:
                all_findings.append(
                    DefectFinding(
                        defect_type=DefectType.OPTICAL_DEFOCUS,
                        severity_score=9.2,
                        site_id=0,
                        region_bbox=BoundingBox(0, 0, W, H),
                        description=focus_res.explanation,
                        metrics={
                            "sharpness_score": focus_res.sharpness_score,
                            "relative_to_cluster": focus_res.relative_to_cluster,
                        },
                    )
                )

            # 2. 각 사이트별 7대 오계측 결함 판별 (Golden Reference 자신이 아닌 경우 또는 자체 결함 검사)
            if idx != golden_idx or num_images == 1:
                for k, cur_site in enumerate(cur_sites):
                    ref_site = golden_sites[k] if k < len(golden_sites) else cur_site
                    cluster_sites_k = [
                        s_list[k] for s_list in sites_per_image if k < len(s_list)
                    ]
                    site_findings = cls.evaluate_site_defects(cur_site, ref_site, cluster_sites_k)
                    all_findings.extend(site_findings)

            all_findings.sort(key=lambda f: f.severity_score, reverse=True)

            is_outlier = len(all_findings) > 0
            status_label = "OUTLIER" if is_outlier else "NORMAL"

            if is_outlier:
                primary_defect = all_findings[0].defect_type.value
                unique_types = []
                for f in all_findings:
                    if f.defect_type.value not in unique_types:
                        unique_types.append(f.defect_type.value)

                # 주원인 및 상세 설명 요약 생성
                top_descriptions = [f.description for f in all_findings[:3]]
                summary_explanation = (
                    f"[오계측 판정] 주원인: {primary_defect}"
                    + (f" (복합 결함: {', '.join(unique_types)})" if len(unique_types) > 1 else "")
                    + " | 상세: " + " / ".join(top_descriptions)
                )
            else:
                primary_defect = None
                unique_types = []
                summary_explanation = (
                    f"[정상 계측] 군집 기준({golden_name}) 대비 모든 계측 위치·선분·형태 정합 완료 "
                    f"({focus_res.explanation})"
                )

            results.append(
                ImageInspectionResult(
                    image_path=image_paths[idx],
                    image_name=image_names[idx],
                    cluster_id=cluster_id,
                    width=W,
                    height=H,
                    is_outlier=is_outlier,
                    status_label=status_label,
                    focus_result=focus_res,
                    sites=cur_sites,
                    defects=all_findings,
                    primary_defect=primary_defect,
                    defect_types=unique_types,
                    summary_explanation=summary_explanation,
                    reference_image_name=golden_name,
                    reference_image_path=golden_path,
                )
            )

        return results
