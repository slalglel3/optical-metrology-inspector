"""ROI Localizer and Cluster Normalizer / Rescuer (요구사항 2.3 & 2.4).

핵심 기능:
1. 마커 성분(하늘색 등)과 검출 계측 요소(빨간색 등)를 모두 포함하는 최소 영역 추출 (잘림 방지)
2. 원거리 복수 계측 위치 자동 분리 (밀집된 단일 대형 패턴은 하나로 유지)
3. 군집 기준 영역 정규화 및 개별 검출 영역 보존 (Union 보장)
4. 초점 상실 / 패턴 소실로 영역 특정이 불가능한 이미지의 군집 기준 강제 구제 (Fallback Rescue)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from .models import BoundingBox
from .sanitizer import SanitizedLayers


@dataclass
class LocalizedSite:
    """단일 계측 사이트의 국소화 결과."""
    site_id: int
    raw_bbox: Optional[BoundingBox]      # 패딩 전 순수 (마커 U 계측요소) 최소 외접 박스 (결측 시 None)
    padded_bbox: BoundingBox             # 안전 마진 및 군집 정규화(Union)가 적용된 최종 ROI 박스
    is_rescued: bool                     # 초점상실/패턴소실로 군집 기준에 의해 구제되었는지 여부
    anchor_cx: float                     # 해당 사이트의 정밀 정렬 기준 중심 X
    anchor_cy: float                     # 해당 사이트의 정밀 정렬 기준 중심 Y
    anchor_w: float                      # 기준 폭
    anchor_h: float                      # 기준 높이


class ROILocalizer:
    """계측 영역 국소화, 원거리 다중 타깃 분리 및 군집 기반 정규화/구제를 수행합니다."""

    # 밀집된 단일 대형 패턴은 하나로 묶고 원거리 타깃은 분리하는 상대 반경 비율
    CLUSTER_RADIUS_RATIO = 0.075
    MIN_SITE_PIXELS = 20

    @classmethod
    def extract_raw_sites(
        cls,
        layers: SanitizedLayers,
    ) -> List[BoundingBox]:
        """단일 이미지에서 (마커 U 계측요소) 합집합을 클러스터링하여 독립된 최소 영역들을 추출합니다."""
        H, W = layers.gray_wafer.shape
        union_mask = (layers.cyan_marker_mask | layers.clean_red_mask).astype(np.uint8)

        if int(union_mask.sum()) < cls.MIN_SITE_PIXELS:
            return []

        # 적응형 공간 클러스터링: 단변의 7.5% 반경 내 요소들은 단일 패턴으로 결합, 원거리는 분리
        rad = max(12, int(cls.CLUSTER_RADIUS_RATIO * min(H, W)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        dilated = cv2.dilate(union_mask, kernel)

        num, labels, _, _ = cv2.connectedComponentsWithStats(dilated)
        raw_boxes: List[BoundingBox] = []

        for i in range(1, num):
            cluster_pts = (union_mask > 0) & (labels == i)
            if int(cluster_pts.sum()) < cls.MIN_SITE_PIXELS:
                continue
            ys, xs = np.where(cluster_pts)
            bbox = BoundingBox(
                x1=int(xs.min()),
                y1=int(ys.min()),
                x2=int(xs.max()) + 1,
                y2=int(ys.max()) + 1,
            )
            raw_boxes.append(bbox)

        # 공간 순서 일관성을 위해 (y 중심 상위 -> x 중심 좌측) 순으로 정렬
        raw_boxes.sort(key=lambda b: (round(b.center[1] / max(1.0, H * 0.15)), b.center[0]))
        return raw_boxes

    @classmethod
    def _compute_site_anchor(
        cls,
        site_bbox: BoundingBox,
        layers: SanitizedLayers,
        consensus_bbox: BoundingBox,
    ) -> Tuple[float, float, float, float]:
        """사이트 내부의 정밀 상대 좌표계 원점(Anchor Center)과 기준 스케일을 산출합니다.

        우선순위:
        1. 사이트 내부에 검출된 타깃 중심 십자가(+)가 존재하고 단일 홀/타깃 중심이면 해당 중심 활용
        2. 하늘색 마커(cyan_marker_mask)가 대칭 프레임/박스를 형성하면 마커 외접 중심 활용
        3. 마커가 반지름 직선(예: G4 우하단 원)이거나 결측된 경우 정규화 박스 중심 활용
        """
        H, W = layers.gray_wafer.shape
        x1, y1, x2, y2 = site_bbox.to_tuple()
        sub_cyan = layers.cyan_marker_mask[y1:y2, x1:x2]
        sub_red = layers.clean_red_mask[y1:y2, x1:x2]

        # 1. 사이트 내부에 십자가(+) 중심이 있는지 확인
        inside_crosses = [
            (cx, cy) for (cx, cy) in layers.crosshair_centers
            if x1 <= cx <= x2 and y1 <= cy <= y2
        ]

        # 하늘색 마커 외접 영역 계산
        if int(sub_cyan.sum()) >= 20:
            c_ys, c_xs = np.where(sub_cyan)
            cyan_w = float(c_xs.max() - c_xs.min() + 1)
            cyan_h = float(c_ys.max() - c_ys.min() + 1)
            cyan_cx = x1 + (c_xs.min() + c_xs.max()) / 2.0
            cyan_cy = y1 + (c_ys.min() + c_ys.max()) / 2.0

            # 만약 하늘색 마커가 빨간색 계측 요소의 절반도 감싸지 못하는 지시선 형태(예: G4 Site 2)라면
            if int(sub_red.sum()) >= 20:
                r_ys, r_xs = np.where(sub_red)
                red_x1, red_y1 = x1 + int(r_xs.min()), y1 + int(r_ys.min())
                red_x2, red_y2 = x1 + int(r_xs.max()), y1 + int(r_ys.max())
                cyan_x1, cyan_y1 = x1 + int(c_xs.min()), y1 + int(c_ys.min())
                cyan_x2, cyan_y2 = x1 + int(c_xs.max()), y1 + int(c_ys.max())

                # 지시선 마커 판별: 빨간색 영역이 하늘색 마커 영역 바깥(좌/하단 등)으로 크게 뻗어 있고 내부 십자가가 존재할 때
                if inside_crosses and (cyan_x1 > red_x1 + 15 or cyan_y2 < red_y2 - 15):
                    ccx, ccy = inside_crosses[0]
                    return (
                        ccx,
                        ccy,
                        max(float(consensus_bbox.width), 1.0),
                        max(float(consensus_bbox.height), 1.0),
                    )

            return (
                cyan_cx,
                cyan_cy,
                max(cyan_w, 1.0),
                max(cyan_h, 1.0),
            )

        if inside_crosses:
            ccx, ccy = inside_crosses[0]
            return (
                ccx,
                ccy,
                max(float(consensus_bbox.width), 1.0),
                max(float(consensus_bbox.height), 1.0),
            )

        cx, cy = consensus_bbox.center
        return (
            cx,
            cy,
            max(float(consensus_bbox.width), 1.0),
            max(float(consensus_bbox.height), 1.0),
        )

    @classmethod
    def normalize_and_rescue_batch(
        cls,
        raw_sites_per_image: List[List[BoundingBox]],
        image_shapes: List[Tuple[int, int]],
        sanitized_layers_list: List[SanitizedLayers],
    ) -> List[List[LocalizedSite]]:
        """동일 군집 내 모든 이미지의 계측 영역을 정규화하고, 초점상실/패턴소실 이미지를 구제합니다."""
        num_images = len(raw_sites_per_image)
        if num_images == 0:
            return []

        # 1. 군집 내 정상 사이트 개수(K_consensus) 결정 (결측이 아닌 이미지들의 최빈값/중앙값)
        valid_counts = [len(boxes) for boxes in raw_sites_per_image if len(boxes) > 0]
        if not valid_counts:
            # 모든 이미지에서 전혀 검출되지 않은 극단적 상황: 화면 중앙 기본 ROI 부여
            results: List[List[LocalizedSite]] = []
            for idx, (H, W) in enumerate(image_shapes):
                fallback = BoundingBox(int(W * 0.2), int(H * 0.2), int(W * 0.8), int(H * 0.8))
                cx, cy = fallback.center
                results.append([
                    LocalizedSite(
                        site_id=0,
                        raw_bbox=None,
                        padded_bbox=fallback,
                        is_rescued=True,
                        anchor_cx=cx,
                        anchor_cy=cy,
                        anchor_w=float(fallback.width),
                        anchor_h=float(fallback.height),
                    )
                ])
            return results

        # 최빈 사이트 수 산출 (동률일 경우 더 완전하게 검출된 쪽 선택)
        counts_vals, counts_freq = np.unique(valid_counts, return_counts=True)
        max_freq = counts_freq.max()
        target_k = int(counts_vals[counts_freq == max_freq].max())

        # 2. 각 사이트 k (0 .. target_k - 1)에 대한 군집 기준 상대 영역 (Normalized [0, 1] 좌표) 산출
        # 결손된 이미지에 의해 영역이 좁아지지 않도록 각 사이트의 (x1, y1)은 하위 백분위/최소 포괄, (x2, y2)는 상위 포괄 결합
        consensus_rel_boxes: List[Tuple[float, float, float, float]] = []
        for k in range(target_k):
            rel_x1s, rel_y1s, rel_x2s, rel_y2s = [], [], [], []
            for img_idx, boxes in enumerate(raw_sites_per_image):
                if len(boxes) == target_k:
                    H, W = image_shapes[img_idx]
                    b = boxes[k]
                    rel_x1s.append(b.x1 / float(W))
                    rel_y1s.append(b.y1 / float(H))
                    rel_x2s.append(b.x2 / float(W))
                    rel_y2s.append(b.y2 / float(H))

            if rel_x1s:
                # 이미지 수가 적을 때(예: 2장 중 1장이 반쪽 결손) 잘림을 방지하기 위해
                # 중앙값과 포괄 범위(25th / 75th percentile)를 결합
                rx1 = float(np.percentile(rel_x1s, 25.0))
                ry1 = float(np.percentile(rel_y1s, 25.0))
                rx2 = float(np.percentile(rel_x2s, 75.0))
                ry2 = float(np.percentile(rel_y2s, 75.0))
                consensus_rel_boxes.append((rx1, ry1, rx2, ry2))
            else:
                consensus_rel_boxes.append((0.25, 0.25, 0.75, 0.75))

        # 3. 개별 이미지별 정규화, Union 보존 및 결측 구제(Rescue) 수행
        batch_localized: List[List[LocalizedSite]] = []
        for img_idx in range(num_images):
            H, W = image_shapes[img_idx]
            boxes = raw_sites_per_image[img_idx]
            layers = sanitized_layers_list[img_idx]
            img_sites: List[LocalizedSite] = []

            # 현재 이미지 해상도로 투영한 군집 기준 박스들
            img_consensus_boxes = [
                BoundingBox(
                    x1=int(round(rx1 * W)),
                    y1=int(round(ry1 * H)),
                    x2=int(round(rx2 * W)),
                    y2=int(round(ry2 * H)),
                )
                for (rx1, ry1, rx2, ry2) in consensus_rel_boxes
            ]

            if len(boxes) == 0:
                # [구제 케이스 1] 초점 상실 / 패턴 완전 소실 이미지 -> 모든 군집 기준 영역을 강제 투영
                for k, c_box in enumerate(img_consensus_boxes):
                    pad_px = max(16, int(0.05 * max(c_box.width, c_box.height)))
                    padded = c_box.pad(pad_px, pad_px, W, H)
                    cx, cy = c_box.center
                    img_sites.append(
                        LocalizedSite(
                            site_id=k,
                            raw_bbox=None,
                            padded_bbox=padded,
                            is_rescued=True,
                            anchor_cx=cx,
                            anchor_cy=cy,
                            anchor_w=float(max(1, c_box.width)),
                            anchor_h=float(max(1, c_box.height)),
                        )
                    )
            elif len(boxes) == target_k:
                # [정상 개수 검출 케이스] 개별 영역이 절대 잘리지 않도록 Union(raw_bbox, consensus_bbox) 수행
                for k in range(target_k):
                    raw_b = boxes[k]
                    c_box = img_consensus_boxes[k]
                    # 만약 패턴 전체가 평행 이동한 경우(IoU > 0.2) 두 영역의 Union으로 결손 부위와 현재 검출 요소 모두 100% 보존
                    if raw_b.iou(c_box) > 0.15:
                        merged = raw_b.union(c_box)
                    else:
                        # 위치가 크게 이동한 경우에도 raw_b를 보존하되 크기가 비정상 축소되었으면 c_box 크기 이상 확보
                        merged = raw_b.union(c_box)

                    pad_px = max(16, int(0.05 * max(merged.width, merged.height)))
                    padded = merged.pad(pad_px, pad_px, W, H)
                    acx, acy, aw, ah = cls._compute_site_anchor(padded, layers, merged)
                    img_sites.append(
                        LocalizedSite(
                            site_id=k,
                            raw_bbox=raw_b,
                            padded_bbox=padded,
                            is_rescued=False,
                            anchor_cx=acx,
                            anchor_cy=acy,
                            anchor_w=aw,
                            anchor_h=ah,
                        )
                    )
            else:
                # [부분 결측 또는 과분할 케이스] 각 군집 기준 사이트에 가장 가까운 박스를 매칭하고 없는 사이트는 구제
                used_raw = set()
                for k, c_box in enumerate(img_consensus_boxes):
                    c_cx, c_cy = c_box.center
                    # c_box와 겹치거나 중심 거리가 가까운 raw_box 탐색
                    best_j: Optional[int] = None
                    best_dist = float("inf")
                    for j, rb in enumerate(boxes):
                        if j in used_raw:
                            continue
                        rcx, rcy = rb.center
                        dist = float(np.hypot(rcx - c_cx, rcy - c_cy))
                        if rb.iou(c_box) > 0.05 or dist < 0.25 * min(W, H):
                            if dist < best_dist:
                                best_dist = dist
                                best_j = j

                    if best_j is not None:
                        used_raw.add(best_j)
                        raw_b = boxes[best_j]
                        merged = raw_b.union(c_box)
                        pad_px = max(16, int(0.05 * max(merged.width, merged.height)))
                        padded = merged.pad(pad_px, pad_px, W, H)
                        acx, acy, aw, ah = cls._compute_site_anchor(padded, layers, merged)
                        img_sites.append(
                            LocalizedSite(
                                site_id=k,
                                raw_bbox=raw_b,
                                padded_bbox=padded,
                                is_rescued=False,
                                anchor_cx=acx,
                                anchor_cy=acy,
                                anchor_w=aw,
                                anchor_h=ah,
                            )
                        )
                    else:
                        # 해당 사이트가 소실된 경우 군집 기준으로 구제(Rescue)
                        pad_px = max(16, int(0.05 * max(c_box.width, c_box.height)))
                        padded = c_box.pad(pad_px, pad_px, W, H)
                        cx, cy = c_box.center
                        img_sites.append(
                            LocalizedSite(
                                site_id=k,
                                raw_bbox=None,
                                padded_bbox=padded,
                                is_rescued=True,
                                anchor_cx=cx,
                                anchor_cy=cy,
                                anchor_w=float(max(1, c_box.width)),
                                anchor_h=float(max(1, c_box.height)),
                            )
                        )

                # 만약 매칭되지 않고 남은 유효 raw_box가 있다면(원거리 추가 검출 요소) 잘림 방지를 위해 보존
                for j, rb in enumerate(boxes):
                    if j not in used_raw:
                        pad_px = max(16, int(0.05 * max(rb.width, rb.height)))
                        padded = rb.pad(pad_px, pad_px, W, H)
                        acx, acy, aw, ah = cls._compute_site_anchor(padded, layers, rb)
                        img_sites.append(
                            LocalizedSite(
                                site_id=len(img_sites),
                                raw_bbox=rb,
                                padded_bbox=padded,
                                is_rescued=False,
                                anchor_cx=acx,
                                anchor_cy=acy,
                                anchor_w=aw,
                                anchor_h=ah,
                            )
                        )

            batch_localized.append(img_sites)

        return batch_localized
