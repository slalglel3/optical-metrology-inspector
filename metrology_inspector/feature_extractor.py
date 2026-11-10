"""Metrology Feature Extractor (요구사항 2.1, 2.6, 2.7: 범용 기하·위상·광학 정합 피처 추출).

선, 원, 타원, 사각형, 다각형 등 모든 형태의 계측 패턴에 범용 대응하기 위해
사이트 전역 피처(12방위 각도 커버리지, 4방위 대칭성, 스팬, 총 골격 길이)와
개별 세그먼트/코리도어 피처(상대 위치, 길이, 선폭, 휨/곡률 잔차, 직교 침범률, 내부 파편화 수, 기저 광학 정합도)를 추출합니다.
"""

from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np
from .models import BoundingBox, SegmentFeature, SiteFeature
from .roi_localizer import LocalizedSite
from .sanitizer import SanitizedLayers


class MetrologyFeatureExtractor:
    """정제된 이미지 레이어와 정규화 ROI로부터 오계측 판별용 다차원 피처를 추출합니다."""

    @staticmethod
    def _compute_skeleton_and_thickness(comp_mask: np.ndarray) -> Tuple[float, float, float]:
        """단일 이진 마스크의 골격 길이(Skeleton Length), 평균 선폭(Mean Width), 최대 선폭(Max Width)을 계산합니다."""
        area = float(comp_mask.sum())
        if area <= 0:
            return 0.0, 0.0, 0.0

        # 패딩 후 거리 변환 수행 (경계 왜곡 방지)
        padded = np.pad(comp_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        max_rad = float(dist.max())
        max_thickness = max(1.0, 2.0 * max_rad - 0.5)

        # Ridge(국소 최대 거리 축) 픽셀 추출을 통한 골격 근사
        dil_dist = cv2.dilate(dist, np.ones((3, 3), np.uint8))
        ridge_mask = comp_mask & (dist >= (dil_dist - 1e-4)) & (dist > 0)
        ridge_vals = dist[ridge_mask]

        if ridge_vals.size > 0:
            mean_thickness = float(max(1.0, 2.0 * np.median(ridge_vals) - 0.5))
        else:
            mean_thickness = float(max(1.0, 2.0 * np.mean(dist[comp_mask])))

        skeleton_length = max(1.0, area / max(1.0, mean_thickness))
        return skeleton_length, mean_thickness, max_thickness

    @staticmethod
    def _analyze_axis_trajectory(
        comp_mask: np.ndarray,
        is_curved_site: bool = False,
    ) -> Tuple[float, float, float, float]:
        """세그먼트의 주축(수평/수직/대각)을 따라 궤적의 휨 표준편차(Sinuosity), 최대 이탈량, 직교 침범률을 산출합니다."""
        h, w = comp_mask.shape
        ys, xs = np.where(comp_mask)
        if ys.size < 6:
            return 0.0, 0.0, 0.0, 0.0

        # PCA 고유벡터로 주 방향 각도 계산
        pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
        mean_pt = pts.mean(axis=0)
        centered = pts - mean_pt
        cov = np.cov(centered, rowvar=False)
        if cov.ndim == 2:
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal_vec = eigvecs[:, np.argmax(eigvals)]
            angle_deg = float(np.degrees(np.arctan2(principal_vec[1], principal_vec[0]))) % 180.0
        else:
            angle_deg = 0.0

        # 수직 우세(h >= w) vs 수평 우세(w > h)에 따라 행/열별 중심 궤적 및 폭 분포 계산
        if h >= w:
            # 각 행(y)에서의 x 중심 좌표와 행 폭 w(y) 계산
            row_counts = comp_mask.sum(axis=1)
            valid_rows = np.where(row_counts > 0)[0]
            if valid_rows.size < 4:
                return angle_deg, 0.0, 0.0, 0.0
            x_centers = np.array([ np.where(comp_mask[r, :])[0].mean() for r in valid_rows ], dtype=np.float64)
            x_spans = np.array([
                np.where(comp_mask[r, :])[0].max() - np.where(comp_mask[r, :])[0].min() + 1
                for r in valid_rows
            ], dtype=np.float64)
            t = valid_rows.astype(np.float64)
        else:
            col_counts = comp_mask.sum(axis=0)
            valid_cols = np.where(col_counts > 0)[0]
            if valid_cols.size < 4:
                return angle_deg, 0.0, 0.0, 0.0
            x_centers = np.array([ np.where(comp_mask[:, c])[0].mean() for c in valid_cols ], dtype=np.float64)
            x_spans = np.array([
                np.where(comp_mask[:, c])[0].max() - np.where(comp_mask[:, c])[0].min() + 1
                for c in valid_cols
            ], dtype=np.float64)
            t = valid_cols.astype(np.float64)

        # 1차 선형 회귀(직선 기준) 잔차 계산
        if t.size >= 4 and (t.max() - t.min()) > 2:
            coeffs = np.polyfit(t, x_centers, deg=1)
            fitted = np.polyval(coeffs, t)
            residuals = x_centers - fitted
            sinuosity_std = float(np.std(residuals))
            max_dev = float(np.max(np.abs(residuals)))

            # 만약 원래 곡선(원/타원 호)인 경우에는 2차 곡선(포물선/원호 근사) 잔차로 매끄러움 평가
            if is_curved_site and t.size >= 6:
                coeffs2 = np.polyfit(t, x_centers, deg=2)
                res2 = x_centers - np.polyval(coeffs2, t)
                sinuosity_std = float(np.std(res2))
                max_dev = float(np.max(np.abs(res2)))
        else:
            sinuosity_std = 0.0
            max_dev = 0.0

        # 인접 영역 침범(Orthogonal Intrusion):
        # 선분의 정상 중앙 폭(median span) 대비 특정 구간(모서리/끝단)에서 직교 방향으로 꺾여 폭이 급팽창하거나 중심이 급격히 꺾인 비율
        med_span = float(np.median(x_spans))
        intrusion_rows = (x_spans > max(med_span * 2.1, med_span + 6.0))
        # 끝단 25% 구간에서 선분 중심이 중앙 추세선에서 크게 꺾여 나간 경우도 침범에 반영
        if t.size >= 8:
            mid_slice = slice(int(t.size * 0.2), int(t.size * 0.8))
            if (mid_slice.stop - mid_slice.start) >= 3:
                mid_coeffs = np.polyfit(t[mid_slice], x_centers[mid_slice], deg=1)
                end_res = np.abs(x_centers - np.polyval(mid_coeffs, t))
                bent_rows = end_res > max(5.0, med_span * 0.85)
                intrusion_ratio = float(np.mean(intrusion_rows | bent_rows))
            else:
                intrusion_ratio = float(np.mean(intrusion_rows))
        else:
            intrusion_ratio = float(np.mean(intrusion_rows))

        return angle_deg, sinuosity_std, max_dev, intrusion_ratio

    @staticmethod
    def _sample_underlying_wafer_signal(
        seg_bbox: BoundingBox,
        gray_wafer: np.ndarray,
        grad_mag: np.ndarray,
    ) -> Tuple[float, float]:
        """검출된 계측 세그먼트 위치 및 연장선 인접부의 기저 웨이퍼 밝기와 엣지 그래디언트를 측정합니다."""
        H, W = gray_wafer.shape
        x1, y1, x2, y2 = seg_bbox.to_tuple()
        w, h = seg_bbox.width, seg_bbox.height

        # 세그먼트가 수평 방향(w >= h)이면 좌/우 연장선상의 순수 웨이퍼 픽셀을 함께 샘플링하여
        # 빨간색 오버레이 색상에 가려지지 않은 실제 기저 패턴(밝은 라인/엣지) 신호를 정확히 획득
        samples_i = []
        samples_g = []

        if w >= h:
            cy = int((y1 + y2) / 2)
            ry1, ry2 = max(0, cy - max(2, h // 3)), min(H, cy + max(3, h // 3 + 1))
            # 좌측 연장 구간 및 우측 연장 구간 샘플링
            lx1, lx2 = max(0, x1 - 28), max(0, x1 - 5)
            rx1, rx2 = min(W, x2 + 5), min(W, x2 + 28)
            if lx2 > lx1 and ry2 > ry1:
                samples_i.append(gray_wafer[ry1:ry2, lx1:lx2].ravel())
                samples_g.append(grad_mag[ry1:ry2, lx1:lx2].ravel())
            if rx2 > rx1 and ry2 > ry1:
                samples_i.append(gray_wafer[ry1:ry2, rx1:rx2].ravel())
                samples_g.append(grad_mag[ry1:ry2, rx1:rx2].ravel())
        else:
            cx = int((x1 + x2) / 2)
            rx1, rx2 = max(0, cx - max(2, w // 3)), min(W, cx + max(3, w // 3 + 1))
            ty1, ty2 = max(0, y1 - 28), max(0, y1 - 5)
            by1, by2 = min(H, y2 + 5), min(H, y2 + 28)
            if ty2 > ty1 and rx2 > rx1:
                samples_i.append(gray_wafer[ty1:ty2, rx1:rx2].ravel())
                samples_g.append(grad_mag[ty1:ty2, rx1:rx2].ravel())
            if by2 > by1 and rx2 > rx1:
                samples_i.append(gray_wafer[by1:by2, rx1:rx2].ravel())
                samples_g.append(grad_mag[by1:by2, rx1:rx2].ravel())

        # 세그먼트 직하방 및 경계 패치도 보조로 포함
        px1, py1 = max(0, x1 - 4), max(0, y1 - 4)
        px2, py2 = min(W, x2 + 4), min(H, y2 + 4)
        patch_g = grad_mag[py1:py2, px1:px2].ravel()

        if samples_i:
            all_i = np.concatenate(samples_i)
            all_g = np.concatenate(samples_g + [patch_g])
            mean_intensity = float(np.mean(all_i))
            edge_grad = float(np.percentile(all_g, 85.0)) if all_g.size > 0 else 0.0
        else:
            patch_i = gray_wafer[py1:py2, px1:px2].ravel()
            mean_intensity = float(np.mean(patch_i)) if patch_i.size > 0 else 0.0
            edge_grad = float(np.percentile(patch_g, 85.0)) if patch_g.size > 0 else 0.0

        return mean_intensity, edge_grad

    @classmethod
    def _group_collinear_corridors(
        cls,
        comp_boxes: List[BoundingBox],
        comp_labels: np.ndarray,
    ) -> List[List[int]]:
        """원래 하나의 선분이어야 하나 내부 파편화/단절로 인해 쪼개진 동일 축(Corridor) 조각들을 그룹화합니다."""
        n = len(comp_boxes)
        if n <= 1:
            return [[i] for i in range(n)]

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            bi = comp_boxes[i]
            wi, hi = bi.width, bi.height
            for j in range(i + 1, n):
                bj = comp_boxes[j]
                wj, hj = bj.width, bj.height

                # 1. 수직 선분 축(Collinear Vertical Corridor) 검사:
                # 두 조각 모두 세로형(또는 작은 파편)이고 x 범위가 겹치거나 12px 이내로 정렬되어 있으며,
                # y 방향 끝단 간격(Gap)이 0 ~ 25px 이내인 경우 (단, 서로 다른 내/외곽 평행 선분은 x 거리가 멀어 분리됨)
                x_dist = max(0, max(bi.x1, bj.x1) - min(bi.x2, bj.x2))
                y_gap = max(0, max(bi.y1, bj.y1) - min(bi.y2, bj.y2))
                cxi, cxj = bi.center[0], bj.center[0]

                if abs(cxi - cxj) <= 14.0 and x_dist <= 8 and 0 <= y_gap <= 24:
                    # 두 조각 모두 수직 선분 성분일 때만 수직 코리도어로 병합
                    if hi >= wi and hj >= wj:
                        union(i, j)
                        continue

                # 2. 수평 선분 축(Collinear Horizontal Corridor) 검사:
                y_dist = max(0, max(bi.y1, bj.y1) - min(bi.y2, bj.y2))
                x_gap = max(0, max(bi.x1, bj.x1) - min(bi.x2, bj.x2))
                cyi, cyj = bi.center[1], bj.center[1]

                if abs(cyi - cyj) <= 14.0 and y_dist <= 8 and 0 <= x_gap <= 24:
                    if wi >= hi and wj >= hj:
                        union(i, j)

        groups_dict = {}
        for i in range(n):
            root = find(i)
            groups_dict.setdefault(root, []).append(i)

        return list(groups_dict.values())

    @classmethod
    def extract_site_features(
        cls,
        site: LocalizedSite,
        layers: SanitizedLayers,
    ) -> SiteFeature:
        """정규화된 단일 계측 사이트(Sub-ROI)에서 모든 정량적 피처를 추출합니다."""
        x1, y1, x2, y2 = site.padded_bbox.to_tuple()
        sub_gray = layers.gray_wafer[y1:y2, x1:x2]
        sub_cyan = layers.cyan_marker_mask[y1:y2, x1:x2]
        sub_red_clean = layers.clean_red_mask[y1:y2, x1:x2]
        sub_red_recon = layers.reconnected_red_mask[y1:y2, x1:x2]

        # 기저 웨이퍼 그래디언트 계산
        gx = cv2.Sobel(sub_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sub_gray, cv2.CV_32F, 0, 1, ksize=3)
        sub_grad = cv2.magnitude(gx, gy)

        total_red_area = float(sub_red_clean.sum())
        total_cyan_area = float(sub_cyan.sum())

        if total_red_area < 5:
            # 계측 요소가 거의 소실된 상태 (초점 상실 또는 완전 미검출)
            return SiteFeature(
                site_id=site.site_id,
                raw_bbox=site.raw_bbox,
                normalized_bbox=site.padded_bbox,
                is_rescued=site.is_rescued,
                anchor_cx=site.anchor_cx,
                anchor_cy=site.anchor_cy,
                anchor_w=site.anchor_w,
                anchor_h=site.anchor_h,
                total_red_area=total_red_area,
                total_cyan_area=total_cyan_area,
                total_skeleton_length=0.0,
                mean_thickness=0.0,
                max_thickness=0.0,
                span_x=0.0,
                span_y=0.0,
                rel_centroid_x=0.0,
                rel_centroid_y=0.0,
                angular_sector_areas=[0.0] * 12,
                quadrant_areas=[0.0] * 4,
                symmetry_score=0.0,
                intrinsic_integrity_score=0.0,
                segments=[],
            )

        # 사이트 내 전체 빨간색 픽셀 좌표 (원본 이미지 절대 좌표계로 변환)
        r_ys, r_xs = np.where(sub_red_clean)
        abs_xs = r_xs.astype(np.float64) + x1
        abs_ys = r_ys.astype(np.float64) + y1

        span_x = float(abs_xs.max() - abs_xs.min() + 1.0)
        span_y = float(abs_ys.max() - abs_ys.min() + 1.0)

        # Anchor 기준 상대 무게중심
        rel_cx = float((abs_xs.mean() - site.anchor_cx) / max(1.0, site.anchor_w))
        rel_cy = float((abs_ys.mean() - site.anchor_cy) / max(1.0, site.anchor_h))

        # 12방위 각도 섹터(30도 간격) 및 4방위(우, 하, 좌, 상) 픽셀 분포 계산
        dx = abs_xs - site.anchor_cx
        dy = abs_ys - site.anchor_cy
        angles_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

        sector_areas: List[float] = []
        for s_idx in range(12):
            ang_low = s_idx * 30.0
            ang_high = (s_idx + 1) * 30.0
            cnt = float(np.sum((angles_deg >= ang_low) & (angles_deg < ang_high)))
            sector_areas.append(cnt)

        # 4방위(Quadrant: Right [-45~45], Bottom [45~135], Left [135~225], Top [225~315])
        q_right = float(np.sum((angles_deg < 45.0) | (angles_deg >= 315.0)))
        q_bottom = float(np.sum((angles_deg >= 45.0) & (angles_deg < 135.0)))
        q_left = float(np.sum((angles_deg >= 135.0) & (angles_deg < 225.0)))
        q_top = float(np.sum((angles_deg >= 225.0) & (angles_deg < 315.0)))
        quadrant_areas = [q_right, q_bottom, q_left, q_top]

        # 좌-우 및 상-하 대칭성 점수 산출
        lr_sum = q_left + q_right
        tb_sum = q_top + q_bottom
        lr_sym = 1.0 - (abs(q_left - q_right) / lr_sum) if lr_sum > 25 else 1.0
        tb_sym = 1.0 - (abs(q_top - q_bottom) / tb_sum) if tb_sum > 25 else 1.0
        symmetry_score = 0.5 * lr_sym + 0.5 * tb_sym

        # 곡선/홀 사이트 여부 판별 (대각 섹터 비중 및 잔가시 형태학 확인)
        diag_mass = (
            sector_areas[1] + sector_areas[2] + sector_areas[4] + sector_areas[5]
            + sector_areas[7] + sector_areas[8] + sector_areas[10] + sector_areas[11]
        )
        # 사이트 내에 검출된 십자가 중심이 있고 대각 방위에도 호(Arc) 성분이 고르게 분포하면 곡선/홀 패턴
        is_curved_site = (diag_mass / max(1.0, total_red_area)) > 0.38 and len(layers.crosshair_centers) > 0

        # 곡선/홀 패턴의 경우 마젠타 텍스트(DA2, CD4 등)나 하늘색 지시선이 원호를 가로지르며 만든 갭(최대 15px)을 연결
        if is_curved_site:
            k_arc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            sub_red_recon = cv2.morphologyEx(sub_red_recon.astype(np.uint8), cv2.MORPH_CLOSE, k_arc) > 0

        # 개별 연결 컴포넌트 및 코리도어 그룹 추출 (마젠타 가림 복원 마스크 기준)
        num_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(sub_red_recon.astype(np.uint8))
        raw_comp_boxes: List[BoundingBox] = []
        raw_comp_ids: List[int] = []

        max_carea = max((stats_c[cid][4] for cid in range(1, num_c)), default=0)
        for cid in range(1, num_c):
            cx, cy, cw, ch, carea = stats_c[cid]
            pure_red_cnt = int(((labels_c == cid) & sub_red_clean).sum())
            # 미세 잔여 노이즈(최대 세그먼트의 3% 미만이고 35px 이하)는 배제
            if pure_red_cnt < 12 or carea < 12 or (max_carea > 300 and pure_red_cnt < 55 and min(cw, ch) <= 6):
                continue
            raw_comp_boxes.append(BoundingBox(cx, cy, cx + cw, cy + ch))
            raw_comp_ids.append(cid)

        # 동일 선분 축에서 끊어진 조각들을 코리도어로 그룹화 (내부 파편화 및 휨 분석용)
        corridor_groups = cls._group_collinear_corridors(raw_comp_boxes, labels_c)

        segments: List[SegmentFeature] = []
        total_skel_len = 0.0
        weighted_thickness_sum = 0.0
        global_max_thickness = 0.0

        for seg_idx, group in enumerate(corridor_groups):
            # 그룹 내 조각들의 통합 마스크 생성
            group_mask = np.zeros_like(sub_red_recon, dtype=bool)
            for idx_in_list in group:
                cid = raw_comp_ids[idx_in_list]
                group_mask |= (labels_c == cid)

            # 순수 빨간색 픽셀 기준 면적
            clean_group_mask = group_mask & sub_red_clean
            if int(clean_group_mask.sum()) < 6:
                clean_group_mask = group_mask

            g_ys, g_xs = np.where(clean_group_mask)
            lx1, ly1 = int(g_xs.min()), int(g_ys.min())
            lx2, ly2 = int(g_xs.max()) + 1, int(g_ys.max()) + 1
            local_bbox = BoundingBox(lx1, ly1, lx2, ly2)
            abs_bbox = BoundingBox(x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2)

            cropped_mask = clean_group_mask[ly1:ly2, lx1:lx2]
            seg_area = float(cropped_mask.sum())
            skel_len, mean_thk, max_thk = cls._compute_skeleton_and_thickness(cropped_mask)

            total_skel_len += skel_len
            weighted_thickness_sum += mean_thk * seg_area
            global_max_thickness = max(global_max_thickness, max_thk)

            span_len = float(max(local_bbox.width, local_bbox.height))
            min_dim = float(max(1, min(local_bbox.width, local_bbox.height)))
            aspect_ratio = span_len / min_dim

            orient_deg, sinuosity_std, max_dev, intrusion_ratio = cls._analyze_axis_trajectory(
                cropped_mask, is_curved_site=is_curved_site
            )

            # 동일 선분 축(Corridor) 내 파편화 조각 수 (마젠타 텍스트 가림이 아닌 실제 끊김만 카운트)
            frag_count = len(group)
            if not is_curved_site and span_len > 25:
                num_pure, _, stats_pure, _ = cv2.connectedComponentsWithStats(cropped_mask.astype(np.uint8))
                valid_pure_frags = sum(1 for k in range(1, num_pure) if stats_pure[k][4] >= 15)
                sub_text = layers.text_noise_mask[y1 + ly1:y1 + ly2, x1 + lx1:x1 + lx2]
                if valid_pure_frags >= 2 and int(sub_text.sum()) < 35:
                    frag_count = max(frag_count, valid_pure_frags)

            under_i, under_g = cls._sample_underlying_wafer_signal(
                abs_bbox, layers.gray_wafer, cv2.magnitude(
                    cv2.Sobel(layers.gray_wafer, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(layers.gray_wafer, cv2.CV_32F, 0, 1, ksize=3),
                )
            )

            seg_cx, seg_cy = abs_bbox.center
            seg_rel_cx = float((seg_cx - site.anchor_cx) / max(1.0, site.anchor_w))
            seg_rel_cy = float((seg_cy - site.anchor_cy) / max(1.0, site.anchor_h))

            segments.append(
                SegmentFeature(
                    segment_id=seg_idx,
                    bbox=abs_bbox,
                    rel_cx=seg_rel_cx,
                    rel_cy=seg_rel_cy,
                    area=seg_area,
                    span_length=span_len,
                    skeleton_length=skel_len,
                    mean_thickness=mean_thk,
                    max_thickness=max_thk,
                    aspect_ratio=aspect_ratio,
                    orientation_deg=orient_deg,
                    sinuosity_std=sinuosity_std,
                    max_axis_deviation=max_dev,
                    orthogonal_intrusion_ratio=intrusion_ratio,
                    fragment_count_in_corridor=frag_count,
                    underlying_wafer_intensity=under_i,
                    underlying_wafer_gradient=under_g,
                    is_curved_arc=is_curved_site,
                )
            )

        # 세그먼트를 상대 각도/위치 순으로 일관되게 정렬
        segments.sort(key=lambda s: (round(s.rel_cy, 1), round(s.rel_cx, 1)))

        site_mean_thickness = weighted_thickness_sum / max(1.0, total_red_area)

        # 내재적 계측 무결성 점수(Intrinsic Metrology Integrity Score) 산출:
        # 1) 대칭성(symmetry_score), 2) 파편화 부재, 3) 휨/왜곡 부재, 4) 기저 광학 신호 정합 일관성
        max_sinuosity = max((s.sinuosity_std for s in segments), default=0.0)
        max_frags = max((s.fragment_count_in_corridor for s in segments), default=1)
        min_span_ratio = 1.0
        if len(segments) >= 4:
            spans = [s.span_length for s in segments]
            min_span_ratio = min(spans) / max(1.0, float(np.median(spans)))

        # 세그먼트 간 기저 밝기 일관성 및 기저 패턴 안착도 (예: G3에서 두 패드가 모두 밝은 라인 위에 있는지)
        wafer_consistency = 1.0
        if len(segments) >= 2:
            intensities = [s.underlying_wafer_intensity for s in segments]
            if max(intensities) > 45.0:
                wafer_consistency = min(intensities) / max(1.0, max(intensities))
            # 기저 웨이퍼가 완전히 어두운 허공(intensity < 25 & gradient < 20)에 떠 있는 세그먼트가 있으면 강한 페널티
            if any(s.underlying_wafer_intensity < 25.0 and s.underlying_wafer_gradient < 25.0 for s in segments):
                wafer_consistency *= 0.25

        integrity = (
            0.25 * symmetry_score
            + 0.20 * max(0.0, 1.0 - max(0.0, max_sinuosity - 0.8) / 4.0)
            + 0.15 * (1.0 if max_frags == 1 else 0.3)
            + 0.15 * min(1.0, min_span_ratio)
            + 0.25 * wafer_consistency
        )

        return SiteFeature(
            site_id=site.site_id,
            raw_bbox=site.raw_bbox,
            normalized_bbox=site.padded_bbox,
            is_rescued=site.is_rescued,
            anchor_cx=site.anchor_cx,
            anchor_cy=site.anchor_cy,
            anchor_w=site.anchor_w,
            anchor_h=site.anchor_h,
            total_red_area=round(total_red_area, 1),
            total_cyan_area=round(total_cyan_area, 1),
            total_skeleton_length=round(total_skel_len, 1),
            mean_thickness=round(site_mean_thickness, 2),
            max_thickness=round(global_max_thickness, 2),
            span_x=round(span_x, 1),
            span_y=round(span_y, 1),
            rel_centroid_x=round(rel_cx, 4),
            rel_centroid_y=round(rel_cy, 4),
            angular_sector_areas=[round(v, 1) for v in sector_areas],
            quadrant_areas=[round(v, 1) for v in quadrant_areas],
            symmetry_score=round(symmetry_score, 4),
            intrinsic_integrity_score=round(float(np.clip(integrity, 0.0, 1.0)), 4),
            segments=segments,
        )
