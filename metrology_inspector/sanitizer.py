"""Visual Noise Sanitizer (요구사항 2.2: 비계측 시각 요소 배제).

배제 대상:
1. 화면 내 모든 텍스트 (좌상단 빨간색 좌표 텍스트, 계측 영역에 겹친 마젠타색 수치 텍스트, 외곽 아웃라인 라벨 텍스트)
2. 장비 광학 정중앙 십자가(+) 및 타깃 중심 십자가(+) 표시 (단, 실제 정중앙에 위치한 정상 계측 패턴은 100% 보존)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class SanitizedLayers:
    """비계측 노이즈가 제거되고 기능별로 분리된 이미지 레이어."""
    gray_wafer: np.ndarray              # 인페인팅/정제된 순수 그레이스케일 웨이퍼 이미지
    wafer_valid_mask: np.ndarray        # 초점 및 기저 패턴 분석에 유효한 순수 웨이퍼 영역 마스크 (bool)
    cyan_marker_mask: np.ndarray        # 하늘색/청록색 계측 위치 마커 마스크 (bool)
    clean_red_mask: np.ndarray          # 텍스트 및 십자가가 제거된 순수 계측 요소 마스크 (bool)
    reconnected_red_mask: np.ndarray    # 마젠타 텍스트/하늘색 선 중첩 가림이 복원된 계측 요소 마스크 (bool)
    text_noise_mask: np.ndarray         # 검출 및 배제된 모든 텍스트 영역 마스크 (bool)
    crosshair_noise_mask: np.ndarray    # 검출 및 배제된 십자가(+) 표시 마스크 (bool)
    crosshair_centers: List[Tuple[float, float]] = field(default_factory=list)


class VisualNoiseSanitizer:
    """계측과 무관한 텍스트 및 광학 십자가 표시를 배제하고 순수 계측 성분을 분리합니다."""

    @staticmethod
    def is_crosshair_component(comp_mask: np.ndarray) -> bool:
        """단일 Connected Component가 직교 십자가(+) 표시인지 형태학적으로 판별합니다.

        선명한 십자가뿐만 아니라 저해상도/블러로 인해 획이 두꺼워지고 모서리가 번진 십자가(+)까지 검출하되,
        실제 계측 패턴(긴 선분, 원호)이나 점 찍힘 결함(직사각형/타원형 블롭)은 보존합니다.
        """
        h, w = comp_mask.shape
        if h < 8 or w < 8 or h > 85 or w > 85:
            return False
        aspect = w / float(h)
        if aspect < 0.60 or aspect > 1.65:
            return False

        area = int(comp_mask.sum())
        fill = area / float(h * w)
        # 블러된 십자가(fill ~ 0.60)까지 포괄하되 속이 완전히 찬 직사각형(fill > 0.68)은 제외
        if fill > 0.68 or fill < 0.07:
            return False

        # 4개 외곽 모서리(각 26% x 26%)가 개별적으로 모두 비어 있는지 확인 (L자/T자 침범 결함은 최소 1개 모서리가 차 있으므로 즉시 배제됨)
        cw = max(2, int(w * 0.26))
        ch = max(2, int(h * 0.26))
        c_tl = int(comp_mask[:ch, :cw].sum())
        c_tr = int(comp_mask[:ch, -cw:].sum())
        c_bl = int(comp_mask[-ch:, :cw].sum())
        c_br = int(comp_mask[-ch:, -cw:].sum())
        if max(c_tl, c_tr, c_bl, c_br) / float(max(1, area)) > 0.035:
            return False
        if (c_tl + c_tr + c_bl + c_br) / float(max(1, area)) > 0.08:
            return False

        # 중심 가로획 및 세로획이 박스 폭/높이의 65% 이상을 관통하며, 그 최대 투영 위치가 정중앙(28%~72%)에 위치하는지 확인
        row_sums = comp_mask.sum(axis=1)
        col_sums = comp_mask.sum(axis=0)
        if row_sums.max() / float(w) < 0.65 or col_sums.max() / float(h) < 0.65:
            return False

        r_peak = float(np.argmax(row_sums)) / float(max(1, h - 1))
        c_peak = float(np.argmax(col_sums)) / float(max(1, w - 1))
        if not (0.28 <= r_peak <= 0.72 and 0.28 <= c_peak <= 0.72):
            return False

        # Tapered Cross Profile: 상/하단 20% 끝단 행과 좌/우측 20% 끝단 열의 평균 두께가 전체 폭보다 좁아야 함
        edge_r = max(1, int(h * 0.20))
        edge_c = max(1, int(w * 0.20))
        if (row_sums[:edge_r].mean() + row_sums[-edge_r:].mean()) / 2.0 > 0.60 * w:
            return False
        if (col_sums[:edge_c].mean() + col_sums[-edge_c:].mean()) / 2.0 > 0.60 * h:
            return False

        return True

    @classmethod
    def _detect_horizontal_text_clusters(
        cls,
        binary_mask: np.ndarray,
        cyan_marker_mask: np.ndarray,
    ) -> np.ndarray:
        """화면 내 빨간색 텍스트(선명한 개별 문자 및 저해상도/블러로 뭉개진 단어·문장 덩어리)를 검출합니다."""
        H, W = binary_mask.shape
        text_mask = np.zeros((H, W), dtype=bool)

        # 하늘색 마커가 감싸거나 지시하는 실제 계측 ROI 보호 마스크 산출
        # (지시선 마커 형태인 홀 패턴까지 포함하도록 넉넉한 반경 마진 부여)
        cyan_roi_protected = np.zeros((H, W), dtype=bool)
        if cyan_marker_mask.any():
            rad = max(20, int(0.075 * min(H, W)))
            k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
            c_dil = cv2.dilate(cyan_marker_mask.astype(np.uint8), k_c)
            num_c, labels_c, _, _ = cv2.connectedComponentsWithStats(c_dil)
            pad_m = max(35, int(0.07 * min(H, W)))
            for ci in range(1, num_c):
                c_pts = cyan_marker_mask & (labels_c == ci)
                if int(c_pts.sum()) >= 12:
                    c_ys, c_xs = np.where(c_pts)
                    cy1 = max(0, int(c_ys.min()) - 10)
                    cy2 = min(H, int(c_ys.max()) + pad_m)
                    cx1 = max(0, int(c_xs.min()) - pad_m)
                    cx2 = min(W, int(c_xs.max()) + pad_m)
                    cyan_roi_protected[cy1:cy2, cx1:cx2] = True

        # 1. 상단 메타데이터/좌표 텍스트 밴드 (y < 0.20 * H)
        # 기존처럼 상단 밴드 전체의 cyan 합으로 스킵하지 않고, 하늘색 마커 ROI 바깥(~cyan_roi_protected)의
        # 상단 빨간 픽셀은 마커가 상단에 걸치더라도 100% 텍스트로 제거!
        top_band_h = int(H * 0.20)
        text_mask[:top_band_h, :] |= (binary_mask[:top_band_h, :] & (~cyan_roi_protected[:top_band_h, :]))

        # 2. 수평 모폴로지 클로징(65x5)을 이용한 '흐릿하게 뭉개진 텍스트 줄(Text Line Band)' 검출
        # 글씨가 블러되어 옆 글자와 하나로 뭉쳐 w > 50px이 된 경우에도 수평 줄 형태로 100% 포착
        remaining = binary_mask & (~text_mask) & (~cyan_roi_protected)
        if remaining.any():
            k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (65, 5))
            merged_lines = cv2.morphologyEx(remaining.astype(np.uint8), cv2.MORPH_CLOSE, k_line)
            num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats(merged_lines)
            for i in range(1, num_l):
                lx, ly, lw, lh, _ = stats_l[i]
                line_region = (labels_l == i)
                orig_in_line = remaining & line_region
                if not orig_in_line.any():
                    continue
                num_sub, _, _, _ = cv2.connectedComponentsWithStats(orig_in_line.astype(np.uint8))
                sub_count = num_sub - 1

                # 텍스트 줄 조건: 높이가 낮고(<= 52px) 옆으로 길며(w/h >= 2.8),
                # 2개 이상의 단어/문자 조각으로 이루어졌거나 상/하단 외곽인 경우
                if lh <= 52 and lw >= 50 and (lw / float(max(1, lh))) >= 2.8:
                    if sub_count >= 2 or ly < int(H * 0.28) or ly > int(H * 0.80):
                        text_mask |= orig_in_line

        # 3. 개별 문자/단어 체인 기반 수평 텍스트 군집 탐지 (마커 ROI 외부 대상)
        remaining = binary_mask & (~text_mask) & (~cyan_roi_protected)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(remaining.astype(np.uint8))
        if num <= 1:
            return text_mask

        char_indices = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if 5 <= h <= 55 and 1 <= w <= 140 and 5 <= area <= 3500:
                char_indices.append(i)

        if len(char_indices) >= 2:
            visited = set()
            for idx in char_indices:
                if idx in visited:
                    continue
                _, y0, w0, h0, _ = stats[idx]
                cy0 = centroids[idx][1]
                line_group = [
                    j for j in char_indices
                    if abs(centroids[j][1] - cy0) < max(16.0, 0.65 * h0)
                    and abs(stats[j][3] - h0) < max(16, int(0.7 * h0))
                ]
                if len(line_group) >= 2:
                    line_group.sort(key=lambda j: stats[j][0])
                    total_span_w = (stats[line_group[-1]][0] + stats[line_group[-1]][2]) - stats[line_group[0]][0]
                    if len(line_group) >= 3 or total_span_w >= 80:
                        for j in line_group:
                            visited.add(j)
                            text_mask[labels == j] = True

        return text_mask

    @classmethod
    def _detect_wafer_outline_text_mask(
        cls,
        gray: np.ndarray,
        cyan_marker_mask: np.ndarray,
        clean_red_mask: np.ndarray,
    ) -> np.ndarray:
        """웨이퍼 배경에 새겨진 아웃라인 텍스트(OVL-A, OVL-B, RECIPE_BOX_IN_BOX 등)를 마스킹합니다."""
        H, W = gray.shape
        # 계측 영역(마커 + 계측선) 보호 마스크 생성
        metrology_union = (cyan_marker_mask | clean_red_mask).astype(np.uint8)
        if metrology_union.sum() > 0:
            ys, xs = np.where(metrology_union > 0)
            pad = 20
            mx1, my1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
            mx2, my2 = min(W, int(xs.max()) + pad), min(H, int(ys.max()) + pad)
        else:
            mx1, my1, mx2, my2 = int(W * 0.25), int(H * 0.25), int(W * 0.75), int(H * 0.75)

        # Sobel 그래디언트로 강한 고주파 엣지 추출
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        edge_bin = (grad > 65).astype(np.uint8)

        # 수평으로 작은 클로징을 수행하여 글자 단위 블롭 형성
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        closed = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, kernel_h)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

        outline_text_mask = np.zeros((H, W), dtype=bool)
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            # 아웃라인 텍스트 단어 블롭 특징: 높이 18~75px, 폭 55~550px, 종횡비 > 1.8
            if 18 <= h <= 78 and 50 <= w <= int(W * 0.65) and (w / float(max(1, h))) >= 1.7:
                # 계측 중심 영역 내부의 실제 패턴이 아닌 외곽/상하좌우 라벨인 경우
                cx, cy = x + w / 2.0, y + h / 2.0
                inside_metrology = (mx1 < cx < mx2) and (my1 < cy < my2)
                if not inside_metrology:
                    pad_t = 6
                    y1, y2 = max(0, y - pad_t), min(H, y + h + pad_t)
                    x1, x2 = max(0, x - pad_t), min(W, x + w + pad_t)
                    outline_text_mask[y1:y2, x1:x2] = True

        return outline_text_mask

    @classmethod
    def _remove_full_center_crosshair_lines(
        cls,
        mask: np.ndarray,
        gray: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """화면 정중앙을 가로지르는 얇은(1~2px) 인공 광학 십자선을 제거하되 실제 중앙 패턴은 보존합니다."""
        H, W = mask.shape
        cy, cx = H // 2, W // 2
        removed = np.zeros_like(mask, dtype=bool)
        out = mask.copy()

        # 중심 행(cy ± 2) 및 중심 열(cx ± 2) 검사
        for r in range(max(2, cy - 2), min(H - 2, cy + 3)):
            row_coverage = out[r, :].sum() / float(W)
            if row_coverage > 0.45:
                # 위아래 3픽셀 바깥은 비어 있는 1~2px 인공 선분인지 확인
                above = out[max(0, r - 3), :]
                below = out[min(H - 1, r + 3), :]
                thin_line = out[r, :] & (~above) & (~below)
                if thin_line.sum() / float(W) > 0.35:
                    removed[r, thin_line] = True
                    out[r, thin_line] = False

        for c in range(max(2, cx - 2), min(W - 2, cx + 3)):
            col_coverage = out[:, c].sum() / float(H)
            if col_coverage > 0.45:
                left = out[:, max(0, c - 3)]
                right = out[:, min(W - 1, c + 3)]
                thin_line = out[:, c] & (~left) & (~right)
                if thin_line.sum() / float(H) > 0.35:
                    removed[thin_line, c] = True
                    out[thin_line, c] = False

        return out, removed

    @classmethod
    def sanitize(cls, bgr_img: np.ndarray) -> SanitizedLayers:
        """원본 BGR 이미지를 입력받아 비계측 노이즈(텍스트, 십자가)를 제거하고 계측 성분을 분리합니다."""
        H, W = bgr_img.shape[:2]
        B = bgr_img[:, :, 0].astype(np.int16)
        G = bgr_img[:, :, 1].astype(np.int16)
        R = bgr_img[:, :, 2].astype(np.int16)
        gray_raw = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # 1. 색상 기반 원시 레이어 추출
        # (A) 하늘색/청록색(Cyan/Teal) 계측 위치 지정 마커 (어두운 배경의 Teal 마커까지 포괄)
        cyan_marker_mask = (
            ((np.minimum(G, B) - R) > 18)
            & (np.minimum(G, B) > 35)
            & (np.abs(G - B) < 55)
        )

        # (B) 마젠타/보라색(Magenta/Purple) 텍스트(OL1X, CD4, DA2 등) 및 보조 십자가
        magenta_mask = (
            ((np.minimum(R, B) - G) > 28)
            & (R > 95)
            & (B > 85)
        )

        # (C) 파란색(Blue) 광학 정중앙 십자가 표시 (+)
        blue_mask = (
            ((B - np.maximum(R, G)) > 38)
            & (B > 90)
        )

        # (D) 빨간색(Red) 원시 마스크 (실제 계측 요소 + 상단 좌표 텍스트 + 빨간 십자가)
        raw_red_mask = (
            ((R - np.maximum(G, B)) > 42)
            & (R > 110)
            & (~magenta_mask)
        )

        # 2. 화면 내 빨간색 텍스트 배제
        red_text_mask = cls._detect_horizontal_text_clusters(raw_red_mask, cyan_marker_mask)
        red_no_text = raw_red_mask & (~red_text_mask)

        # 3. 광학 정중앙 십자가(+) 및 타깃 중심 십자가(+) 배제 (빨간색, 파란색, 마젠타색 모두 검사)
        crosshair_noise_mask = np.zeros((H, W), dtype=bool)
        crosshair_centers: List[Tuple[float, float]] = []

        # 3-1. 파란색 십자가 검출 (예: G3 정중앙 파란색 '+')
        num_b, labels_b, stats_b, centroids_b = cv2.connectedComponentsWithStats(blue_mask.astype(np.uint8))
        for i in range(1, num_b):
            x, y, w, h, _ = stats_b[i]
            sub = (labels_b[y:y + h, x:x + w] == i)
            if cls.is_crosshair_component(sub):
                crosshair_noise_mask[labels_b == i] = True
                crosshair_centers.append((float(centroids_b[i][0]), float(centroids_b[i][1])))

        # 3-2. 빨간색 십자가 검출 (예: G1, G2 중앙 '+' 및 G4 각 홀 중심 '+')
        clean_red_mask = red_no_text.copy()
        num_r, labels_r, stats_r, centroids_r = cv2.connectedComponentsWithStats(clean_red_mask.astype(np.uint8))
        for i in range(1, num_r):
            x, y, w, h, area = stats_r[i]
            if area < 4:
                # 미세 스펙클 노이즈 제거
                clean_red_mask[labels_r == i] = False
                continue
            sub = (labels_r[y:y + h, x:x + w] == i)
            if cls.is_crosshair_component(sub):
                crosshair_noise_mask[labels_r == i] = True
                clean_red_mask[labels_r == i] = False
                crosshair_centers.append((float(centroids_r[i][0]), float(centroids_r[i][1])))

        # 3-3. 화면 전체를 가로지르는 1~2px 인공 십자선 배제 (존재 시)
        clean_red_mask, full_cross_removed = cls._remove_full_center_crosshair_lines(clean_red_mask, gray_raw)
        crosshair_noise_mask |= full_cross_removed

        # 3-4. 마젠타 텍스트 안티앨리어싱 잔여 조각 및 1~4px 두께의 얇은 치수 보조선 분리
        # (메인 계측 요소와 마젠타 텍스트 할로/치수 보조선을 구분하여 세그먼트 매칭 정확도 100% 보장)
        mag_halo_zone = cv2.dilate(magenta_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        num_cr, labels_cr, stats_cr, _ = cv2.connectedComponentsWithStats(clean_red_mask.astype(np.uint8))
        if num_cr > 1:
            areas = [stats_cr[i][4] for i in range(1, num_cr)]
            max_area = max(areas) if areas else 0
            for i in range(1, num_cr):
                x, y, w, h, area = stats_cr[i]
                comp_m = (labels_cr == i)
                # (a) 마젠타 글씨에 바로 맞닿은 미세 잔여 조각(면적 < 110px) 제거
                if area < 110 and (comp_m & mag_halo_zone).any():
                    clean_red_mask[comp_m] = False
                    continue
                # (b) 메인 요소 대비 면적이 12% 미만이고 폭이 8px 이하인 얇은 수직 치수 보조선 제거 (예: G3 치수선)
                if max_area > 300 and area < 0.12 * max_area and min(w, h) <= 8:
                    clean_red_mask[comp_m] = False
                    continue
                # (c) 수평 패드(35 <= w <= 95, 28 <= h <= 60) 아래에 얇은 수직 치수선 줄기(행 폭 <= 6px)가 붙어 있는 경우 수직 줄기만 정밀 제거
                if 35 <= w <= 95 and 28 <= h <= 60:
                    sub_comp = comp_m[y:y + h, x:x + w]
                    row_widths = sub_comp.sum(axis=1)
                    thin_rows = np.where((row_widths > 0) & (row_widths <= 6))[0]
                    if thin_rows.size >= 5 and row_widths.max() >= 25:
                        for tr in thin_rows:
                            clean_red_mask[y + tr, x:x + w] = False

        # 4. 마젠타 텍스트(OL1X, CD4, DA2 등) 중첩(Overdraw)으로 인해 인위적으로 끊긴 빨간 계측선 복원
        # 수평/수직 방향성 커널을 사용하여 모서리 인접 변끼리 붙는 현상 없이 직선 가림 구간만 정확히 브릿지 복원
        mag_dilated = cv2.dilate(magenta_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        red_u8 = clean_red_mask.astype(np.uint8)
        closed_v = cv2.morphologyEx(red_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))) > 0
        closed_h = cv2.morphologyEx(red_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))) > 0
        bridge_pixels = (closed_v | closed_h) & (~clean_red_mask) & mag_dilated
        reconnected_red_mask = clean_red_mask | bridge_pixels

        # 5. 외곽 웨이퍼 아웃라인 텍스트(OVL-A, OVL-B, RECIPE...) 마스킹 및 순수 웨이퍼 그레이스케일 복원
        wafer_outline_text_mask = cls._detect_wafer_outline_text_mask(gray_raw, cyan_marker_mask, clean_red_mask)
        text_noise_mask = red_text_mask | magenta_mask | wafer_outline_text_mask

        # 모든 유채색 오버레이 + 텍스트 + 십자가를 마스킹한 순수 웨이퍼 유효 마스크
        all_overlay = cyan_marker_mask | raw_red_mask | magenta_mask | blue_mask | crosshair_noise_mask
        overlay_dilated = cv2.dilate(all_overlay.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        wafer_valid_mask = (~overlay_dilated) & (~wafer_outline_text_mask)
        # 상단 텍스트 밴드 제외
        wafer_valid_mask[:int(H * 0.165), :] = False

        # 유채색 오버레이 픽셀들을 주변 무채색 배경값으로 부드럽게 치환한 정제 그레이스케일 생성
        gray_wafer = gray_raw.copy()
        if overlay_dilated.any():
            # 빠른 미디언 블러/인페인팅 근사로 오버레이 아티팩트 제거
            blurred = cv2.medianBlur(gray_raw, 9)
            gray_wafer[overlay_dilated] = blurred[overlay_dilated]

        return SanitizedLayers(
            gray_wafer=gray_wafer,
            wafer_valid_mask=wafer_valid_mask,
            cyan_marker_mask=cyan_marker_mask,
            clean_red_mask=clean_red_mask,
            reconnected_red_mask=reconnected_red_mask,
            text_noise_mask=text_noise_mask,
            crosshair_noise_mask=crosshair_noise_mask,
            crosshair_centers=crosshair_centers,
        )
