"""Diagnostic & Comparative Visualizer (요구사항 3: 시각적 결함 진단 결과 및 대조 분석 자료 생성).

산출물:
1. 시각적 결함 진단 이미지: 계측 영역(ROI) 바운딩 및 이상 발생 부위 핀포인트 하이라이트 오버레이
2. 대조 분석 자료: 정상 기준 이미지(Golden Reference)와 이상(오계측) 이미지의 1:1 Side-by-Side 전체 및 확대 ROI 비교 시트
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .models import BoundingBox, DefectType, ImageInspectionResult


DEFECT_EN_LABELS: Dict[str, str] = {
    DefectType.POSITION_MISMATCH.value: "POSITION_MISMATCH (계측 위치 불일치)",
    DefectType.LINE_TRUNCATION.value: "LINE_TRUNCATION (선분 단절 및 결손)",
    DefectType.LINE_BLOATING.value: "LINE_BLOATING (선폭 팽창 및 뭉침)",
    DefectType.INTERNAL_FRAGMENTATION.value: "INTERNAL_FRAGMENTATION (내부 파편화)",
    DefectType.ADJACENT_INTRUSION.value: "ADJACENT_INTRUSION (인접 영역 침범)",
    DefectType.PATTERN_DISTORTION.value: "PATTERN_DISTORTION (패턴 왜곡)",
    DefectType.PATTERN_SHRINKAGE_DOT.value: "PATTERN_SHRINKAGE_DOT (패턴 축소 및 점 찍힘)",
    DefectType.OPTICAL_DEFOCUS.value: "OPTICAL_DEFOCUS (광학 초점 불량)",
    DefectType.UNMEASURABLE_RESCUED.value: "UNMEASURABLE_RESCUED (계측 요소 소실/구제)",
}


class DiagnosticVisualizer:
    """오계측 진단 이미지 및 1:1 정상 대조 시트를 생성합니다."""

    _FONT_CACHE: Dict[int, Tuple[ImageFont.FreeTypeFont, bool]] = {}

    @classmethod
    def _get_font(cls, size: int = 16) -> Tuple[ImageFont.ImageFont, bool]:
        """한글 지원 폰트(맑은 고딕/나눔/노토)를 우선 탐색하고 없으면 DejaVu/기본 폰트로 폴백합니다."""
        if size in cls._FONT_CACHE:
            return cls._FONT_CACHE[size]

        kr_candidates = [
            "/mnt/c/Windows/Fonts/malgunbd.ttf",
            "/mnt/c/Windows/Fonts/malgun.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for path in kr_candidates:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size=size)
                    cls._FONT_CACHE[size] = (font, True)
                    return font, True
                except Exception:
                    pass

        en_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for path in en_candidates:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size=size)
                    cls._FONT_CACHE[size] = (font, False)
                    return font, False
                except Exception:
                    pass

        default_font = ImageFont.load_default()
        cls._FONT_CACHE[size] = (default_font, False)
        return default_font, False

    @classmethod
    def _format_text(cls, text_kr: str, text_en: str, supports_kr: bool) -> str:
        return text_kr if supports_kr else text_en

    @classmethod
    def annotate_canvas(
        cls,
        bgr_img: np.ndarray,
        result: ImageInspectionResult,
        is_reference_view: bool = False,
    ) -> np.ndarray:
        """원본 이미지 위에 추출된 계측 영역(Site ROI)과 결함 발생 부위를 시각적으로 표시합니다."""
        canvas = bgr_img.copy()
        H, W = canvas.shape[:2]

        # 1. 추출 및 정규화된 계측 영역(Site ROI) 바운딩 박스 렌더링
        for site in result.sites:
            nb = site.normalized_bbox
            x1, y1, x2, y2 = nb.to_tuple()
            if site.is_rescued:
                # 구제된 영역은 주황/보라색 점선 테두리
                color = (255, 0, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            elif is_reference_view or not result.is_outlier:
                # 정상 기준 영역은 선명한 녹색(Green) 테두리
                color = (40, 200, 40)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            else:
                # 오계측 이미지의 전체 ROI는 황색(Amber) 가이드 테두리
                color = (0, 180, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # 2. 오계측 결함 발생 부위 핀포인트 하이라이트 (Outlier인 경우)
        if not is_reference_view and result.is_outlier:
            overlay = canvas.copy()
            for idx, defect in enumerate(result.defects):
                if defect.region_bbox is None:
                    continue
                db = defect.region_bbox.pad(4, 4, W, H)
                dx1, dy1, dx2, dy2 = db.to_tuple()
                # 결함 부위 반투명 붉은색 강조 + 굵은 적색/노란색 이중 테두리
                cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (0, 0, 255), -1)
                cv2.rectangle(canvas, (dx1 - 2, dy1 - 2), (dx2 + 2, dy2 + 2), (0, 255, 255), 3)
                cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)

            cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

        # 3. PIL ImageDraw를 이용해 깔끔한 라벨 배지 렌더링
        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_canvas)
        draw = ImageDraw.Draw(pil_img)
        font_sm, has_kr = cls._get_font(15)

        for site in result.sites:
            nb = site.normalized_bbox
            tag = (
                cls._format_text(f"ROI #{site.site_id + 1} (구제됨)", f"ROI #{site.site_id + 1} (Rescued)", has_kr)
                if site.is_rescued
                else f"ROI #{site.site_id + 1} ({nb.width}x{nb.height})"
            )
            bx, by = nb.x1, max(2, nb.y1 - 22)
            draw.rectangle([bx, by, bx + len(tag) * 9 + 12, by + 20], fill=(20, 90, 20) if not result.is_outlier or is_reference_view else (140, 90, 0))
            draw.text((bx + 5, by + 2), tag, fill=(255, 255, 255), font=font_sm)

        if not is_reference_view and result.is_outlier:
            for idx, defect in enumerate(result.defects[:4]):
                if defect.region_bbox is None:
                    continue
                db = defect.region_bbox.pad(4, 4, W, H)
                dtype_str = defect.defect_type.value if has_kr else DEFECT_EN_LABELS.get(defect.defect_type.value, "DEFECT").split(" ")[0]
                badge = f"[#{idx + 1} {dtype_str}]"
                tx = max(4, min(W - 180, db.x1))
                ty = min(H - 26, db.y2 + 4)
                draw.rectangle([tx, ty, tx + len(badge) * 11 + 14, ty + 22], fill=(200, 20, 20))
                draw.text((tx + 5, ty + 2), badge, fill=(255, 255, 100), font=font_sm)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @classmethod
    def save_diagnostic_image(
        cls,
        bgr_img: np.ndarray,
        result: ImageInspectionResult,
        output_path: str,
    ) -> str:
        """단일 이미지에 대한 시각적 결함 진단 이미지를 생성하고 저장합니다 (요구사항 3.2)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        annotated = cls.annotate_canvas(bgr_img, result, is_reference_view=False)
        H, W = annotated.shape[:2]

        font_title, has_kr = cls._get_font(18)
        font_body, _ = cls._get_font(14)

        header_h = 110
        header = Image.new("RGB", (W, header_h), color=(28, 32, 40) if not result.is_outlier else (55, 18, 22))
        draw = ImageDraw.Draw(header)

        status_txt = (
            cls._format_text(
                f"판정: 오계측 (OUTLIER)  |  파일: {result.image_name}  |  주원인: {result.primary_defect}",
                f"STATUS: OUTLIER  |  File: {result.image_name}  |  Primary: {DEFECT_EN_LABELS.get(result.primary_defect or '', '')}",
                has_kr,
            )
            if result.is_outlier
            else cls._format_text(
                f"판정: 정상 (NORMAL)  |  파일: {result.image_name}",
                f"STATUS: NORMAL  |  File: {result.image_name}",
                has_kr,
            )
        )
        draw.text((14, 10), status_txt, fill=(255, 90, 90) if result.is_outlier else (90, 240, 120), font=font_title)

        focus_txt = cls._format_text(
            f"광학 초점 검정(독립): {result.focus_result.explanation}",
            f"Focus Check: Score={result.focus_result.sharpness_score:.1f} (Rel={result.focus_result.relative_to_cluster * 100:.1f}%, Defocus={result.focus_result.is_defocused})",
            has_kr,
        )
        draw.text((14, 40), focus_txt, fill=(220, 220, 230), font=font_body)

        if result.defects:
            detail_line = " / ".join(d.description for d in result.defects[:2])
            if not has_kr:
                detail_line = ", ".join(DEFECT_EN_LABELS.get(d.defect_type.value, d.defect_type.value) for d in result.defects)
        else:
            detail_line = cls._format_text("검출된 결함 없음 (모든 계측 요소 정상)", "No defects detected", has_kr)

        draw.text((14, 68), detail_line[:115], fill=(255, 220, 140), font=font_body)

        header_bgr = cv2.cvtColor(np.array(header), cv2.COLOR_RGB2BGR)
        final_img = np.vstack([header_bgr, annotated])
        cv2.imwrite(output_path, final_img)
        return output_path

    @classmethod
    def save_comparison_image(
        cls,
        ref_bgr: np.ndarray,
        ref_result: ImageInspectionResult,
        target_bgr: np.ndarray,
        target_result: ImageInspectionResult,
        output_path: str,
    ) -> str:
        """정상 기준 이미지(Golden Reference)와 이상(오계측) 이미지를 나란히 대조하는 시트를 생성합니다 (요구사항 3.3)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        ref_vis = cls.annotate_canvas(ref_bgr, ref_result, is_reference_view=True)
        tgt_vis = cls.annotate_canvas(target_bgr, target_result, is_reference_view=False)

        # 두 전체 뷰의 높이를 동일하게 맞춤
        target_h = max(ref_vis.shape[0], tgt_vis.shape[0])
        if ref_vis.shape[0] != target_h:
            scale = target_h / float(ref_vis.shape[0])
            ref_vis = cv2.resize(ref_vis, (int(round(ref_vis.shape[1] * scale)), target_h))
        if tgt_vis.shape[0] != target_h:
            scale = target_h / float(tgt_vis.shape[0])
            tgt_vis = cv2.resize(tgt_vis, (int(round(tgt_vis.shape[1] * scale)), target_h))

        divider = np.full((target_h, 8, 3), 200, dtype=np.uint8)
        side_by_side = np.hstack([ref_vis, divider, tgt_vis])
        total_w = side_by_side.shape[1]

        # 하단에 각 사이트(ROI) 1:1 확대 대조 패널(Zoomed ROI Comparison Strip) 생성
        zoom_panels = []
        num_sites = min(len(ref_result.sites), len(target_result.sites))
        for k in range(num_sites):
            rb = ref_result.sites[k].normalized_bbox
            tb = target_result.sites[k].normalized_bbox
            r_crop = ref_vis[rb.y1:rb.y2, rb.x1:rb.x2]
            t_crop = tgt_vis[tb.y1:tb.y2, tb.x1:tb.x2]
            if r_crop.size == 0 or t_crop.size == 0:
                continue
            zh = 240
            rw = max(1, int(round(r_crop.shape[1] * (zh / float(r_crop.shape[0])))))
            tw = max(1, int(round(t_crop.shape[1] * (zh / float(t_crop.shape[0])))))
            r_zoom = cv2.resize(r_crop, (rw, zh))
            t_zoom = cv2.resize(t_crop, (tw, zh))
            sub_div = np.full((zh, 6, 3), (80, 200, 80), dtype=np.uint8)
            pair_zoom = np.hstack([r_zoom, sub_div, t_zoom])
            zoom_panels.append(pair_zoom)

        if zoom_panels:
            combined_zoom = np.hstack([
                p if i == 0 else np.hstack([np.full((p.shape[0], 16, 3), 40, dtype=np.uint8), p])
                for i, p in enumerate(zoom_panels)
            ])
            if combined_zoom.shape[1] > total_w:
                scale = total_w / float(combined_zoom.shape[1])
                combined_zoom = cv2.resize(combined_zoom, (total_w, max(1, int(round(combined_zoom.shape[0] * scale)))))
            elif combined_zoom.shape[1] < total_w:
                pad_left = (total_w - combined_zoom.shape[1]) // 2
                pad_right = total_w - combined_zoom.shape[1] - pad_left
                combined_zoom = cv2.copyMakeBorder(
                    combined_zoom, 12, 12, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(30, 34, 42)
                )
        else:
            combined_zoom = None

        # 상단 대조 정보 헤더 생성
        font_title, has_kr = cls._get_font(20)
        font_sub, _ = cls._get_font(15)
        header_h = 125
        header = Image.new("RGB", (total_w, header_h), color=(25, 30, 38))
        draw = ImageDraw.Draw(header)

        left_title = cls._format_text(
            f"[정상 기준 이미지 (Golden Reference)] {ref_result.image_name}",
            f"[GOLDEN NORMAL REFERENCE] {ref_result.image_name}",
            has_kr,
        )
        right_title = cls._format_text(
            f"[오계측 진단 대상 (Outlier)] {target_result.image_name}",
            f"[DETECTED OUTLIER] {target_result.image_name}",
            has_kr,
        )
        draw.text((20, 12), left_title, fill=(100, 255, 130), font=font_title)
        draw.text((ref_vis.shape[1] + 28, 12), right_title, fill=(255, 100, 100), font=font_title)

        cause_txt = cls._format_text(
            f"검출된 오계측 주원인: {target_result.primary_defect}  |  복합 유형: {', '.join(target_result.defect_types)}",
            f"Primary Defect: {DEFECT_EN_LABELS.get(target_result.primary_defect or '', '')}",
            has_kr,
        )
        draw.text((20, 48), cause_txt, fill=(255, 220, 110), font=font_sub)

        summary_txt = target_result.summary_explanation if has_kr else "; ".join(
            f"Site #{d.site_id + 1}: {DEFECT_EN_LABELS.get(d.defect_type.value, '')}" for d in target_result.defects
        )
        draw.text((20, 80), summary_txt[:155], fill=(220, 225, 235), font=font_sub)

        header_bgr = cv2.cvtColor(np.array(header), cv2.COLOR_RGB2BGR)
        if combined_zoom is not None:
            final_sheet = np.vstack([header_bgr, side_by_side, combined_zoom])
        else:
            final_sheet = np.vstack([header_bgr, side_by_side])

        cv2.imwrite(output_path, final_sheet)
        return output_path
