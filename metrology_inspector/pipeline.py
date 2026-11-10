"""Main Batch Inspection Pipeline & Report Generator (요구사항 2.1 ~ 2.7 및 3).

대용량 이미지 스트리밍 처리, 동일 조건 군집 자동 매칭, 결측 영역 구제,
독립 병렬 초점 검정, 7대 결함 분류, 시각적 진단 및 1:1 대조 자료 생성,
상세 판정 보고서(Markdown / JSON / CSV) 출력을 총괄합니다.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import csv
import gc
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from .defect_classifier import ClusterDefectClassifier
from .feature_extractor import MetrologyFeatureExtractor
from .focus_analyzer import IndependentFocusEvaluator
from .models import BoundingBox, FocusResult, ImageInspectionResult, SiteFeature
from .roi_localizer import ROILocalizer
from .sanitizer import SanitizedLayers, VisualNoiseSanitizer
from .visualizer import DiagnosticVisualizer


@dataclass
class _Pass1Item:
    """Pass 1에서 추출된 이미지별 경량 중간 데이터."""
    index: int
    image_path: str
    image_name: str
    height: int
    width: int
    bg_median_gray: float
    cyan_span: float
    site_count: int
    focus_result: FocusResult
    raw_sites: List[BoundingBox]
    layers: SanitizedLayers


class MetrologyInspectionPipeline:
    """반도체 계측 이미지 이상치(오계측) 자동 검출 파이프라인."""

    def __init__(
        self,
        auto_group_recipes: bool = True,
        save_visuals_for_normal: bool = False,
    ) -> None:
        self.auto_group_recipes = auto_group_recipes
        self.save_visuals_for_normal = save_visuals_for_normal

    @staticmethod
    def _cluster_by_recipe_condition(items: List[_Pass1Item]) -> Dict[str, List[_Pass1Item]]:
        """동일 조건에서 수집된 이미지 군집을 해상도 비율, 배경 명암, 마커 스케일 기준으로 자동 그룹화합니다.

        초점 상실이나 패턴 소실로 인해 마커가 검출되지 않은 이미지(site_count == 0)는
        해상도 비율과 배경 밝기가 가장 유사한 정상 군집으로 자동 편입시켜 구제(Rescue)가 작동하도록 보장합니다.
        """
        if not items:
            return {}

        clusters: List[Dict[str, object]] = []
        unassigned_empty: List[_Pass1Item] = []

        for item in items:
            if item.site_count == 0 or item.cyan_span < 10.0:
                unassigned_empty.append(item)
                continue

            aspect = item.width / float(max(1, item.height))
            is_dark_bg = item.bg_median_gray < 90.0
            rel_cyan_span = item.cyan_span / float(max(1, min(item.width, item.height)))

            matched_cluster = None
            for cl in clusters:
                c_aspect = float(cl["aspect"])
                c_dark = bool(cl["is_dark_bg"])
                c_sites = int(cl["site_count"])
                c_span = float(cl["rel_cyan_span"])

                if (
                    c_dark == is_dark_bg
                    and abs(c_aspect - aspect) < 0.12
                    and (c_sites == item.site_count or min(c_sites, item.site_count) >= 2)
                    and abs(c_span - rel_cyan_span) < 0.065
                ):
                    matched_cluster = cl
                    break

            if matched_cluster is not None:
                member_list: List[_Pass1Item] = matched_cluster["members"]  # type: ignore[assignment]
                member_list.append(item)
            else:
                clusters.append({
                    "id": f"Cluster_{len(clusters) + 1:02d}",
                    "aspect": aspect,
                    "is_dark_bg": is_dark_bg,
                    "site_count": item.site_count,
                    "rel_cyan_span": rel_cyan_span,
                    "members": [item],
                })

        # 패턴/마커가 소실된 결측 이미지들을 해상도/배경 조건이 가장 유사한 군집에 편입 (구제용)
        if not clusters and unassigned_empty:
            return {"Cluster_01": unassigned_empty}

        for emp in unassigned_empty:
            aspect = emp.width / float(max(1, emp.height))
            is_dark_bg = emp.bg_median_gray < 90.0
            best_cl = min(
                clusters,
                key=lambda cl: (
                    0.0 if bool(cl["is_dark_bg"]) == is_dark_bg else 10.0
                ) + abs(float(cl["aspect"]) - aspect),
            )
            best_cl["members"].append(emp)  # type: ignore[ union-attr ]

        return {str(cl["id"]): list(cl["members"]) for cl in clusters}  # type: ignore[arg-type]

    def inspect_batch(
        self,
        image_paths: Sequence[str],
        output_dir: Optional[str] = None,
    ) -> List[ImageInspectionResult]:
        """다수의 이미지 경로를 입력받아 오계측 자동 검출 및 시각적 진단/보고서 산출을 일괄 수행합니다."""
        valid_paths = [p for p in image_paths if os.path.isfile(p)]
        if not valid_paths:
            return []

        # Pass 1: 개별 이미지 로드, 비계측 요소 배제, 독립 초점 검정, 원시 ROI 추출
        pass1_items: List[_Pass1Item] = []
        for idx, path in enumerate(valid_paths):
            bgr = cv2.imread(path)
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            layers = VisualNoiseSanitizer.sanitize(bgr)
            focus_res = IndependentFocusEvaluator.evaluate_single(
                layers.gray_wafer, layers.wafer_valid_mask
            )
            raw_sites = ROILocalizer.extract_raw_sites(layers)

            bg_med = float(np.median(layers.gray_wafer))
            if int(layers.cyan_marker_mask.sum()) > 20:
                c_ys, c_xs = np.where(layers.cyan_marker_mask)
                cyan_span = float(max(c_xs.max() - c_xs.min(), c_ys.max() - c_ys.min()))
            elif raw_sites:
                cyan_span = float(max(raw_sites[0].width, raw_sites[0].height))
            else:
                cyan_span = 0.0

            pass1_items.append(
                _Pass1Item(
                    index=idx,
                    image_path=path,
                    image_name=os.path.basename(path),
                    height=H,
                    width=W,
                    bg_median_gray=bg_med,
                    cyan_span=cyan_span,
                    site_count=len(raw_sites),
                    focus_result=focus_res,
                    raw_sites=raw_sites,
                    layers=layers,
                )
            )
            # 원본 BGR 배열 즉시 해제 (대용량 처리 시 OOM 방지)
            del bgr

        if not pass1_items:
            return []

        # 군집 분리 (단일 배치 강제 또는 레시피 조건별 자동 군집화)
        if self.auto_group_recipes and len(pass1_items) > 2:
            grouped_items = self._cluster_by_recipe_condition(pass1_items)
        else:
            grouped_items = {"Cluster_01": pass1_items}

        # Pass 2: 군집별 영역 정규화/구제, 초점 캘리브레이션, 피처 추출 및 7대 결함 분류
        results_by_index: Dict[int, ImageInspectionResult] = {}

        for cluster_id, group in grouped_items.items():
            c_focus_list = [it.focus_result for it in group]
            IndependentFocusEvaluator.calibrate_cluster_focus(c_focus_list)

            c_raw_sites = [it.raw_sites for it in group]
            c_shapes = [(it.height, it.width) for it in group]
            c_layers = [it.layers for it in group]

            localized_batch = ROILocalizer.normalize_and_rescue_batch(
                c_raw_sites, c_shapes, c_layers
            )

            sites_features_batch: List[List[SiteFeature]] = []
            for it, loc_sites in zip(group, localized_batch):
                img_site_feats = [
                    MetrologyFeatureExtractor.extract_site_features(ls, it.layers)
                    for ls in loc_sites
                ]
                sites_features_batch.append(img_site_feats)

            c_paths = [it.image_path for it in group]
            c_names = [it.image_name for it in group]
            cluster_results = ClusterDefectClassifier.classify_cluster(
                image_paths=c_paths,
                image_names=c_names,
                cluster_id=cluster_id,
                image_shapes=c_shapes,
                focus_results=c_focus_list,
                sites_per_image=sites_features_batch,
            )

            for it, res in zip(group, cluster_results):
                results_by_index[it.index] = res

        # 중간 마스크 메모리 해제
        del pass1_items
        gc.collect()

        ordered_results = [results_by_index[i] for i in sorted(results_by_index.keys())]

        # 산출물 저장 (요구사항 3: 상세 보고서, 시각적 결함 진단 결과, 1:1 대조 분석 자료)
        if output_dir is not None:
            self.generate_outputs(ordered_results, output_dir)

        return ordered_results

    def generate_outputs(
        self,
        results: List[ImageInspectionResult],
        output_dir: str,
    ) -> Dict[str, str]:
        """검출 결과를 바탕으로 진단 이미지, 1:1 대조 이미지 및 상세 리포트(MD, JSON, CSV)를 생성합니다."""
        os.makedirs(output_dir, exist_ok=True)
        diag_dir = os.path.join(output_dir, "diagnostics")
        comp_dir = os.path.join(output_dir, "comparisons")
        os.makedirs(diag_dir, exist_ok=True)
        os.makedirs(comp_dir, exist_ok=True)

        # 이름으로 결과 빠르게 조회하기 위한 맵
        res_by_path: Dict[str, ImageInspectionResult] = {r.image_path: r for r in results}

        for res in results:
            if not res.is_outlier and not self.save_visuals_for_normal:
                continue

            bgr = cv2.imread(res.image_path)
            if bgr is None:
                continue

            stem = os.path.splitext(res.image_name)[0]
            diag_path = os.path.join(diag_dir, f"{stem}_diagnostic.jpg")
            DiagnosticVisualizer.save_diagnostic_image(bgr, res, diag_path)
            res.diagnostic_image_path = diag_path

            if res.is_outlier and res.reference_image_path and os.path.isfile(res.reference_image_path):
                ref_bgr = cv2.imread(res.reference_image_path)
                ref_res = res_by_path.get(res.reference_image_path, res)
                if ref_bgr is not None:
                    comp_path = os.path.join(comp_dir, f"{stem}_vs_golden_comparison.jpg")
                    DiagnosticVisualizer.save_comparison_image(
                        ref_bgr, ref_res, bgr, res, comp_path
                    )
                    res.comparison_image_path = comp_path
                del ref_bgr

            del bgr

        # 1. JSON 상세 보고서 저장
        json_path = os.path.join(output_dir, "inspection_report.json")
        serializable = [asdict(r) for r in results]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_images": len(results),
                    "normal_count": sum(1 for r in results if not r.is_outlier),
                    "outlier_count": sum(1 for r in results if r.is_outlier),
                    "results": serializable,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 2. CSV 요약 보고서 저장
        csv_path = os.path.join(output_dir, "inspection_report.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Image Name",
                "Cluster ID",
                "Status",
                "Primary Defect",
                "All Defect Types",
                "Focus Defocused",
                "Focus Score",
                "Site Count",
                "Rescued ROI",
                "Golden Reference",
                "Summary Explanation",
                "Diagnostic Image",
                "Comparison Image",
            ])
            for r in results:
                writer.writerow([
                    r.image_name,
                    r.cluster_id,
                    r.status_label,
                    r.primary_defect or "NONE",
                    ", ".join(r.defect_types) if r.defect_types else "NONE",
                    "YES" if r.focus_result.is_defocused else "NO",
                    r.focus_result.sharpness_score,
                    len(r.sites),
                    "YES" if any(s.is_rescued for s in r.sites) else "NO",
                    r.reference_image_name or "",
                    r.summary_explanation,
                    r.diagnostic_image_path or "",
                    r.comparison_image_path or "",
                ])

        # 3. Markdown 상세 보고서 저장
        md_path = os.path.join(output_dir, "INSPECTION_REPORT.md")
        self._write_markdown_report(results, md_path)

        return {
            "json_report": json_path,
            "csv_report": csv_path,
            "markdown_report": md_path,
            "diagnostics_dir": diag_dir,
            "comparisons_dir": comp_dir,
        }

    @staticmethod
    def _write_markdown_report(results: List[ImageInspectionResult], md_path: str) -> None:
        total = len(results)
        normals = sum(1 for r in results if not r.is_outlier)
        outliers = total - normals

        lines = [
            "# 반도체 계측 이미지 이상치(오계측) 자동 검출 결과 보고서",
            "",
            "## 1. 종합 검정 요약",
            f"- **총 검사 이미지 수**: {total}장",
            f"- **정상(NORMAL) 판정**: {normals}장",
            f"- **오계측(OUTLIER) 판정**: {outliers}장",
            "",
            "## 2. 이미지별 판정 요약표",
            "",
            "| 번호 | 이미지 파일명 | 군집 ID | 최종 판정 | 광학 초점 상태 (점수) | 검출 영역(Site) 수 | 주된 오계측 원인 | 복합 결함 유형 | 기준 정상 이미지 |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |",
        ]

        for idx, r in enumerate(results, 1):
            status_badge = "**오계측 (OUTLIER)**" if r.is_outlier else "정상 (NORMAL)"
            focus_badge = (
                f"초점불량 ({r.focus_result.sharpness_score:.1f})"
                if r.focus_result.is_defocused
                else f"정상 ({r.focus_result.sharpness_score:.1f})"
            )
            primary = r.primary_defect or "-"
            all_types = ", ".join(r.defect_types) if r.defect_types else "-"
            lines.append(
                f"| {idx} | `{r.image_name}` | `{r.cluster_id}` | {status_badge} | "
                f"{focus_badge} | {len(r.sites)}개 | {primary} | {all_types} | `{r.reference_image_name}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. 오계측(OUTLIER) 검출 이미지 상세 진단 내역",
            "",
        ])

        outlier_list = [r for r in results if r.is_outlier]
        if not outlier_list:
            lines.append("- 검출된 오계측 이미지가 없습니다.")
        else:
            for r in outlier_list:
                lines.extend([
                    f"### [{r.cluster_id}] `{r.image_name}`",
                    f"- **최종 판정**: **오계측 (OUTLIER)**",
                    f"- **주된 이상 원인**: **{r.primary_defect}**",
                    f"- **검출된 전체 결함 유형**: {', '.join(r.defect_types)}",
                    f"- **독립 광학 초점 검정**: {r.focus_result.explanation}",
                    f"- **비교 기준 정상 이미지**: `{r.reference_image_name}`",
                    f"- **시각적 결함 진단 결과**: `{r.diagnostic_image_path}`",
                    f"- **1:1 정상 대조 분석 자료**: `{r.comparison_image_path}`",
                    "- **세부 결함 분석**:",
                ])
                for d_idx, d in enumerate(r.defects, 1):
                    lines.append(f"  {d_idx}. **[{d.defect_type.value}]** (심각도 {d.severity_score:.2f}): {d.description}")
                lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
