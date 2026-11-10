#!/usr/bin/env python3
"""1. 특정 타겟 이미지 단일 검정 스크립트 (군집 내 상대 비교 기반).

사용법:
    python3 inspect_target.py <sample_directory_path> <타겟이미지파일명> [-o output_dir]

예시:
    python3 inspect_target.py samples e16b9a33-8bcf-4ab6-9810-78a6b849f140.jpg
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List, Optional
import cv2

from metrology_inspector.models import ImageInspectionResult
from metrology_inspector.pipeline import MetrologyInspectionPipeline
from metrology_inspector.visualizer import DiagnosticVisualizer


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_directory_images(sample_dir: str) -> List[str]:
    if not os.path.isdir(sample_dir):
        return []
    paths: List[str] = []
    for entry in sorted(os.listdir(sample_dir)):
        full = os.path.join(sample_dir, entry)
        if os.path.isfile(full) and entry.lower().endswith(SUPPORTED_EXTS):
            paths.append(full)
    return paths


def resolve_target_image(all_paths: List[str], target_arg: str) -> Optional[str]:
    """파일명, 확장자 생략 이름, 또는 외부 파일 경로 인자로부터 대상 이미지 경로를 찾습니다."""
    target_base = os.path.basename(target_arg)
    target_stem = os.path.splitext(target_base)[0]

    # 1. 디렉토리 내 정확한 파일명 일치
    for p in all_paths:
        if os.path.basename(p) == target_base:
            return p

    # 2. 디렉토리 내 확장자 제외 파일명(stem) 일치 또는 부분 문자열 일치
    for p in all_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem == target_stem or target_base in os.path.basename(p):
            return p

    # 3. 직접 지정된 외부 파일 경로가 존재하는 경우 해당 파일 경로 반환
    if os.path.isfile(target_arg):
        return os.path.abspath(target_arg)

    return None


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="샘플 디렉토리 군집 내에서 지정한 특정 타겟 이미지 1장에 대해 오계측 여부를 검정합니다."
    )
    parser.add_argument(
        "sample_directory_path",
        help="이미지들이 들어있는 샘플 디렉토리 경로",
    )
    parser.add_argument(
        "target_image_name",
        help="검사할 특정 타겟 이미지 파일명 (예: e16b9a33-8bcf-4ab6-9810-78a6b849f140.jpg)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output_target",
        help="타겟 이미지 검정 결과 및 시각화 자료를 저장할 폴더 (기본값: output_target)",
    )

    args = parser.parse_args(argv)

    if not os.path.isdir(args.sample_directory_path):
        print(f"[오류] 디렉토리를 찾을 수 없습니다: {args.sample_directory_path}", file=sys.stderr)
        return 1

    all_images = list_directory_images(args.sample_directory_path)
    if not all_images:
        print(f"[오류] 디렉토리 내에 유효한 이미지 파일이 없습니다: {args.sample_directory_path}", file=sys.stderr)
        return 1

    target_path = resolve_target_image(all_images, args.target_image_name)
    if target_path is None:
        print(
            f"[오류] '{args.sample_directory_path}' 내에서 타겟 이미지 '{args.target_image_name}'를 찾을 수 없습니다.",
            file=sys.stderr,
        )
        return 1

    if target_path not in all_images:
        all_images.append(target_path)

    target_name = os.path.basename(target_path)
    print("=" * 88)
    print(f"[단일 타겟 검정 시작] 대상 디렉토리: {args.sample_directory_path} (총 {len(all_images)}장 군집 참조)")
    print(f"[타겟 이미지 파일]   : {target_name}")
    print("=" * 88)

    # 군집 비교를 수행하여 정상 기준을 확립한 뒤 타겟 이미지의 결과를 추출
    pipeline = MetrologyInspectionPipeline(auto_group_recipes=True, save_visuals_for_normal=True)
    all_results = pipeline.inspect_batch(image_paths=all_images, output_dir=None)

    res_by_path = {r.image_path: r for r in all_results}
    target_res: Optional[ImageInspectionResult] = res_by_path.get(target_path)
    if target_res is None:
        print(f"[오류] 타겟 이미지 검정 실패: {target_path}", file=sys.stderr)
        return 1

    # 타겟 이미지 전용 산출물(보고서, 진단 이미지, 1:1 대조 이미지) 생성
    os.makedirs(args.output_dir, exist_ok=True)
    diag_dir = os.path.join(args.output_dir, "diagnostics")
    comp_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(diag_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    stem = os.path.splitext(target_res.image_name)[0]
    target_bgr = cv2.imread(target_res.image_path)
    if target_bgr is not None:
        diag_path = os.path.join(diag_dir, f"{stem}_diagnostic.jpg")
        DiagnosticVisualizer.save_diagnostic_image(target_bgr, target_res, diag_path)
        target_res.diagnostic_image_path = diag_path

        if target_res.reference_image_path and os.path.isfile(target_res.reference_image_path):
            ref_bgr = cv2.imread(target_res.reference_image_path)
            ref_res = res_by_path.get(target_res.reference_image_path, target_res)
            if ref_bgr is not None:
                comp_path = os.path.join(comp_dir, f"{stem}_vs_golden_comparison.jpg")
                DiagnosticVisualizer.save_comparison_image(
                    ref_bgr, ref_res, target_bgr, target_res, comp_path
                )
                target_res.comparison_image_path = comp_path

    # 타겟 전용 리포트(MD/JSON/CSV) 기록
    pipeline.generate_outputs([target_res], args.output_dir)

    status_ko = "오계측 (OUTLIER)" if target_res.is_outlier else "정상 (NORMAL)"
    print(f"• 타겟 이미지   : {target_res.image_name}")
    print(f"• 소속 레시피군 : {target_res.cluster_id} (비교 기준 정상 이미지: {target_res.reference_image_name})")
    print(f"• 최종 판정     : {status_ko}")
    print(f"• 광학 초점 상태: {target_res.focus_result.explanation}")
    print(f"• 검출 영역 수  : {len(target_res.sites)}개")
    if target_res.is_outlier:
        print(f"• 주된 이상 원인: {target_res.primary_defect}")
        print(f"• 전체 결함 유형: {', '.join(target_res.defect_types)}")
        print("• 세부 진단 내역:")
        for idx, d in enumerate(target_res.defects, 1):
            print(f"   {idx}. [{d.defect_type.value}] (심각도 {d.severity_score:.2f}) {d.description}")
    else:
        print(f"• 판정 요약     : {target_res.summary_explanation}")

    print("-" * 88)
    print(f"- 시각적 진단 이미지: {target_res.diagnostic_image_path}")
    print(f"- 1:1 대조 분석 자료: {target_res.comparison_image_path}")
    print(f"- 상세 보고서 (MD)  : {os.path.join(args.output_dir, 'INSPECTION_REPORT.md')}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
