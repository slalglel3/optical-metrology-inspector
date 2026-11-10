#!/usr/bin/env python3
"""2. 샘플 디렉토리 내 전체 이미지 순차 검정 스크립트.

사용법:
    python3 inspect_all.py <sample_directory_path> [-o output_dir]

예시:
    python3 inspect_all.py samples
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List

from metrology_inspector.pipeline import MetrologyInspectionPipeline


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


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="샘플 디렉토리 내의 모든 계측 이미지에 대해 순차적으로 오계측 여부를 전수 검사합니다."
    )
    parser.add_argument(
        "sample_directory_path",
        help="검사할 이미지들이 들어있는 샘플 디렉토리 경로",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output_report",
        help="전수 검사 보고서 및 시각화 자료를 저장할 폴더 (기본값: output_report)",
    )
    parser.add_argument(
        "--single-cluster",
        action="store_true",
        help="자동 레시피 군집 분리를 끄고 디렉토리 내 모든 이미지를 단일 군집으로 비교",
    )
    parser.add_argument(
        "--save-normal-visuals",
        action="store_true",
        help="오계측뿐만 아니라 정상 이미지에 대해서도 시각화 이미지를 생성",
    )

    args = parser.parse_args(argv)

    if not os.path.isdir(args.sample_directory_path):
        print(f"[오류] 디렉토리를 찾을 수 없습니다: {args.sample_directory_path}", file=sys.stderr)
        return 1

    image_paths = list_directory_images(args.sample_directory_path)
    if not image_paths:
        print(f"[오류] 디렉토리 내에 유효한 이미지 파일이 없습니다: {args.sample_directory_path}", file=sys.stderr)
        return 1

    print("=" * 88)
    print(f"[전체 순차 검정 시작] 대상 디렉토리: {args.sample_directory_path} (총 {len(image_paths)}장)")
    print("=" * 88)

    pipeline = MetrologyInspectionPipeline(
        auto_group_recipes=not args.single_cluster,
        save_visuals_for_normal=args.save_normal_visuals,
    )
    results = pipeline.inspect_batch(image_paths=image_paths, output_dir=args.output_dir)

    normals = sum(1 for r in results if not r.is_outlier)
    outliers = sum(1 for r in results if r.is_outlier)

    for idx, r in enumerate(results, 1):
        status_tag = "오계측(OUTLIER)" if r.is_outlier else "정상  (NORMAL )"
        print(f"[{idx:02d}/{len(results):02d}] {status_tag} | {r.image_name} (군집: {r.cluster_id}, 영역: {len(r.sites)}개)")
        print(f"       -> 초점 상태: {r.focus_result.explanation}")
        if r.is_outlier:
            print(f"       -> 주된 원인: {r.primary_defect} (복합: {', '.join(r.defect_types)})")
            for d in r.defects:
                print(f"          * [{d.defect_type.value}] {d.description}")
            print(f"       -> 진단 이미지: {r.diagnostic_image_path}")
            print(f"       -> 1:1 대조표 : {r.comparison_image_path}")
        else:
            print(f"       -> 판정 요약: 군집 정상 기준({r.reference_image_name}) 대비 모든 계측 요소 정상")
        print("-" * 88)

    print("=" * 88)
    print(f"[전수 검사 완료] 총 {len(results)}장  |  정상(NORMAL): {normals}장  |  오계측(OUTLIER): {outliers}장")
    print(f"- 상세 보고서 (MD)  : {os.path.join(args.output_dir, 'INSPECTION_REPORT.md')}")
    print(f"- 상세 보고서 (JSON): {os.path.join(args.output_dir, 'inspection_report.json')}")
    print(f"- 요약 보고서 (CSV) : {os.path.join(args.output_dir, 'inspection_report.csv')}")
    print(f"- 시각적 진단 폴더  : {os.path.join(args.output_dir, 'diagnostics')}")
    print(f"- 1:1 대조 자료 폴더: {os.path.join(args.output_dir, 'comparisons')}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
