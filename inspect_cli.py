#!/usr/bin/env python3
"""CLI entrypoint for the Semiconductor Optical Metrology Outlier Detection System."""

from __future__ import annotations
import argparse
import glob
import os
import sys
from typing import List

from metrology_inspector.pipeline import MetrologyInspectionPipeline


def collect_image_paths(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for item in inputs:
        if os.path.isdir(item):
            for entry in sorted(os.listdir(item)):
                full = os.path.join(item, entry)
                if os.path.isfile(full) and entry.lower().endswith(exts):
                    paths.append(full)
        elif os.path.isfile(item):
            paths.append(item)
        else:
            for matched in sorted(glob.glob(item)):
                if os.path.isfile(matched) and matched.lower().endswith(exts):
                    paths.append(matched)
    return paths


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="반도체 계측 이미지 이상치(오계측) 자동 검출 시스템 CLI"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="검사할 이미지 파일 경로 또는 디렉토리 목록",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output_report",
        help="진단 이미지, 대조 분석 시트 및 상세 보고서를 저장할 디렉토리 (기본값: output_report)",
    )
    parser.add_argument(
        "--single-cluster",
        action="store_true",
        help="자동 레시피 군집 분리를 끄고 입력 전체를 단일 군집으로 강제 비교",
    )
    parser.add_argument(
        "--save-normal-visuals",
        action="store_true",
        help="오계측뿐만 아니라 정상 판정 이미지에 대해서도 진단 시각화 이미지를 생성",
    )

    args = parser.parse_args(argv)
    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        print("[오류] 유효한 이미지 파일을 찾을 수 없습니다.", file=sys.stderr)
        return 1

    pipeline = MetrologyInspectionPipeline(
        auto_group_recipes=not args.single_cluster,
        save_visuals_for_normal=args.save_normal_visuals,
    )
    results = pipeline.inspect_batch(image_paths=image_paths, output_dir=args.output_dir)

    normals = sum(1 for r in results if not r.is_outlier)
    outliers = sum(1 for r in results if r.is_outlier)

    print("=" * 88)
    print(f"[검사 완료] 총 {len(results)}장 처리  |  정상(NORMAL): {normals}장  |  오계측(OUTLIER): {outliers}장")
    print("=" * 88)
    for idx, r in enumerate(results, 1):
        flag = "OUTLIER" if r.is_outlier else "NORMAL "
        print(f"{idx:02d}. [{flag}] [{r.cluster_id}] {r.image_name} (Sites: {len(r.sites)})")
        print(f"    -> {r.summary_explanation}")
    print("=" * 88)
    print(f"- Markdown 보고서: {os.path.join(args.output_dir, 'INSPECTION_REPORT.md')}")
    print(f"- JSON 보고서    : {os.path.join(args.output_dir, 'inspection_report.json')}")
    print(f"- CSV 보고서     : {os.path.join(args.output_dir, 'inspection_report.csv')}")
    print(f"- 시각적 진단    : {os.path.join(args.output_dir, 'diagnostics')}")
    print(f"- 1:1 대조 자료  : {os.path.join(args.output_dir, 'comparisons')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
