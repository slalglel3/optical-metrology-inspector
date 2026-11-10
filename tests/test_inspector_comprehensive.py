"""Comprehensive Unit & Integration Test Suite for Optical Metrology Outlier Inspector."""

from __future__ import annotations
import os
import tempfile
import unittest
import cv2
import numpy as np

from metrology_inspector.models import DefectType
from metrology_inspector.pipeline import MetrologyInspectionPipeline
from metrology_inspector.sanitizer import VisualNoiseSanitizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")


class TestOpticalMetrologyInspector(unittest.TestCase):
    """요구사항 정의서(2.1~2.7 및 3) 및 8종 실제 샘플 기반 종합 검증 테스트."""

    def setUp(self) -> None:
        self.sample_files = [
            os.path.join(SAMPLES_DIR, "56d8f7f6-cb4d-4034-bbf0-d8e81d7c71b7.jpg"),
            os.path.join(SAMPLES_DIR, "e16b9a33-8bcf-4ab6-9810-78a6b849f140.jpg"),
            os.path.join(SAMPLES_DIR, "4e4ab8e4-3be3-4e9a-89c6-bc94c9312dde.jpg"),
            os.path.join(SAMPLES_DIR, "b31efdf0-31d2-4c32-8e67-504b52f7f78d.jpg"),
            os.path.join(SAMPLES_DIR, "558c07e3-cb30-42e0-9e4e-6305729118f8.jpg"),
            os.path.join(SAMPLES_DIR, "99cf0dfb-f221-4aaa-9b95-9cc06d774a92.jpg"),
            os.path.join(SAMPLES_DIR, "7e323b20-c51a-47d5-a256-519fe63bf36d.jpg"),
            os.path.join(SAMPLES_DIR, "3bd01813-20fe-44fb-a16f-2580cc1d52e8.jpg"),
        ]

    def test_01_real_samples_batch_inspection_and_outputs(self) -> None:
        """제공된 8장의 실제 샘플 이미지(직선 Box-in-Box, 격자 CD, 다중 위치 Hole) 일괄 검정 및 산출물 검증."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MetrologyInspectionPipeline(auto_group_recipes=True)
            results = pipeline.inspect_batch(self.sample_files, output_dir=tmpdir)

            self.assertEqual(len(results), 8)
            res_map = {r.image_name: r for r in results}

            # 1. 그룹 1 (Box-in-Box 1): 56d8f7f6(정상) vs e16b9a33(오계측: 내측 좌측 변 왜곡/파편화)
            self.assertFalse(res_map["56d8f7f6-cb4d-4034-bbf0-d8e81d7c71b7.jpg"].is_outlier)
            r_g1_anom = res_map["e16b9a33-8bcf-4ab6-9810-78a6b849f140.jpg"]
            self.assertTrue(r_g1_anom.is_outlier)
            self.assertIn(DefectType.PATTERN_DISTORTION.value, r_g1_anom.defect_types)

            # 2. 그룹 2 (Box-in-Box 2): 4e4ab8e4(정상) vs b31efdf0(오계측: 우측 외곽 점 찍힘 및 위치 이탈)
            self.assertFalse(res_map["4e4ab8e4-3be3-4e9a-89c6-bc94c9312dde.jpg"].is_outlier)
            r_g2_anom = res_map["b31efdf0-31d2-4c32-8e67-504b52f7f78d.jpg"]
            self.assertTrue(r_g2_anom.is_outlier)
            self.assertIn(DefectType.PATTERN_SHRINKAGE_DOT.value, r_g2_anom.defect_types)
            self.assertIn(DefectType.POSITION_MISMATCH.value, r_g2_anom.defect_types)

            # 3. 그룹 3 (Grid CD): 558c07e3(정상) vs 99cf0dfb(오계측: 하단 패드 검은 허공 위치 불일치)
            self.assertFalse(res_map["558c07e3-cb30-42e0-9e4e-6305729118f8.jpg"].is_outlier)
            r_g3_anom = res_map["99cf0dfb-f221-4aaa-9b95-9cc06d774a92.jpg"]
            self.assertTrue(r_g3_anom.is_outlier)
            self.assertEqual(r_g3_anom.primary_defect, DefectType.POSITION_MISMATCH.value)

            # 4. 그룹 4 (Multi-Site Hole): 7e323b20(정상, 2 Sites) vs 3bd01813(오계측, 2 Sites 선분/호 단절 및 결손)
            r_g4_norm = res_map["7e323b20-c51a-47d5-a256-519fe63bf36d.jpg"]
            r_g4_anom = res_map["3bd01813-20fe-44fb-a16f-2580cc1d52e8.jpg"]
            self.assertFalse(r_g4_norm.is_outlier)
            self.assertEqual(len(r_g4_norm.sites), 2)
            self.assertTrue(r_g4_anom.is_outlier)
            self.assertEqual(len(r_g4_anom.sites), 2)
            self.assertEqual(r_g4_anom.primary_defect, DefectType.LINE_TRUNCATION.value)

            # 5. 필수 산출물(보고서 3종 + 진단 이미지 + 1:1 대조 이미지) 생성 확인
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "INSPECTION_REPORT.md")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "inspection_report.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "inspection_report.csv")))
            for anom_res in [r_g1_anom, r_g2_anom, r_g3_anom, r_g4_anom]:
                self.assertTrue(anom_res.diagnostic_image_path and os.path.isfile(anom_res.diagnostic_image_path))
                self.assertTrue(anom_res.comparison_image_path and os.path.isfile(anom_res.comparison_image_path))

    def test_02_center_crosshair_removal_and_center_pattern_preservation(self) -> None:
        """요구사항 2.2: 정중앙 광학 십자가(+)는 제거하되, 실제 정중앙에 위치한 정상 계측 패턴은 보존되는지 검증."""
        img = np.full((400, 400, 3), 240, dtype=np.uint8)
        # 하늘색 마커 박스
        cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 0), 2)
        # 정중앙(x=200, y=130~270)을 지나는 두께 9px의 실제 정상 수직 계측 패턴 선분
        cv2.rectangle(img, (196, 130), (204, 270), (0, 0, 230), -1)
        # 우측(x=280, y=200)에 찍힌 1px 두께의 비계측 십자가(+)
        cv2.line(img, (270, 200), (290, 200), (0, 0, 230), 2)
        cv2.line(img, (280, 190), (280, 210), (0, 0, 230), 2)

        layers = VisualNoiseSanitizer.sanitize(img)
        # 정중앙의 실제 두께감 있는 계측 선분은 보존되어야 함
        center_pattern_pixels = int(layers.clean_red_mask[130:270, 196:205].sum())
        self.assertGreater(center_pattern_pixels, 1000)
        # 비계측 십자가(+)는 제거되어야 함
        crosshair_remained = int(layers.clean_red_mask[190:211, 270:291].sum())
        self.assertEqual(crosshair_remained, 0)

    def test_03_defocus_independent_and_unmeasurable_rescue(self) -> None:
        """요구사항 2.4 & 2.5: 광학 초점 불량 독립 검정 및 패턴 완전 소실 이미지의 군집 기준 영역 구제(Rescue) 검증."""
        base_img = cv2.imread(self.sample_files[0])
        self.assertIsNotNone(base_img)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm1_path = os.path.join(tmpdir, "norm1.jpg")
            norm2_path = os.path.join(tmpdir, "norm2.jpg")
            defocus_path = os.path.join(tmpdir, "defocus_with_pattern.jpg")
            lost_path = os.path.join(tmpdir, "lost_pattern_rescued.jpg")

            cv2.imwrite(norm1_path, base_img)
            cv2.imwrite(norm2_path, base_img)

            # 1) 패턴 마커/계측선은 그대로이나 기저 광학 이미지가 심하게 흐려진 이미지 생성 (독립 초점 검정 확인)
            layers = VisualNoiseSanitizer.sanitize(base_img)
            blurred_wafer = cv2.GaussianBlur(layers.gray_wafer, (51, 51), 18.0)
            defocus_img = cv2.cvtColor(blurred_wafer, cv2.COLOR_GRAY2BGR)
            # 원래의 하늘색 마커와 빨간색 계측선을 그대로 다시 입힘 (패턴 검출은 성공하나 광학 초점은 불량인 상황)
            overlay_mask = layers.cyan_marker_mask | layers.clean_red_mask
            defocus_img[overlay_mask] = base_img[overlay_mask]
            cv2.imwrite(defocus_path, defocus_img)

            # 2) 초점 상실 및 패턴/마커 완전 소실 이미지 생성 (자체 영역 특정 불가 -> 군집 기준으로 구제되어야 함)
            lost_img = cv2.cvtColor(blurred_wafer, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(lost_path, lost_img)

            pipeline = MetrologyInspectionPipeline(auto_group_recipes=False)
            results = pipeline.inspect_batch(
                [norm1_path, norm2_path, defocus_path, lost_path],
                output_dir=os.path.join(tmpdir, "out"),
            )
            rmap = {r.image_name: r for r in results}

            # 패턴 검출이 정상이어도 광학 초점 불량이 독립 판정되어야 함
            self.assertTrue(rmap["defocus_with_pattern.jpg"].focus_result.is_defocused)
            self.assertTrue(rmap["defocus_with_pattern.jpg"].is_outlier)
            self.assertIn(DefectType.OPTICAL_DEFOCUS.value, rmap["defocus_with_pattern.jpg"].defect_types)

            # 패턴 완전 소실 이미지가 누락되지 않고 군집 기준 ROI로 구제(is_rescued=True)되어야 함
            r_lost = rmap["lost_pattern_rescued.jpg"]
            self.assertTrue(r_lost.is_outlier)
            self.assertEqual(len(r_lost.sites), 1)
            self.assertTrue(r_lost.sites[0].is_rescued)
            self.assertIn(DefectType.UNMEASURABLE_RESCUED.value, r_lost.defect_types)

    def test_04_line_bloating_and_adjacent_intrusion_defects(self) -> None:
        """요구사항 2.6: 선폭 팽창 및 뭉침(LINE_BLOATING)과 인접 영역 침범(ADJACENT_INTRUSION) 분류 검증."""
        base_img = cv2.imread(self.sample_files[2])  # G2_norm
        self.assertIsNotNone(base_img)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm1_path = os.path.join(tmpdir, "norm1.jpg")
            norm2_path = os.path.join(tmpdir, "norm2.jpg")
            bloat_path = os.path.join(tmpdir, "bloated.jpg")
            intrude_path = os.path.join(tmpdir, "intruded.jpg")

            cv2.imwrite(norm1_path, base_img)
            cv2.imwrite(norm2_path, base_img)

            # 1) 선폭 팽창 및 뭉침: 상단 외곽 계측선(x=452~509, y=341~352)이 두께 24px로 비정상 팽창
            bloat_img = base_img.copy()
            cv2.rectangle(bloat_img, (452, 334), (509, 358), (0, 0, 235), -1)
            cv2.imwrite(bloat_path, bloat_img)

            # 2) 인접 영역 침범: 좌측 외곽 수직 계측선(x=344~355, y=448~499)이 하단에서 인접 수평 방향으로 꺾여 침범
            intrude_img = base_img.copy()
            cv2.rectangle(intrude_img, (344, 492), (382, 501), (0, 0, 235), -1)
            cv2.imwrite(intrude_path, intrude_img)

            pipeline = MetrologyInspectionPipeline(auto_group_recipes=False)
            results = pipeline.inspect_batch([norm1_path, norm2_path, bloat_path, intrude_path])
            rmap = {r.image_name: r for r in results}

            self.assertTrue(rmap["bloated.jpg"].is_outlier)
            self.assertIn(DefectType.LINE_BLOATING.value, rmap["bloated.jpg"].defect_types)

            self.assertTrue(rmap["intruded.jpg"].is_outlier)
            self.assertIn(DefectType.ADJACENT_INTRUSION.value, rmap["intruded.jpg"].defect_types)


if __name__ == "__main__":
    unittest.main()
