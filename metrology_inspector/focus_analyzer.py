"""Independent Optical Focus (Defocus) Analyzer (요구사항 2.5: 광학 초점 불량 검출 및 독립적 병렬 검정).

패턴/마커 검출 성공 여부와 무관하게, 비계측 오버레이(텍스트, 십자가, 마커, 계측선)가 배제된
순수 광학 웨이퍼 영역만을 대상으로 선명도(Tenengrad, Laplacian, Edge Acutance)를 산출하고
군집 비교 및 절대 기준을 결합하여 초점 불량(Defocus) 여부를 독립 판정합니다.
"""

from __future__ import annotations
from typing import List
import cv2
import numpy as np
from .models import FocusResult


class IndependentFocusEvaluator:
    """패턴 검출 파이프라인과 분리되어 독립·병렬 동작하는 광학 초점 검정기."""

    # 절대 초점 불량 하한 임계치 (군집 전체가 흐린 경우까지 방어)
    ABSOLUTE_MIN_SHARPNESS = 14.0
    # 군집 중앙값 대비 상대적 선명도 저하 임계 비율
    RELATIVE_DEFOCUS_RATIO = 0.56
    # 강건 Z-score 임계치
    ROBUST_Z_THRESHOLD = -3.5

    @classmethod
    def evaluate_single(
        cls,
        gray_wafer: np.ndarray,
        wafer_valid_mask: np.ndarray,
    ) -> FocusResult:
        """단일 이미지의 순수 웨이퍼 광학 영역으로부터 초점 선명도 지표를 산출합니다."""
        # Sobel 1차 미분 (Tenengrad) 및 Laplacian 2차 미분 계산
        gx = cv2.Sobel(gray_wafer, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_wafer, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        lap_abs = np.abs(cv2.Laplacian(gray_wafer, cv2.CV_32F, ksize=3))

        valid_grads = grad_mag[wafer_valid_mask]
        valid_laps = lap_abs[wafer_valid_mask]

        if valid_grads.size < 100:
            valid_grads = grad_mag.ravel()
            valid_laps = lap_abs.ravel()

        # 배경 여백 비율에 영향을 받지 않도록 상위 엣지 대역(90~99.5 백분위)에서 선명도 측정
        p90_grad = float(np.percentile(valid_grads, 90.0))
        p98_grad = float(np.percentile(valid_grads, 98.0))
        p995_grad = float(np.percentile(valid_grads, 99.5))

        p98_lap = float(np.percentile(valid_laps, 98.0))
        p995_lap = float(np.percentile(valid_laps, 99.5))

        # Tenengrad 대표 점수 및 Laplacian 대표 점수
        tenengrad_score = 0.5 * p98_grad + 0.3 * p995_grad + 0.2 * p90_grad
        laplacian_score = 0.6 * p98_lap + 0.4 * p995_lap

        # Edge Acutance 결합 종합 선명도 점수
        sharpness_score = 0.65 * tenengrad_score + 0.35 * laplacian_score

        is_abs_defocus = sharpness_score < cls.ABSOLUTE_MIN_SHARPNESS
        explanation = (
            f"광학 초점 불량 (절대 선명도 {sharpness_score:.1f} < 기준 {cls.ABSOLUTE_MIN_SHARPNESS:.1f})"
            if is_abs_defocus
            else f"정상 초점 (선명도 점수 {sharpness_score:.1f})"
        )

        return FocusResult(
            sharpness_score=round(sharpness_score, 3),
            tenengrad_score=round(tenengrad_score, 3),
            laplacian_score=round(laplacian_score, 3),
            relative_to_cluster=1.0,
            robust_z_score=0.0,
            is_defocused=is_abs_defocus,
            explanation=explanation,
        )

    @classmethod
    def calibrate_cluster_focus(cls, focus_results: List[FocusResult]) -> None:
        """동일 군집 내 이미지들의 초점 점수를 강건 통계(Median/MAD)로 상호 대조하여 Defocus 여부를 확정합니다."""
        if not focus_results:
            return

        scores = np.array([fr.sharpness_score for fr in focus_results], dtype=np.float64)
        # 군집 내에 극단적인 초점 불량 이미지가 다수 포함되어 있어도 정상 기준이 왜곡되지 않도록
        # 상위 선명도 그룹(75th percentile 이상)의 중앙값을 정상 초점 기준(Baseline)으로 수립 (요구사항 2.7)
        p75 = float(np.percentile(scores, 75.0))
        top_group = scores[scores >= p75 * 0.85]
        baseline_score = float(np.median(top_group)) if top_group.size > 0 else float(np.max(scores))

        top_mad = float(np.median(np.abs(top_group - baseline_score))) if top_group.size > 1 else 0.0
        robust_sigma = max(top_mad / 0.6745, 0.08 * max(1.0, baseline_score))

        for fr in focus_results:
            rel_ratio = fr.sharpness_score / max(1e-6, baseline_score)
            z_score = (fr.sharpness_score - baseline_score) / max(1e-6, robust_sigma)
            fr.relative_to_cluster = round(rel_ratio, 3)
            fr.robust_z_score = round(z_score, 2)

            if (
                fr.sharpness_score < cls.ABSOLUTE_MIN_SHARPNESS
                or (rel_ratio < cls.RELATIVE_DEFOCUS_RATIO and z_score < cls.ROBUST_Z_THRESHOLD)
            ):
                fr.is_defocused = True
                fr.explanation = (
                    f"광학 초점 불량(Defocus) 검출: 선명도 {fr.sharpness_score:.1f} "
                    f"(군집 정상 기준 {baseline_score:.1f} 대비 {rel_ratio * 100:.1f}% 수준)"
                )
            else:
                fr.is_defocused = False
                fr.explanation = (
                    f"정상 초점: 선명도 {fr.sharpness_score:.1f} "
                    f"(군집 기준 대비 {rel_ratio * 100:.1f}%)"
                )
