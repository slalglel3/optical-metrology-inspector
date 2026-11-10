# 반도체 계측 이미지 이상치(오계측) 자동 검출 시스템 (`optical-metrology-inspector`)

정답 라벨(Ground Truth)이 사전에 주어지지 않는 환경에서, 동일 조건의 계측 이미지 군집을 상호 대조하여 **비계측 노이즈(화면 내 텍스트, 장비 광학 정중앙 십자가 등)를 배제**하고 **광학 초점 불량(Defocus) 및 7대 오계측 결함 유형을 자동으로 검출·분류·시각화**하는 범용 광학 계측 검사 시스템입니다.

---

## 1. 핵심 아키텍처 및 구현 기능

| 요구사항 항목 | 담당 모듈 | 핵심 구현 내용 |
| :--- | :--- | :--- |
| **2.1 대용량 처리 & 형상 범용성** | [`pipeline.py`](metrology_inspector/pipeline.py)<br/>[`feature_extractor.py`](metrology_inspector/feature_extractor.py) | • 2-Pass 스트리밍 구조로 원본 이미지 메모리 즉시 해제 (OOM 방지)<br/>• 선, 원, 타원, 사각형, 다각형에 범용 대응하는 골격·거리변환·12방위 커버리지·궤적 피처 추출 |
| **2.2 비계측 시각 요소 배제** | [`sanitizer.py`](metrology_inspector/sanitizer.py) | • 좌상단 빨간색 좌표 텍스트, 계측 영역 중첩 마젠타 텍스트(`OL1X`, `CD4`, `DA2`), 외곽 아웃라인 라벨 완전 배제<br/>• 마젠타 텍스트 중첩(Overdraw)으로 끊어진 계측 선분의 방향성 브릿지 복원<br/>• 화면 정중앙 및 로컬 타깃 중심 십자가(`+`) 형태학적 정밀 배제 (**정중앙 실제 정상 계측 패턴 100% 보존**) |
| **2.3 계측 영역 국소화 & 분리** | [`roi_localizer.py`](metrology_inspector/roi_localizer.py) | • `Union(하늘색 마커, 빨간색 계측요소)` 최소 외접 영역 추출 및 잘림 방지 안전 패딩<br/>• 적응형 공간 클러스터링으로 **원거리 복수 계측 위치(Multi-Site) 자동 분리** 및 밀집 대형 패턴 단일 유지 |
| **2.4 군집 정규화 & 결측 구제** | [`roi_localizer.py`](metrology_inspector/roi_localizer.py) | • 군집 기준 영역(`Consensus ROI`)과 개별 검출 영역의 `Union` 정규화로 잘림 원천 차단<br/>• 초점 상실/패턴 소실로 영역 특정이 불가능한 이미지를 **군집 기준 영역으로 강제 구제(Rescue)** |
| **2.5 광학 초점 불량 독립 검정** | [`focus_analyzer.py`](metrology_inspector/focus_analyzer.py) | • 패턴 검출 성공 여부와 무관하게 순수 웨이퍼 광학 영역의 상위 엣지 대역 Tenengrad + Laplacian 선명도를 **독립·병렬 검정** |
| **2.6 & 2.7 7대 결함 분류 & 강건 판정** | [`defect_classifier.py`](metrology_inspector/defect_classifier.py) | • 내재적 무결성 점수 및 강건 통계(Median/MAD/75th Percentile Anchoring) 기반 정상 기준(`Golden Reference`) 수립<br/>• **7대 결함 분류**: `계측 위치 불일치`, `선분 단절 및 결손`, `선폭 팽창 및 뭉침`, `내부 파편화`, `인접 영역 침범`, `패턴 왜곡`, `패턴 축소 및 점 찍힘` |
| **3. 최종 산출물 생성** | [`visualizer.py`](metrology_inspector/visualizer.py)<br/>[`pipeline.py`](metrology_inspector/pipeline.py) | • 상세 판정 보고서 (`INSPECTION_REPORT.md`, `inspection_report.json`, `inspection_report.csv`)<br/>• 이상 발생 부위 핀포인트 하이라이트 진단 이미지 (`diagnostics/`)<br/>• 정상 기준 이미지 vs 오계측 이미지 1:1 Side-by-Side 대조 시트 (`comparisons/`) |

---

## 2. 실행 방법

### 2.1 CLI 일괄 검정 실행
```bash
python3 inspect_cli.py samples/ -o output_report
```

### 2.2 단위 및 통합 테스트 실행
```bash
python3 -m unittest discover -s tests -v
```
