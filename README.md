# 반도체 계측 이미지 이상치(오계측) 자동 검출 시스템 (`optical-metrology-inspector`)

정답 라벨(Ground Truth)이 사전에 주어지지 않는 환경에서, 동일 조건의 계측 이미지 군집을 상호 대조하여 **비계측 노이즈(화면 내 텍스트, 장비 광학 정중앙 십자가 등)를 배제**하고 **광학 초점 불량(Defocus) 및 7대 오계측 결함 유형을 자동으로 검출·분류·시각화**하는 범용 광학 계측 검사 시스템입니다.

---

## 1. 실행 환경 및 외부 라이브러리 요구사항 (Requirements)

본 프로젝트는 무거운 딥러닝 프레임워크(PyTorch, TensorFlow 등) 없이 저사양 PC 및 서버 환경에서도 대용량 고해상도 이미지를 고속·저메모리로 처리할 수 있도록 **3가지 경량 핵심 외부 라이브러리**만 사용합니다.

### 1.1 필수 요구 사항
* **Python 버전**: `Python 3.9` 이상 (`3.10 ~ 3.14` 권장)
* **필수 외부 라이브러리** ([`requirements.txt`](requirements.txt)):

| 패키지명 | 최소 권장 버전 | 프로젝트 내 주요 역할 |
| :--- | :---: | :--- |
| **`numpy`** | `>= 1.24.0` | 고속 배열 연산, 강건 통계(Median/MAD/백분위), 선분 주축 회귀 및 휨(Sinuosity) 계산 |
| **`opencv-python`** | `>= 4.8.0` | 색공간 분리, Connected Components, 형태학적 필터링, 거리 변환(선폭 측정), Sobel/Laplacian 초점 측정 |
| **`Pillow`** | `>= 10.0.0` | 시각적 결함 진단 이미지 및 1:1 정상 대조 시트의 헤더/배지 텍스트(한글·영문 폰트) 렌더링 |

> **참고 (Headless 서버 환경)**: GUI 디스플레이가 없는 리눅스 컨테이너/서버 환경에서는 `opencv-python` 대신 `opencv-python-headless>=4.8.0`을 설치해도 100% 동일하게 동작합니다.

---

## 2. 설치 가이드 (Installation Guide)

### 2.1 저장소 클론 및 디렉토리 이동
```bash
git clone https://github.com/slalglel3/optical-metrology-inspector.git
cd optical-metrology-inspector
```

### 2.2 가상환경(Virtual Environment) 생성 및 활성화 (권장)
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS / WSL2
# Windows PowerShell의 경우: .venv\Scripts\Activate.ps1
```

### 2.3 필수 외부 라이브러리 일괄 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.4 (선택) 진단 이미지 내 한글 폰트 표시 안내
* 시각적 진단 및 대조 이미지 생성 시 시스템 내 한글 폰트(Windows/WSL2 `맑은 고딕`, Linux `NanumGothic` 또는 `NotoSansCJK`)를 자동 탐색합니다.
* 순수 Ubuntu/Debian 리눅스 환경에서 진단 이미지 내부 라벨을 한글로 렌더링하려면 나눔고딕 폰트를 설치할 수 있습니다(미설치 시에도 깨짐 없이 영문 라벨로 자동 폴백됩니다):
  ```bash
  sudo apt-get update && sudo apt-get install -y fonts-nanum
  ```

---

## 3. 실행 및 테스트 가이드 (Usage)

사용 목적에 따라 **특정 타겟 이미지 1장만 검사하는 스크립트(`inspect_target.py`)**와 **폴더 내 전체 이미지를 순차 전수 검사하는 스크립트(`inspect_all.py`)**를 사용할 수 있습니다.

### 3.1 특정 타겟 이미지 1장에 대해서만 검사 (`inspect_target.py`)
```bash
# 사용법: python3 inspect_target.py <sample_directory_path> <타겟이미지파일명> [-o 출력폴더]
python3 inspect_target.py samples e16b9a33-8bcf-4ab6-9810-78a6b849f140.jpg
```

### 3.2 샘플 디렉토리의 모든 이미지에 대해 순차적으로 전수 검사 (`inspect_all.py`)
```bash
# 사용법: python3 inspect_all.py <sample_directory_path> [-o 출력폴더]
python3 inspect_all.py samples
```

### 3.3 통합 CLI 실행 (`inspect_cli.py`)
검사할 이미지 파일들이 들어있는 폴더(또는 개별 이미지 경로들)를 자유롭게 조합하여 실행할 수도 있습니다:
```bash
# 샘플 이미지 폴더 일괄 검정 및 output_report 디렉토리에 결과 생성
python3 inspect_cli.py samples/ -o output_report

# 개별 이미지 파일들을 직접 지정하여 실행
python3 inspect_cli.py samples/56d8f7f6-cb4d-4034-bbf0-d8e81d7c71b7.jpg samples/e16b9a33-8bcf-4ab6-9810-78a6b849f140.jpg -o output_report
```

#### 주요 CLI 옵션
* `-o, --output-dir <경로>`: 상세 보고서(MD/JSON/CSV), 진단 이미지(`diagnostics/`), 1:1 대조 자료(`comparisons/`)를 저장할 출력 폴더 지정 (기본값: `output_report`)
* `--single-cluster`: 자동 레시피 조건 군집 분리를 비활성화하고 입력된 모든 이미지를 단일 군집으로 강제 비교
* `--save-normal-visuals`: 오계측(OUTLIER) 이미지뿐만 아니라 정상(NORMAL) 이미지에 대해서도 진단 시각화 이미지를 함께 저장

### 3.2 단위 및 통합 테스트 실행
```bash
python3 -m unittest discover -s tests -v
```

---

## 4. 핵심 아키텍처 및 구현 기능 매핑

| 요구사항 항목 | 담당 모듈 | 핵심 구현 내용 |
| :--- | :--- | :--- |
| **2.1 대용량 처리 & 형상 범용성** | [`pipeline.py`](metrology_inspector/pipeline.py)<br/>[`feature_extractor.py`](metrology_inspector/feature_extractor.py) | • 2-Pass 스트리밍 구조로 원본 이미지 메모리 즉시 해제 (OOM 방지)<br/>• 선, 원, 타원, 사각형, 다각형에 범용 대응하는 골격·거리변환·12방위 커버리지·궤적 피처 추출 |
| **2.2 비계측 시각 요소 배제** | [`sanitizer.py`](metrology_inspector/sanitizer.py) | • 좌상단 빨간색 좌표 텍스트, 계측 영역 중첩 마젠타 텍스트(`OL1X`, `CD4`, `DA2`), 외곽 아웃라인 라벨 완전 배제<br/>• 마젠타 텍스트 중첩(Overdraw)으로 끊어진 계측 선분의 방향성 브릿지 복원<br/>• 화면 정중앙 및 로컬 타깃 중심 십자가(`+`) 형태학적 정밀 배제 (**정중앙 실제 정상 계측 패턴 100% 보존**) |
| **2.3 계측 영역 국소화 & 분리** | [`roi_localizer.py`](metrology_inspector/roi_localizer.py) | • `Union(하늘색 마커, 빨간색 계측요소)` 최소 외접 영역 추출 및 잘림 방지 안전 패딩<br/>• 적응형 공간 클러스터링으로 **원거리 복수 계측 위치(Multi-Site) 자동 분리** 및 밀집 대형 패턴 단일 유지 |
| **2.4 군집 정규화 & 결측 구제** | [`roi_localizer.py`](metrology_inspector/roi_localizer.py) | • 군집 기준 영역(`Consensus ROI`)과 개별 검출 영역의 `Union` 정규화로 잘림 원천 차단<br/>• 초점 상실/패턴 소실로 영역 특정이 불가능한 이미지를 **군집 기준 영역으로 강제 구제(Rescue)** |
| **2.5 광학 초점 불량 독립 검정** | [`focus_analyzer.py`](metrology_inspector/focus_analyzer.py) | • 패턴 검출 성공 여부와 무관하게 순수 웨이퍼 광학 영역의 상위 엣지 대역 Tenengrad + Laplacian 선명도를 **독립·병렬 검정** |
| **2.6 & 2.7 7대 결함 분류 & 강건 판정** | [`defect_classifier.py`](metrology_inspector/defect_classifier.py) | • 내재적 무결성 점수 및 강건 통계(Median/MAD/75th Percentile Anchoring) 기반 정상 기준(`Golden Reference`) 수립<br/>• **7대 결함 분류**: `계측 위치 불일치`, `선분 단절 및 결손`, `선폭 팽창 및 뭉침`, `내부 파편화`, `인접 영역 침범`, `패턴 왜곡`, `패턴 축소 및 점 찍힘` |
| **3. 최종 산출물 생성** | [`visualizer.py`](metrology_inspector/visualizer.py)<br/>[`pipeline.py`](metrology_inspector/pipeline.py) | • 상세 판정 보고서 (`INSPECTION_REPORT.md`, `inspection_report.json`, `inspection_report.csv`)<br/>• 이상 발생 부위 핀포인트 하이라이트 진단 이미지 (`diagnostics/`)<br/>• 정상 기준 이미지 vs 오계측 이미지 1:1 Side-by-Side 대조 시트 (`comparisons/`) |
