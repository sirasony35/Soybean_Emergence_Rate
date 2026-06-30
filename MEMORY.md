# Soybean_Emergence_Rate — 프로젝트 학습 메모리

새만금 간척지 논콩의 **입모율(emergence rate) / 결주율** 분석 프로젝트. Claude Code 작업 중 학습/시도한 내용을 한 곳에 정리. **집/회사 모두에서 이 폴더만 동기화하면 컨텍스트 이어갈 수 있음**.

---

## 1. 프로젝트 개요

- **목적**: 마스카·포리고 파종기의 두둑별 입모율·결주율 비교
- **데이터**: 13.8GB 드론 정사영상 + 5개 필지 SHP (`shapefile/GJSM-{1-1,1-2,1-3,2-2,2-3}_Boundary.zip`)
- **해상도**: **5.25mm/px (0.5cm GSD)** — 농진청 일반 드론 2cm GSD의 4배 고화질
- **결과 저장**: `result/fields/` (필지별 TIF), `result/emergence/` (분석 결과)

## 2. 농학 제원 (사용자 확정)

- **모든 두둑은 dual-row** (2줄 파종, 단일줄 두둑 없음)
- **한 두둑 안 두 줄 간격**: 보통 **30 cm**, 일부 두둑은 **70 cm** (이때 두둑 폭이 더 넓음)
- **주간(개체 간)**: **20 cm**
- **행간(두둑 사이)**: 가변, 측정상 약 35~50 cm
- **두둑 위에만 파종, 고랑은 비어있음**
- GJSM-1-1엔 가로 두둑 + 세로 두둑 **혼합 방향** 있음 (대부분 세로)

## 3. 처리 환경

### Python 환경
- `C:\Users\user\miniconda3\envs\satelite` — conda env
- 주요 패키지: rasterio, geopandas, scikit-image, PyTorch 2.5.1+cu121, segment-anything 1.0, opencv 4.13

### GPU
- **NVIDIA RTX 4090 Laptop, 17.2GB VRAM** — SAM 처리에 충분

### 모델 파일
- `models/sam_vit_l_0b3195.pth` (1.2GB, SAM vit_l 가중치)

### 캐시 파일 (다음 세션 가속용)
- `result/fields/_cache_GJSM-1-1_bw.npy`, `_valid.npy`, `_exg.npy` — 1차 이진화 + ExG (ExG 12분 절약)
- `result/fields/_cache_GJSM-1-2_bw.npy`, `_valid.npy` — 1차 이진화 (ExG 캐시 없음)
- `result/fields/_cache_GJSM-1-3_bw.npy`, `_valid.npy`
- 2-2, 2-3 캐시 없음 (필요 시 ExG 12분 추가)

## 4. 시도한 알고리즘 4가지

### 1차: 두둑 라인 1D 피크 검출
- ExG → Otsu+Sauvola → Radon → 회전 → 가로 누적 1D → find_peaks
- 결과: 입모율 100%+ (과대카운팅), 라인이 고랑에 그어짐

### 2차: tile-based + dominant 각도 SNAP + dual-row 클러스터링
- 15m × 15m 타일별 Radon → dominant 각도 검출 → 타일별 SNAP(±15°)
- 모든 줄 검출 → 인접 간격 < 75cm = 같은 두둑 (최대 2줄)
- GJSM-1-1: 64타일/2,837 라인/161,420 개체/**입모율 90.4%, 조간 30.7cm, 행간 51.8cm**
- 문제: 고랑에 일부 라인 + 두둑 방향 혼합 못 잡음
- 사용자 평가: "라인이 두둑 수직 방향" → 일부 잘못된 방향 검출

### 3차: satelite/wv3_detecting_gaps 방식 (결주 폴리곤 면적)
- ExG → Otsu → closing(disk, 행간 메움) → 결주 마스크 → 폴리곤화
- 5.25mm GSD 그대로는 closing이 너무 느리고 cap=10px로 불충분 → 10배 다운샘플(5cm GSD)
- 결과: ExG로 콩 vs 잡초/노이즈 분리 불가능
  - Otsu만: 식생 0.7% → 결주율 96% (비현실)
  - Otsu+Sauvola: 식생 32% → closing 후 두둑영역 99% (필지 전체)
  - **두둑/고랑 구분 ExG로 불가능 — 본질적 한계**

### 4차: SAM (Segment Anything Model) ← 현재 진행 중
- 농진청 2cm GSD vs 우리 5.25mm GSD = **콩 새싹이 우리 영상에서 4-6 픽셀 (분산)** → ExG 신호 약함이 원인
- SAM Automatic Mask Generator 사용
- 다운샘플 2배(1cm GSD)는 새싹이 1-2 픽셀로 작아져 안 됨 → **원본 5.25mm 해상도 사용**

#### SAM v1 (엄격 설정)
- `points_per_side=48, pred_iou_thresh=0.85, stability=0.85`
- 면적 필터 4-100 cm², HSV Hue 30-90
- 결과: 4685 마스크 → 콩잎 2165개 → 입모율 31.3%
- 한계: 일부 영역(농진청 동그라미 영역)에서 일부만 검출 (~20% 검출률)

#### SAM v3 (완화 설정, 현재 최선)
- `points_per_side=48, pred_iou_thresh=0.80, stability=0.80`
- 면적 필터 **1-200 cm²**, HSV Hue 20-100, S>20, V>30
- 결과: 5921 마스크 → 콩잎 4694개 → **입모율 67.8%**
- 처리 시간: SAM 6분 + 필터 13분 (필터링이 의외로 오래 — 5921 마스크 색조 평가)

#### 알려진 버그: torchvision 0.20.1 NMS 디바이스 호환
- `points_per_side > 48` 또는 마스크 > 4000 시 `_batched_nms_vanilla` 디바이스 mismatch 에러
- 우회: points_per_side=48 유지, 임계만 완화

#### 시각화 병목 (해결됨)
- 원래 5921 마스크 × 5714² 영상 색칠 + 4694 scatter 개별 호출 → 30분+ 무한 루프
- **해결**: `sam_roi_test.py`에 다운샘플 색칠 + 일괄 scatter + npz 저장 + 별도 `sam_visualize_only.py`

## 5. 본질적 학습 (다시 시도 시 함정 방지)

### 규칙 1: 5.25mm GSD에 satelite 위성/저해상도 드론 코드 직접 적용 금지
- 위성 30cm GSD: 두둑이 한 픽셀에 들어가 면적 기반 자연
- 우리 5.25mm: 콩 새싹이 4-6 픽셀로 흩어져 ExG 분산 → Otsu가 0.7%만 잡음

### 규칙 2: ExG만으로 콩잎 vs 잡초/노이즈 분리 불가능
- 새만금 간척지 입모기 단계는 콩과 잡초 모두 ExG 양수
- 사질토 노이즈도 약한 양수
- 색조 기반 단순 임계로 분리 안 됨 → **AI 모델 또는 두둑 구조 활용 필수**

### 규칙 3: 두둑/고랑 구분이 ExG 데이터에서 불가능
- 큰 closing으로 두둑 마스크 만들려 하면 행간 65cm를 메워 필지 전체가 두둑이 됨
- 작은 closing이면 두둑이 sparse한 점들로 남음
- **데이터 특성상 픽셀 단위 분리 안 됨**

### 규칙 4: 두둑 dual-row 구조 = 모든 두둑 2줄 (30cm 또는 70cm 간격)
- 라인 검출 시 dual-row 클러스터링 임계 `intra_ridge_max_cm = 75`

### 규칙 5: 두둑 방향 혼합 — tile-based 필수
- 필지에 가로 + 세로 두둑 섞일 수 있음
- 단일 Radon으로 전체 처리 안 됨

### 규칙 6: estimate_row_angle 버그
- skimage radon은 projection 각도, 라인의 직교 방향에서 variance max
- row_angle = angle_proj - 90 (mod 180 → [-90, 90))

## 6. 코드 파일 가이드

### 분석 스크립트
| 파일 | 역할 | 상태 |
|---|---|---|
| `scripts/01_crop_fields.py` | 정사영상 → 필지별 TIF | 안정 |
| `scripts/02_analyze_emergence.py` | 입모율 분석 (현재 4차 시도 잔재 — 결주 폴리곤 방식) | 보류 (재작성 필요) |
| `scripts/03_postprocess_visualize.py` | PPT 시각화 + 파종기 비교 | 2차 라인 기반 (재작성 필요) |
| `scripts/04_detail_overlay.py` | RGB 상세 오버레이 + 조간/주간 CSV | 2차 라인 기반 (재작성 필요) |
| `scripts/emergence_lib.py` | 공통 알고리즘 모듈 | 안정 |
| `scripts/run_all.py` | 01~04 순차 실행 | 안정 |

### SAM 스크립트 (현재 메인)
| 파일 | 역할 |
|---|---|
| `scripts/sam_roi_test.py` | SAM ROI 시험 (현재 v3 설정). npz + PNG 자동 저장 |
| `scripts/sam_visualize_only.py` | npz 읽어 PNG 빠르게 재생성 |

### 진단 스크립트
- `scripts/diag_row_detect.py`, `diag_ridge_detect.py`, `diag_visualize_rotation.py` — 검증용

## 7. 다음 세션 시작 가이드

### 빠르게 시작 (SAM v3 본 실행)
```powershell
cd C:\Users\user\Desktop\분석프로젝트\Soybean_Emergence_Rate
conda activate satelite
python -u scripts/sam_roi_test.py
```
- 약 20분 (SAM 6분 + 필터 13분 + 빠른 시각화)
- 산출: `result/sam_test/GJSM-1-1_sam_roi_test_ds1_v3.png` + `.npz`

### npz가 이미 있으면 시각화만
```powershell
python -u scripts/sam_visualize_only.py
```
- 1분 이내

### 5필지 본 처리 (검증 통과 시)
- 현재 ROI 30m×30m 처리에 약 20분 → 1.795 ha 필지 → 약 1ha당 6~8분 추정
- 5필지 처리 총 ~3시간 (각 필지 30분 평균)
- 본 처리 코드 작성 필요 (현재는 ROI 테스트만 있음)

## 8. 사용자 협업 패턴 (다음 세션 컨텍스트)

- 큰 알고리즘 변경은 사용자가 직접 검토 후 결정
- "시도 → 결과 보고 → 옵션 제시 → 사용자 결정" 사이클
- 결과 비현실적이면 즉시 보고 (가짜로 잘 됐다 하면 안 됨)
- 한국어로 보고. 표 형식 + 핵심만

## 9. 남은 옵션 (다음 세션 결정 사항)

| 옵션 | 시간 | 장단점 |
|---|---|---|
| **SAM v3 결과 시각화 확인** | 즉시 | npz 있으면 1분, PNG 보고 만족도 판단 |
| **5필지 본 처리** | ~3시간 | 현재 67.8% 입모율 수치로 만족 시 |
| **검출 더 개선** — SAM 2 시도 | 30분~ | 더 정확. SAM 2는 별도 설치 |
| **검출 더 개선** — 두둑 라인 prompt 기반 | 2~3시간 | ExG로 두둑 검출 → 그 위에서만 SAM. 정확도 ↑ |
| **torchvision downgrade** | 30분 | torchvision 0.19 로 NMS 버그 회피, points_per_side ↑ 가능 |
| **학습 데이터 누적 → YOLO 학습** | 장기 | 가장 정확. SAM 결과 라벨로 활용 |

## 10. 참고 정보

### 농진청 비교 데이터
- 사용자가 노션 공유 ([링크](https://aquamarine-vanadium-7cb.notion.site/_-_-_-37cf6d18e88c808e9563f00a66cfd32b))
- 농진청 알고리즘: ExG → 이진화 → 히스토그램 기반 파종 라인 → 카운팅
- 농진청 결과 이미지: 한 두둑에 2줄 빨간 점, total=338 lines=10 표시 (작은 영역)
- 농진청도 같은 날 같은 데이터 일부 사용 (다른 드론?)
- 농진청 영상은 2cm GSD라 콩잎 1-2 픽셀로 신호 집중. 우리는 4-6 픽셀로 분산이 원인

### 관련 satelite 폴더 코드 (참고용)
- `C:/Users/user/Desktop/분석프로젝트/satelite/wv3_detecting_gaps.py` — `detect_missing_plants()` 함수 (위성/저해상 드론용 결주 폴리곤)
- 우리 데이터(5.25mm GSD)엔 부적합 — closing 반경이 너무 작아 행간 못 메움

---

**마지막 업데이트**: 2026-06-30 (SAM v3 완료, 시각화 npz 저장 추가, 시각화 최적화 완료)
