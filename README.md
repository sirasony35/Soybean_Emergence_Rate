# 콩 입모율 분석 파이프라인

13.8 GB 드론 정사영상에서 5개 필지(GJSM-1-1 ~ GJSM-2-3)의 콩 입모율·결주·간격 적합도를 산출한다.
GJSM-1-1은 위/아래 파종기(일반 국내 vs 스마트 파종기 실증) 비교까지 수행한다.

## 입력 데이터

```
data/      간척지 초고해상도 정사영상-orthomosaic.tiff   # 13.8 GB, EPSG:32652, GSD 5.25mm
shapefile/ GJSM-{1-1, 1-2, 1-3, 2-2, 2-3}_Boundary.zip   # EPSG:4326 단일 폴리곤
```

## 실행 순서

```bash
# 한번에 (권장, 약 80분)
python -u scripts/run_all.py

# 또는 단계별
python -u scripts/01_crop_fields.py            # 약  8분
python -u scripts/02_analyze_emergence.py       # 약 70분
python -u scripts/03_postprocess_visualize.py   # 약  1분
python -u scripts/04_detail_overlay.py          # 약  2분
```

| 단계 | 스크립트 | 역할 | 입력 | 출력 |
|---|---|---|---|---|
| 0 | `emergence_lib.py` | 공통 알고리즘 모듈 (다른 스크립트가 import) | — | — |
| 1 | `01_crop_fields.py` | 정사영상을 SHP로 마스크 클립 → 필지별 TIF | `data/`+`shapefile/` | `result/fields/{GJSM-X-Y}.tif` |
| 2 | `02_analyze_emergence.py` | 필지별 입모율 본 분석 | `result/fields/` | `result/emergence/{GJSM-X-Y}_{plants,rows,gaps}.gpkg`, `{...}_rows.csv`, `emergence_summary.csv` |
| 3 | `03_postprocess_visualize.py` | PPT용 종합 시각화 + GJSM-1-1 파종기 사선 분할 비교 | 2단계 산출 | `{GJSM-X-Y}_overview_ppt.png`, `GJSM-1-1_seeder_compare.png`, `GJSM-1-1_seeder_stats.csv` |
| 4 | `04_detail_overlay.py` | 작은 영역 RGB 상세 오버레이 + 조간/주간 컬럼 추가 | 2·3단계 산출 | `{GJSM-X-Y}_detail.png`, `GJSM-1-1_detail_위/아래_*.png`, `emergence_summary.csv` 갱신, `seeder_stats.csv` 갱신 |

## 알고리즘 요약

1. ExG (2g-r-b) 색지수 + 가우시안 detrend (조명 보정)
2. Otsu+Sauvola 1차 이진화 (노이즈 허용 — 약한 새싹 보존)
3. Radon transform 으로 두둑 각도 자동 검출
4. 두둑방향 closing + size 필터 (잡노이즈 정리)
5. 회전 후 1D 프로파일 → find_peaks → 두둑 라인
6. 각 두둑 ±0.12m 밴드 → 라인 따라 1D 피크 → 개체 위치
7. 인접 피크 간격 > 1.5×median → 결주 구간
8. (GJSM-1-1) 식생 밀도 가장 낮은 사선 위치 자동 검출 → 파종기 경계

## 출력물 (최종)

### `result/fields/` (중간 산출, 다시 안 만들면 04 못 돌림)
- `GJSM-{X-Y}.tif` × 5 — 필지별 RGBA 정사영상 (EPSG:32652, LZW 압축)

### `result/emergence/` (최종)

**필지별 (5세트)**
- `{GJSM-X-Y}_overview_ppt.png` — 4패널 PPT용 종합 시각화 (RGB, 두둑검출, 두둑별 입모율, 주간분포)
- `{GJSM-X-Y}_detail.png` — 6m×5m 상세 RGB 오버레이 (노란선 두둑, 빨간원 개체, total=N lines=M 캡션)
- `{GJSM-X-Y}_plants.gpkg` — 검출된 모든 개체 포인트
- `{GJSM-X-Y}_rows.gpkg` — 두둑 중심선 (입모율/기대수 속성 포함)
- `{GJSM-X-Y}_gaps.gpkg` — 결주 구간 라인스트링
- `{GJSM-X-Y}_rows.csv` — 두둑별 통계 (개체수, 기대수, 입모율, 결주수)

**필지 통합**
- `emergence_summary.csv` — 5필지 통합 요약 + 조간/주간 컬럼

**GJSM-1-1 파종기 비교**
- `GJSM-1-1_seeder_compare.png` — 사선 분할 비교 시각화 + 통계 표
- `GJSM-1-1_seeder_stats.csv` — zone별 통계 (조간/주간 포함)
- `GJSM-1-1_detail_위_일반국내파종기.png` — 위 영역 상세
- `GJSM-1-1_detail_아래_스마트파종기.png` — 아래 영역 상세

## 표준 재배 제원 (재단 가능, `emergence_lib.py`)

- 조간 (1두둑 안 행간) = 30 cm  (적합 허용 ±7 cm)
- 주간 (행 내 개체 간격) = 20 cm  (적합 허용 ±5 cm)
- 입모율 = 검출개체수 / 기대개체수 × 100,  기대개체수 = 두둑유효길이 / 20cm
