# 콩 입모율·파종기 효율성 분석 파이프라인

새만금 간척지 5.25mm GSD 드론 정사영상 기반. **두둑 검출 → 두둑별 dual-row 검출 → 줄 내 파종 간격 → 시각화 → 파종기 비교 리포트** 5단계 파이프라인.

## 최종 목적

1. **파종기 효율성 비교**: 파종 간격 균일성(CV), 결주율, 입모율 정량 비교 (스마트 vs 일반)
2. **정밀농업 모니터링 알고리즘**: 드론/위성 RGB → 발아 잎 검출 → 파종 간격·결주율 자동 산출

## 폴더 구조

```
data/           간척지 초고해상도 정사영상-orthomosaic.tiff  # 13.8GB, EPSG:32652, 5.25mm GSD
shapefile/      GJSM-{1-1,1-2,1-3,2-2,2-3}_Boundary.zip     # 5필지 폴리곤 (EPSG:4326)
scripts/
  01_crop_fields.py                # 정사영상 → 필지 TIF crop
  sam_run_field_full.py            # SAM으로 콩잎 검출 → npz 저장
  run_pipeline.py                  # 통합 실행기
  pipeline/
    common.py                      # 공통 유틸 (좌표, 경로)
    step1_ridges.py                # 두둑 검출 (gray 밴드패스 + SAM 결합)
    step2_rows.py                  # 두둑별 dual-row 검출 + 조간
    step3_spacing.py               # 줄 내 파종 간격 + 결주
    step4_visualize.py             # 필지 overview + 두둑별 그리드
    step5_seeder_compare.py        # 파종기 A/B 비교 리포트
    _verify_step1_crop.py          # (검증) Step 1 crop 3장 확인
result/
  fields/                          # 필지별 TIF (SAM 입력)
  sam_leaves/                      # SAM 콩잎 검출 npz (파이프라인 입력)
  pipeline/{field}/                # step1~4 산출 (npz + PNG)
  report/                          # step5 파종기 비교 (PNG, CSV, MD)
```

## 실행

### 한 필지 처리 (SAM → step1~4)

```bash
conda activate satelite
python -u scripts/run_pipeline.py --field GJSM-1-1_Smart
```

`--skip-sam` 옵션: `result/sam_leaves/{field}_sam_FULL_v4.npz`가 이미 있으면 SAM 건너뛰고 step1부터.

### 파종기 A/B 비교 리포트

```bash
python -u scripts/pipeline/step5_seeder_compare.py \
    GJSM-1-1_Smart GJSM-1-1_normal \
    --labels "스마트 파종기,일반 파종기"
```

### 두 필지 처리 + 자동 비교

```bash
python -u scripts/run_pipeline.py \
    --field GJSM-1-1_Smart --field GJSM-1-1_normal \
    --skip-sam --then-compare \
    --labels "스마트 파종기,일반 파종기"
```

## 알고리즘

| 단계 | 신호 | 결과 |
|---|---|---|
| 1. 두둑 검출 | 필지 mask 내 gray weighted profile + SAM 밀도 → 반복주기 60~130cm bandpass → 각도 스캔 → 두둑 중심 1D peak | 두둑 중심선 + 두둑 방향 (각도) |
| 2. 두둑별 dual-row | 두둑 밴드 내 gray 반전(파종 자국 어두운 홈) + SAM 밀도 결합 → sub-profile bandpass 5~90cm → top-2 peak | 파종 줄 2개 + 조간 |
| 3. 줄 내 파종 간격 | SAM 잎 → 줄 밴드 내 8cm dedup(개체 병합) → ridge 방향 사영 → 인접 gap → median, CV, 결주(>30cm) | 줄별 median, mean, std, 결주 수, 입모율 |
| 4. 시각화 | RGB + 파종 줄 + 잎 dot + 통계 | field_overview.png, ridge_grid_p*.png |
| 5. 파종기 비교 | 필지 A/B의 median, 줄별 CV, 결주율, 입모율, 적합률 | seeder_comparison.png/csv/md |

## 표준 재배 제원

- **주간(파종 간격)**: 20 cm (적합 ±5 cm → 15~25cm)
- **조간(두둑 안 dual-row 간격)**: 30 cm(스마트) 또는 70 cm(일반)
- **입모율**: 검출 잎 개체수 / 기대 개체수(=두둑 유효 길이 / 20cm) × 100
- **결주 판정**: gap > 30cm (스펙 20cm × 1.5)
- **파종 CV**: 줄별 std/mean × 100 → 여러 줄 CV의 median (낮을수록 균일)

## 검증 결과 (GJSM-1-1)

| 지표 | 스마트 파종기 | 일반 파종기 |
|---|---:|---:|
| 파종 median | **22.0 cm** (스펙 20) | 29.6 cm |
| 파종 CV (균일성) | **21.0%** | 27.9% |
| 결주율 | **20.4%** | 46.3% |
| 입모율 median | **67.6%** | 45.8% |
| 적합 줄 비율 | **63.8%** | 2.1% |
| 종합점수 | **74.6 / 100** | 45.5 / 100 |

→ 모든 지표에서 스마트 파종기 우수.
