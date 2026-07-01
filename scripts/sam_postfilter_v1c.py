"""
SAM 결과 후처리 강화 (1-C)
─────────────────────────────────────────────
기존 npz (HSV + 면적만 통과한 콩잎 후보) 위에 추가 필터:
  1. HSV 정제 — h ∈ [35,85] (진짜 녹색만, 황색·청록 컷)
  2. 종횡비 — bbox 가로/세로 비율 ∈ [1.0, 2.8]
              (콩잎은 거의 원형~타원, 가늘고 긴 풀잎 컷)
  3. 충실도 (extent) — mask_area / bbox_area > 0.45
                       (컴팩트한 잎만, 가지/줄기 컷)
  4. 면적 정밀화 — 5–80 cm² (어린 새싹~성장 잎)

사용:
  python -u scripts/sam_postfilter_v1c.py
  python -u scripts/sam_postfilter_v1c.py [npz_path]

출력:
  result/sam_test/GJSM-1-1_postfilter_v1c.png       (3-panel 비교 시각화)
  result/sam_test/GJSM-1-1_postfilter_v1c_stats.txt (필터별 입모율)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
OUT_DIR = ROOT / "result" / "sam_test"
ROI_M2 = 30.0 * 30.0       # 30 m × 30 m ROI
STD_PER_HA = 76923         # 65cm × 20cm 100% 입모 기준

# ─── 후처리 필터 파라미터 ──────────────────────────────
HUE_MIN_TIGHT, HUE_MAX_TIGHT = 35, 85   # 진짜 녹색만
SAT_MIN_TIGHT = 35
ASPECT_MIN, ASPECT_MAX = 1.0, 2.8
EXTENT_MIN = 0.45
AREA_MIN_CM2, AREA_MAX_CM2 = 5.0, 80.0


def compute_shape_metrics(leaf_arr: np.ndarray, gsd_ds: float):
    """leaf_arr 컬럼: [y, x, area_cm2, h, s, v, y0, x0, y1, x1]"""
    y, x = leaf_arr[:, 0], leaf_arr[:, 1]
    area_cm2 = leaf_arr[:, 2]
    h, s, v = leaf_arr[:, 3], leaf_arr[:, 4], leaf_arr[:, 5]
    y0, x0, y1, x1 = leaf_arr[:, 6], leaf_arr[:, 7], leaf_arr[:, 8], leaf_arr[:, 9]

    bbox_h = np.maximum(y1 - y0, 1)
    bbox_w = np.maximum(x1 - x0, 1)
    bbox_area_px = bbox_h * bbox_w
    px_area_cm2 = (gsd_ds * 100) ** 2
    bbox_area_cm2 = bbox_area_px * px_area_cm2

    # 종횡비: 항상 ≥ 1
    aspect = np.maximum(bbox_h, bbox_w) / np.minimum(bbox_h, bbox_w)
    # 충실도 (extent) = mask area / bbox area
    extent = area_cm2 / np.maximum(bbox_area_cm2, 1e-9)
    return aspect, extent


def apply_filter(leaf_arr: np.ndarray, aspect: np.ndarray, extent: np.ndarray,
                  use_hsv=True, use_shape=True, use_area=True):
    keep = np.ones(len(leaf_arr), dtype=bool)
    if use_hsv:
        keep &= (leaf_arr[:, 3] >= HUE_MIN_TIGHT) & (leaf_arr[:, 3] <= HUE_MAX_TIGHT)
        keep &= leaf_arr[:, 4] >= SAT_MIN_TIGHT
    if use_shape:
        keep &= (aspect >= ASPECT_MIN) & (aspect <= ASPECT_MAX)
        keep &= extent >= EXTENT_MIN
    if use_area:
        keep &= (leaf_arr[:, 2] >= AREA_MIN_CM2) & (leaf_arr[:, 2] <= AREA_MAX_CM2)
    return keep


def emergence_rate(n_leaves: int) -> tuple[float, float]:
    density_per_ha = n_leaves / (ROI_M2 / 10000)
    rate = density_per_ha / STD_PER_HA * 100
    return density_per_ha, rate


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
                OUT_DIR / "GJSM-1-1_sam_roi_test_ds1_v3.npz"
    if not npz_path.exists():
        print(f"❌ npz 없음: {npz_path}")
        print("   먼저 sam_roi_test.py를 실행해 주세요.")
        return

    print(f"로드: {npz_path}")
    d = np.load(npz_path)
    rgb = d["rgb_disp"]
    step = int(d["rgb_step"][0])
    leaf_arr = d["leaf_arr"]
    gsd_ds = float(d["gsd_ds"][0])
    n_all = int(d["n_all_masks"][0])
    print(f"  RGB {rgb.shape},  콩잎 후보 {len(leaf_arr)}개,  GSD={gsd_ds*1000:.2f}mm")

    aspect, extent = compute_shape_metrics(leaf_arr, gsd_ds)

    # ─── 단계별 필터 적용 ───
    keep_base = np.ones(len(leaf_arr), dtype=bool)              # 기존 (HSV+면적 통과한 것)
    keep_hsv = apply_filter(leaf_arr, aspect, extent, use_hsv=True, use_shape=False, use_area=False)
    keep_shape = apply_filter(leaf_arr, aspect, extent, use_hsv=False, use_shape=True, use_area=False)
    keep_area = apply_filter(leaf_arr, aspect, extent, use_hsv=False, use_shape=False, use_area=True)
    keep_all = apply_filter(leaf_arr, aspect, extent, use_hsv=True, use_shape=True, use_area=True)

    n_base = int(keep_base.sum())
    n_hsv = int(keep_hsv.sum())
    n_shape = int(keep_shape.sum())
    n_area = int(keep_area.sum())
    n_all_f = int(keep_all.sum())

    # ─── 통계 출력 ───
    lines = []
    lines.append("=" * 60)
    lines.append("SAM 후처리 필터 v1-C — 단계별 입모율")
    lines.append("=" * 60)
    lines.append(f"SAM 원본 마스크 수: {n_all}")
    lines.append(f"기존 필터 통과 (HSV[20,100] + 면적[1,200]): {n_base}\n")
    for name, n in [("HSV 강화 [35,85]", n_hsv),
                    ("형태 (aspect/extent)", n_shape),
                    ("면적 정밀 [5,80cm²]", n_area),
                    ("3개 모두 적용", n_all_f)]:
        d_ha, rate = emergence_rate(n)
        lines.append(f"  {name:20s}: {n:5d}개 → 밀도 {d_ha:>6,.0f}/ha → 입모율 {rate:>5.1f}%")
    lines.append(f"\n참고: 표준 100% 입모 = {STD_PER_HA:,}/ha (65×20cm 파종)")
    text = "\n".join(lines)
    print(text)

    (OUT_DIR / "GJSM-1-1_postfilter_v1c_stats.txt").write_text(text, encoding="utf-8")

    # ─── 시각화: 3-panel 비교 ───
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # (1) 원본 RGB
    axes[0].imshow(rgb)
    axes[0].set_title("(1) RGB ROI")
    axes[0].axis("off")

    # (2) 기존 필터 (npz에 저장된 콩잎 후보 전부)
    axes[1].imshow(rgb)
    xs = leaf_arr[keep_base, 1] / step
    ys = leaf_arr[keep_base, 0] / step
    axes[1].scatter(xs, ys, s=12, facecolors="none",
                    edgecolors="red", lw=0.9, alpha=0.85)
    d_ha, rate = emergence_rate(n_base)
    axes[1].set_title(f"(2) 기존 필터 — {n_base}개  ({rate:.1f}%)")
    axes[1].text(0.01, 0.99, f"total = {n_base}",
                 transform=axes[1].transAxes,
                 fontsize=13, fontweight="bold", color="white",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))
    axes[1].axis("off")

    # (3) 강화 필터 (HSV + 형태 + 면적 모두)
    axes[2].imshow(rgb)
    xs = leaf_arr[keep_all, 1] / step
    ys = leaf_arr[keep_all, 0] / step
    axes[2].scatter(xs, ys, s=12, facecolors="none",
                    edgecolors="lime", lw=0.9, alpha=0.95)
    d_ha, rate = emergence_rate(n_all_f)
    axes[2].set_title(f"(3) 1-C 강화 필터 — {n_all_f}개  ({rate:.1f}%)")
    axes[2].text(0.01, 0.99, f"total = {n_all_f}",
                 transform=axes[2].transAxes,
                 fontsize=13, fontweight="bold", color="white",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))
    axes[2].axis("off")

    plt.suptitle(f"SAM ROI 후처리 v1-C — GJSM-1-1 (HSV {HUE_MIN_TIGHT}-{HUE_MAX_TIGHT}, "
                 f"aspect ≤{ASPECT_MAX}, extent ≥{EXTENT_MIN}, area {AREA_MIN_CM2}-{AREA_MAX_CM2}cm²)",
                 fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / "GJSM-1-1_postfilter_v1c.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n저장: {out_png}")


if __name__ == "__main__":
    main()
