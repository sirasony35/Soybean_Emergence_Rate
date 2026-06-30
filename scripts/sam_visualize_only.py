"""
npz 결과만 읽어 빠르게 시각화.
SAM 다시 안 돌리고 npz에 저장된 콩잎 centroid만으로 PNG 생성.
실행: python -u scripts/sam_visualize_only.py [npz_path]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
OUT_DIR = ROOT / "result" / "sam_test"


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
                OUT_DIR / "GJSM-1-1_sam_roi_test_ds1_v3.npz"
    print(f"로드: {npz_path}")
    d = np.load(npz_path)
    rgb = d["rgb_disp"]
    step = int(d["rgb_step"][0])
    leaves = d["leaf_arr"]   # (N, 10) [y, x, area, h, s, v, y0, x0, y1, x1]
    gsd = float(d["gsd_ds"][0])
    n_all = int(d["n_all_masks"][0])
    print(f"  rgb shape: {rgb.shape},  콩잎 {len(leaves)}개,  GSD ds={gsd*1000:.2f}mm")

    H, W = rgb.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # (1) 원본 RGB
    axes[0].imshow(rgb)
    axes[0].set_title("(1) 원본 RGB (다운샘플 표시)")
    axes[0].axis("off")

    # (2) RGB + 콩잎 빨간 점 (일괄 scatter)
    axes[1].imshow(rgb)
    xs = leaves[:, 1] / step
    ys = leaves[:, 0] / step
    axes[1].scatter(xs, ys, s=15, facecolors="none",
                   edgecolors="red", lw=1.0, alpha=0.9)
    axes[1].set_title(f"(2) 콩잎 검출 ({len(leaves)}개 / SAM 마스크 {n_all}개)")
    axes[1].axis("off")
    axes[1].text(0.01, 0.99, f"total = {len(leaves)}",
                transform=axes[1].transAxes,
                fontsize=14, fontweight="bold", color="white",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", alpha=0.6))

    # 통계
    if len(leaves) > 0:
        area_mean = leaves[:, 2].mean()
        ROI_M2 = 30.0 * 30.0
        density = len(leaves) / (ROI_M2 / 10000)
        STD = 76923  # 65×20cm 100% 입모 ha당
        rate = density / STD * 100
        info = (f"면적 평균 {area_mean:.1f}cm²  ·  "
                f"밀도 {density:,.0f}/ha  ·  "
                f"추정 입모율 {rate:.1f}%")
        plt.suptitle(f"SAM ROI v3 — {info}", fontsize=13)

    plt.tight_layout()
    out_png = npz_path.with_suffix(".png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {out_png}")


if __name__ == "__main__":
    main()
