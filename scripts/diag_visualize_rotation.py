"""회전 마스크를 다운샘플 PNG로 저장해 두둑이 가로/세로 어느 축인지 확인."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
CACHE = ROOT / "result" / "_diag_GJSM-1-1_rot_cleaned.npy"
CACHE_ANGLE = ROOT / "result" / "_diag_GJSM-1-1_angle.txt"

import rasterio
from emergence_lib import (compute_exg, detrend_local, binarize_vegetation_raw,
                            estimate_row_angle, refine_mask_row_aware,
                            detect_row_lines_v2)


def get_rot():
    if CACHE.exists() and CACHE_ANGLE.exists():
        rot = np.load(CACHE).astype(bool)
        ang = float(CACHE_ANGLE.read_text().strip())
        # 원본도 필요 — bw_raw를 다시 구해야 하니 cache 따로
        return rot, ang
    print("building cache ...")
    tif = ROOT / "result" / "fields" / "GJSM-1-1.tif"
    with rasterio.open(tif) as src:
        data = src.read()
        gsd_m = abs(src.transform.a)
    rgb = data[:3].astype(np.float32)
    alpha = data[3] if data.shape[0] >= 4 else np.full(rgb.shape[1:], 255, dtype=np.uint8)
    del data
    valid_mask = alpha > 0
    exg = compute_exg(rgb)
    exg = detrend_local(exg, sigma=150)
    exg[~valid_mask] = 0
    bw_raw, _ = binarize_vegetation_raw(exg, valid_mask)
    angle = estimate_row_angle(bw_raw)
    rot_cleaned, _ = refine_mask_row_aware(bw_raw, angle, gsd_m,
                                            close_len_m=0.06, open_radius=1, min_size=30)
    np.save(CACHE, rot_cleaned.astype(np.uint8))
    CACHE_ANGLE.write_text(f"{angle:.4f}")
    # 원본 RGB도 다운샘플로 저장
    step = max(1, max(rgb.shape[1:]) // 2000)
    rgb_ds = rgb[:, ::step, ::step].astype(np.uint8)
    np.save(ROOT / "result" / "_diag_GJSM-1-1_rgb_ds.npy", rgb_ds)
    np.save(ROOT / "result" / "_diag_GJSM-1-1_bw_ds.npy", bw_raw[::step, ::step].astype(np.uint8))
    return rot_cleaned.astype(bool), angle


def main():
    rot, angle = get_rot()
    gsd_m = 0.00525
    print(f"row_angle from estimate_row_angle: {angle:.2f}°")
    print(f"rot shape: {rot.shape}")

    # 다운샘플 (보기 좋게)
    step = max(1, max(rot.shape) // 1500)
    rot_ds = rot[::step, ::step]
    print(f"downsampled: {rot_ds.shape}")

    # 두둑이 가로축인지 세로축인지 — 둘 다 시도
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(rot_ds, cmap="Greens")
    axes[0, 0].set_title("회전 마스크 rot_cleaned (downsampled)")
    axes[0, 0].axis("off")

    # axis=1 projection (각 row의 sum across X) — 두둑이 가로라면 여기 peak
    profile_y = rot_ds.astype(float).sum(axis=1)
    axes[0, 1].plot(profile_y, color="green")
    axes[0, 1].set_title("axis=1 sum (행 단위 합) — 두둑 가로면 여기 peak")
    axes[0, 1].set_xlabel("Y index")

    # axis=0 projection (각 col의 sum across Y) — 두둑이 세로라면 여기 peak
    profile_x = rot_ds.astype(float).sum(axis=0)
    axes[1, 0].plot(profile_x, color="green")
    axes[1, 0].set_title("axis=0 sum (열 단위 합) — 두둑 세로면 여기 peak")
    axes[1, 0].set_xlabel("X index")

    # 두 프로파일 비교 (FFT-like spikiness 측정)
    from scipy.ndimage import gaussian_filter1d
    profile_y_s = gaussian_filter1d(profile_y, sigma=2)
    profile_x_s = gaussian_filter1d(profile_x, sigma=2)
    pyn = profile_y_s[profile_y_s > 0]
    pxn = profile_x_s[profile_x_s > 0]
    y_cv = pyn.std()/pyn.mean() if pyn.size else 0
    x_cv = pxn.std()/pxn.mean() if pxn.size else 0
    axes[1, 1].text(0.1, 0.7, f"axis=1 (가로 ridge) CV={y_cv:.3f}\n"
                              f"axis=0 (세로 ridge) CV={x_cv:.3f}\n"
                              "(높은 CV = 명확한 peak = 두둑 방향)",
                    fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    plt.suptitle(f"GJSM-1-1 회전 마스크 진단 (row_angle={angle}°)", fontsize=14)
    plt.tight_layout()
    out_png = ROOT / "result" / "_diag_GJSM-1-1_rotation.png"
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    main()
