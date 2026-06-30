"""
두둑(ridge) → 파종 라인(line) → 개체(plant) 2단계 검출 진단.
GJSM-1-1 캐시 사용 (ExG 12분 스킵).
한 타일을 골라 시각화 + 파라미터 sweep.
"""
from __future__ import annotations
from pathlib import Path
import sys
import time
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, rotate as ndi_rotate

sys.path.insert(0, str(Path(__file__).parent))
from emergence_lib import (
    estimate_row_angle, refine_mask_row_aware,
    estimate_tile_angle_only, find_dominant_angles, snap_to_dominant,
)

ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELDS_DIR = ROOT / "result" / "fields"
OUT_DIR = ROOT / "result"
OUT_DIR.mkdir(exist_ok=True)

FIELD = "GJSM-1-1"
TILE_SIZE_M = 15.0

# 표준 재배 제원
STD_RIDGE_PITCH_M = 0.65    # 두둑 주기 (= 조간 30cm + 두둑 간격 35cm)
STD_INTRA_RIDGE_CM = 30.0
STD_LINE_PITCH_M = 0.30     # 한 두둑 안 dual-row 라인 간격


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # 캐시 로드
    print(f"[load] {FIELD} bw 캐시 로드")
    t0 = time.time()
    bw_raw = np.load(FIELDS_DIR / f"_cache_{FIELD}_bw.npy").astype(bool)
    valid_mask = np.load(FIELDS_DIR / f"_cache_{FIELD}_valid.npy").astype(bool)
    print(f"  shape={bw_raw.shape}, {time.time()-t0:.1f}s")

    # TIF transform/RGB 다운샘플 로드 (시각화용)
    with rasterio.open(FIELDS_DIR / f"{FIELD}.tif") as src:
        gsd_m = abs(src.transform.a)
        H, W = src.height, src.width
        # 다운샘플 RGB
        step = max(1, max(H, W) // 1500)
        rgb_ds = src.read(indexes=[1,2,3],
                          out_shape=(3, H//step, W//step)).astype(np.uint8)
    rgb_ds = np.transpose(rgb_ds, (1, 2, 0))
    print(f"  gsd={gsd_m*1000:.3f}mm/px, rgb_ds={rgb_ds.shape}")

    # 타일 격자 — 1차로 각도 수집해 dominant 찾기
    tile_px = int(TILE_SIZE_M / gsd_m)
    n_ty = (H + tile_px - 1) // tile_px
    n_tx = (W + tile_px - 1) // tile_px
    angles = []
    for ty in range(n_ty):
        for tx in range(n_tx):
            y0=ty*tile_px; x0=tx*tile_px
            y1=min(y0+tile_px,H); x1=min(x0+tile_px,W)
            if valid_mask[y0:y1,x0:x1].sum() < 0.3 * (y1-y0)*(x1-x0):
                continue
            a, _ = estimate_tile_angle_only(bw_raw[y0:y1,x0:x1], gsd_m)
            if a is not None:
                angles.append(a)
    dominant = find_dominant_angles(angles, bin_width=5.0)
    print(f"  dominant 각도: {dominant}")

    # 한 두둑 영역 잘 보이는 타일 선택 — center 근처
    sample_ty = n_ty // 2
    sample_tx = n_tx // 2
    sy0 = sample_ty * tile_px; sx0 = sample_tx * tile_px
    sy1 = min(sy0+tile_px,H); sx1 = min(sx0+tile_px,W)
    tile_bw = bw_raw[sy0:sy1, sx0:sx1]
    angle_used = snap_to_dominant(estimate_row_angle(tile_bw), dominant, 15.0) or dominant[0]
    print(f"  샘플 타일 (ty,tx)=({sample_ty},{sample_tx}), 각도={angle_used:.2f}°")

    # 회전 + closing 정리 (기존 흐름)
    rot_cleaned, _ = refine_mask_row_aware(
        tile_bw, angle_used, gsd_m,
        close_len_m=0.06, open_radius=1, min_size=30,
    )
    print(f"  rot_cleaned shape={rot_cleaned.shape}")

    # 가로축 합 → 1D profile (회전 후 두둑은 가로축, 단면은 Y축)
    profile = rot_cleaned.astype(np.float32).sum(axis=1)
    profile_s = gaussian_filter1d(profile, sigma=5.0)

    nz = profile_s[profile_s > 0]
    print(f"\n1D 프로파일 (sigma=5):  min={nz.min():.0f}, max={nz.max():.0f}, "
          f"median={np.median(nz):.0f}, q25={np.quantile(nz,0.25):.0f}, "
          f"q75={np.quantile(nz,0.75):.0f}")

    # ===== 새 흐름: 모든 줄을 잡고 → 간격 분포 분석 → 두둑 클러스터링 =====
    print(f"\n=== 1단계: 모든 파종 줄 검출 (min_dist=20cm, 조밀하게) ===")
    # 더 관대한 임계 — 두둑 안 가까운 두 줄도 모두 잡기
    for min_dist_cm, h_q, prom_f in [
        (18, 0.20, 0.05),
        (20, 0.25, 0.05),
        (22, 0.25, 0.06),
        (25, 0.30, 0.08),
    ]:
        h_thr = float(np.quantile(nz, h_q))
        prom = max(1, float(nz.mean()) * prom_f)
        dist = max(5, int(min_dist_cm / 100 / gsd_m))
        peaks, _ = find_peaks(profile_s, distance=dist,
                               height=h_thr, prominence=prom)
        if len(peaks) >= 2:
            gaps_cm = np.diff(peaks) * gsd_m * 100
            print(f"  min_dist={min_dist_cm}cm, h_q={h_q:.2f}, prom={prom_f:.2f}: "
                  f"줄 {len(peaks)}, gap 평균 {gaps_cm.mean():.1f}cm")

    # 선택: min_dist 20cm, h_q 0.25 (관대)
    h_thr = float(np.quantile(nz, 0.25))
    prom = max(1, float(nz.mean()) * 0.05)
    dist = max(5, int(0.20 / gsd_m))
    all_lines_y, _ = find_peaks(profile_s, distance=dist,
                                  height=h_thr, prominence=prom)
    print(f"\n선택: 줄 {len(all_lines_y)}개  (min_dist=20cm, h_q=0.25, prom=mean*0.05)")

    # 줄 간격 분포 분석
    if len(all_lines_y) >= 2:
        gaps_cm = np.diff(all_lines_y) * gsd_m * 100
        print(f"\n=== 줄 간격 분포 ===")
        print(f"  전체 n={len(gaps_cm)}, min={gaps_cm.min():.1f}cm, max={gaps_cm.max():.1f}cm")
        # 히스토그램 (10cm bin)
        for lo in range(0, 120, 10):
            n = ((gaps_cm >= lo) & (gaps_cm < lo+10)).sum()
            bar = "#" * min(60, n)
            print(f"  {lo:3d}~{lo+10:3d}cm: {bar} ({n})")
        # 두 mode 추정 (intra vs inter)
        # GMM-like 단순 클러스터링: 50cm 이하 = intra, 50cm 초과 = inter
        intra = gaps_cm[gaps_cm < 50]
        inter = gaps_cm[gaps_cm >= 50]
        if len(intra):
            print(f"\n  intra (<50cm): 평균 {intra.mean():.1f}cm ± {intra.std():.1f} (n={len(intra)})")
        if len(inter):
            print(f"  inter (≥50cm): 평균 {inter.mean():.1f}cm ± {inter.std():.1f} (n={len(inter)})")

    # ===== 2단계: 두둑 클러스터링 (greedy) =====
    # 인접 줄 간격 ≤ INTRA_MAX(75cm: 30cm 또는 70cm 양 dual-row 포함) → 같은 두둑
    # 단, 두 줄로 한 두둑 이므로 줄을 2개씩 묶음
    INTRA_MAX_CM = 75.0
    print(f"\n=== 2단계: 두둑 클러스터 (인접 줄 간격 ≤ {INTRA_MAX_CM}cm) ===")
    if len(all_lines_y) >= 2:
        ridges = []
        current = [int(all_lines_y[0])]
        for i in range(1, len(all_lines_y)):
            gap = (all_lines_y[i] - all_lines_y[i-1]) * gsd_m * 100  # cm
            if gap <= INTRA_MAX_CM and len(current) < 2:
                current.append(int(all_lines_y[i]))
            else:
                ridges.append(current)
                current = [int(all_lines_y[i])]
        ridges.append(current)
        n_dual = sum(1 for r in ridges if len(r) == 2)
        n_single = sum(1 for r in ridges if len(r) == 1)
        print(f"  두둑 {len(ridges)}개  (dual-row {n_dual}, single {n_single})")
        # 두둑 안 dual-row 간격 (조간) 분포
        dual_gaps_cm = []
        for r in ridges:
            if len(r) == 2:
                dual_gaps_cm.append((r[1] - r[0]) * gsd_m * 100)
        if dual_gaps_cm:
            dg = np.array(dual_gaps_cm)
            print(f"  dual-row 간격 (조간): 평균 {dg.mean():.1f}cm, "
                  f"min {dg.min():.1f}cm, max {dg.max():.1f}cm, n={len(dg)}")
            # 사용자가 말한 30cm vs 70cm 두 mode?
            narrow = dg[dg < 50]; wide = dg[dg >= 50]
            if len(narrow):
                print(f"    좁은(< 50cm): 평균 {narrow.mean():.1f}cm (n={len(narrow)})")
            if len(wide):
                print(f"    넓은(≥ 50cm): 평균 {wide.mean():.1f}cm (n={len(wide)})")
        # 두둑 간 거리 (행간 = inter-ridge): 인접 두둑의 첫 줄 - 이전 두둑의 마지막 줄
        inter_gaps_cm = []
        for i in range(1, len(ridges)):
            prev_last = ridges[i-1][-1]
            curr_first = ridges[i][0]
            inter_gaps_cm.append((curr_first - prev_last) * gsd_m * 100)
        if inter_gaps_cm:
            ig = np.array(inter_gaps_cm)
            print(f"  두둑 간 거리 (행간): 평균 {ig.mean():.1f}cm, n={len(ig)}")
        # 시각화용
        ridge_peaks = np.array([np.mean(r) for r in ridges]).astype(int)
    else:
        ridge_peaks = np.array([], dtype=int)

    # ----- 시각화 -----
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1])

    # (1) 회전된 마스크 + 검출선 표시
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.imshow(rot_cleaned, cmap="Greens", aspect="auto")
    for rp in ridge_peaks:
        ax0.axhline(rp, color="orange", lw=0.8, alpha=0.8)
    for lp in all_lines_y:
        ax0.axhline(lp, color="red", lw=0.4, alpha=0.6)
    ax0.set_title(f"(1) 회전된 마스크 — 주황: 두둑({len(ridge_peaks)}) / 빨강: 파종라인({len(all_lines_y)})")
    ax0.set_xlabel("X (회전 좌표)")
    ax0.set_ylabel("Y (회전 좌표 = 단면 위치)")

    # (2) 1D 프로파일 + 두둑 + 라인 위치
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.plot(profile_s, np.arange(len(profile_s)), color="green", lw=0.8)
    ax1.scatter(profile_s[ridge_peaks], ridge_peaks, color="orange",
                s=30, marker="o", label=f"두둑 {len(ridge_peaks)}")
    ax1.scatter(profile_s[all_lines_y], all_lines_y, color="red",
                s=10, marker=".", label=f"라인 {len(all_lines_y)}")
    ax1.set_title("(2) 가로축 합 1D 단면")
    ax1.set_xlabel("식생 픽셀 합")
    ax1.set_ylabel("Y (회전좌표)")
    ax1.invert_yaxis()
    ax1.legend()

    # (3) 두둑 간격 히스토그램
    ax2 = fig.add_subplot(gs[1, 0])
    if len(ridge_peaks) >= 2:
        rg_cm = np.diff(ridge_peaks) * gsd_m * 100
        ax2.hist(rg_cm, bins=20, color="orange", alpha=0.7)
        ax2.axvline(STD_RIDGE_PITCH_M*100, color="red", ls="--",
                   label=f"표준 {STD_RIDGE_PITCH_M*100:.0f}cm")
        ax2.set_title(f"(3) 두둑 간격 (n={len(rg_cm)}, 평균 {rg_cm.mean():.1f}cm)")
        ax2.legend()

    # (4) 라인 간격 분포 (intra / inter 구분)
    ax3 = fig.add_subplot(gs[1, 1])
    if len(all_lines_y) >= 2:
        lg_cm = np.diff(all_lines_y) * gsd_m * 100
        ax3.hist(lg_cm, bins=30, range=(0, 80), color="darkred", alpha=0.7)
        ax3.axvline(STD_INTRA_RIDGE_CM, color="green", ls="--", label="조간 30cm")
        ax3.axvline(STD_RIDGE_PITCH_M*100, color="orange", ls="--", label="행간 ~65cm")
        ax3.set_title(f"(4) 라인 간격 (n={len(lg_cm)})")
        ax3.legend()

    # (5) 샘플 두둑 sub-profile (가운데 두둑 1개 확대)
    ax4 = fig.add_subplot(gs[1, 2])
    if len(ridge_peaks) > 0:
        mid_ridge = ridge_peaks[len(ridge_peaks)//2]
        y_lo = max(0, mid_ridge - band_half_px)
        y_hi = min(len(profile_s), mid_ridge + band_half_px + 1)
        ax4.plot(profile_s[y_lo:y_hi], np.arange(y_lo, y_hi), color="green")
        ax4.axhline(mid_ridge, color="orange", lw=1.5, label="두둑 중심")
        # 이 두둑에 속하는 라인들
        for lp in all_lines_y:
            if y_lo <= lp < y_hi:
                ax4.axhline(lp, color="red", lw=1, alpha=0.7)
        ax4.set_title(f"(5) 샘플 두둑 ±30cm sub-profile")
        ax4.invert_yaxis()
        ax4.legend()

    # (6) 회전된 마스크 zoom — 가운데 두둑 1개 ±30cm
    ax5 = fig.add_subplot(gs[2, :])
    if len(ridge_peaks) > 0:
        mid_ridge = ridge_peaks[len(ridge_peaks)//2]
        y_lo = max(0, mid_ridge - band_half_px*2)
        y_hi = min(rot_cleaned.shape[0], mid_ridge + band_half_px*2 + 1)
        zoom = rot_cleaned[y_lo:y_hi, :]
        ax5.imshow(zoom, cmap="Greens", aspect="auto")
        for rp in ridge_peaks:
            if y_lo <= rp < y_hi:
                ax5.axhline(rp - y_lo, color="orange", lw=1.5)
        for lp in all_lines_y:
            if y_lo <= lp < y_hi:
                ax5.axhline(lp - y_lo, color="red", lw=1)
        ax5.set_title(f"(6) 가운데 두둑 1개 zoom (±{band_half_px*2*gsd_m*100:.0f}cm 범위)")

    plt.suptitle(f"{FIELD} 두둑 검출 진단 (타일 ({sample_ty},{sample_tx}), 각도 {angle_used:.1f}°)",
                fontsize=14)
    plt.tight_layout()
    out_png = OUT_DIR / f"_diag_{FIELD}_ridge.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n시각화 저장: {out_png}")


if __name__ == "__main__":
    main()
