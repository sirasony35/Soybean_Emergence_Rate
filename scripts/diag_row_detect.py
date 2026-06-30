"""
빠른 진단: GJSM-1-1 한 필지에 대해 row 검출 파라미터 튜닝.
중간 마스크(rot_cleaned)를 1회만 계산해 디스크에 저장 → 이후 빠르게 재실험.
"""
from pathlib import Path
import sys
import time
import numpy as np
import rasterio
sys.path.insert(0, str(Path(__file__).parent))
from emergence_lib import (
    compute_exg, detrend_local, binarize_vegetation_raw,
    estimate_row_angle, refine_mask_row_aware,
    detect_row_lines_v2, cluster_lines_to_ridges,
    compute_line_spacings_split,
)

ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELD = "GJSM-1-1"
CACHE = ROOT / "result" / f"_diag_{FIELD}_rot_cleaned.npy"
CACHE_ANGLE = ROOT / "result" / f"_diag_{FIELD}_angle.txt"

def get_or_build_cache():
    if CACHE.exists() and CACHE_ANGLE.exists():
        print(f"  [cache HIT] {CACHE.name}")
        rot = np.load(CACHE)
        ang = float(CACHE_ANGLE.read_text().strip())
        return rot, ang
    tif = ROOT / "result" / "fields" / f"{FIELD}.tif"
    print(f"  [build] reading {tif.name} ...")
    t0 = time.time()
    with rasterio.open(tif) as src:
        data = src.read()
        gsd_m = abs(src.transform.a)
    rgb = data[:3].astype(np.float32)
    alpha = data[3] if data.shape[0] >= 4 else np.full(rgb.shape[1:], 255, dtype=np.uint8)
    del data
    valid_mask = alpha > 0
    print(f"  [load] {time.time()-t0:.1f}s")

    t0 = time.time()
    exg = compute_exg(rgb)
    exg = detrend_local(exg, sigma=150)
    exg[~valid_mask] = 0
    print(f"  [exg] {time.time()-t0:.1f}s")

    t0 = time.time()
    bw_raw, _ = binarize_vegetation_raw(exg, valid_mask)
    print(f"  [bin] {time.time()-t0:.1f}s")

    t0 = time.time()
    angle = estimate_row_angle(bw_raw)
    print(f"  [angle] {time.time()-t0:.1f}s, {angle:.2f}°")

    t0 = time.time()
    rot_cleaned, _ = refine_mask_row_aware(bw_raw, angle, gsd_m,
                                            close_len_m=0.06, open_radius=1, min_size=30)
    print(f"  [clean] {time.time()-t0:.1f}s, shape={rot_cleaned.shape}")

    np.save(CACHE, rot_cleaned.astype(np.uint8))
    CACHE_ANGLE.write_text(f"{angle:.4f}")
    print(f"  → cached")
    return rot_cleaned, angle


def try_params(rot, gsd_m,
               min_line_spacing_m=0.22, smooth_sigma=3.0,
               height_q=0.55, prom_q1=0.45, prom_q2=0.85):
    profile = rot.astype(np.float32).sum(axis=1)
    from scipy.ndimage import gaussian_filter1d
    profile_s = gaussian_filter1d(profile, sigma=smooth_sigma)
    nonzero = profile_s[profile_s > 0]
    if nonzero.size < 20:
        return None
    h = float(np.quantile(nonzero, height_q))
    prom = max(1.0, float(np.quantile(nonzero, prom_q2) - np.quantile(nonzero, prom_q1)) * 0.5)
    from scipy.signal import find_peaks
    min_dist_px = max(5, int(min_line_spacing_m / gsd_m))
    peaks, _ = find_peaks(profile_s, distance=min_dist_px, height=h, prominence=prom)
    return peaks, profile_s, h, prom


def main():
    rot, angle = get_or_build_cache()
    rot = rot.astype(bool)
    gsd_m = 0.00525

    # 프로파일 통계 먼저 출력
    profile = rot.astype(np.float32).sum(axis=1)
    from scipy.ndimage import gaussian_filter1d as g1d
    profile_s = g1d(profile, sigma=3.0)
    nonzero = profile_s[profile_s > 0]
    print(f"\n=== Profile stats (sigma=3) ===")
    print(f"  shape={profile_s.shape}, nonzero count={len(nonzero)}")
    print(f"  nonzero: min={nonzero.min():.0f}, max={nonzero.max():.0f}, "
          f"mean={nonzero.mean():.0f}, median={np.median(nonzero):.0f}")
    print(f"  quantiles: q10={np.quantile(nonzero, 0.10):.0f}, "
          f"q25={np.quantile(nonzero, 0.25):.0f}, "
          f"q50={np.quantile(nonzero, 0.50):.0f}, "
          f"q75={np.quantile(nonzero, 0.75):.0f}, "
          f"q90={np.quantile(nonzero, 0.90):.0f}")

    from scipy.signal import find_peaks as fp
    def sweep(label, profile_s, min_dist_px, height, prominence):
        peaks, _ = fp(profile_s, distance=min_dist_px, height=height, prominence=prominence)
        ridges = cluster_lines_to_ridges(peaks, gsd_m, intra_max_cm=42.0)
        intra, inter = compute_line_spacings_split(peaks, gsd_m, intra_max_cm=42.0)
        intra_str = f"{intra.mean():.1f}±{intra.std():.1f}cm n={len(intra)}" if len(intra) else "—"
        inter_str = f"{inter.mean():.1f}±{inter.std():.1f}cm n={len(inter)}" if len(inter) else "—"
        print(f"  {label}: lines={len(peaks)}, ridges={len(ridges)}, "
              f"intra={intra_str}, inter={inter_str}")

    min_dist_px = max(5, int(0.22 / gsd_m))   # 22cm = 42px
    print(f"\n=== Aggressive sweep (min_dist={min_dist_px}px=22cm) ===")
    sweep("no_thr (only min_dist)", profile_s, min_dist_px, None, None)
    sweep("h=q50 only", profile_s, min_dist_px, np.quantile(nonzero, 0.50), None)
    sweep("h=q25 only", profile_s, min_dist_px, np.quantile(nonzero, 0.25), None)
    sweep("h=q10 only", profile_s, min_dist_px, np.quantile(nonzero, 0.10), None)
    sweep("h=q25 + prom=500", profile_s, min_dist_px, np.quantile(nonzero, 0.25), 500)
    sweep("h=q25 + prom=1000", profile_s, min_dist_px, np.quantile(nonzero, 0.25), 1000)
    sweep("h=q25 + prom=2000", profile_s, min_dist_px, np.quantile(nonzero, 0.25), 2000)
    sweep("h=mean*0.5 + prom=q25", profile_s, min_dist_px,
          float(nonzero.mean()*0.5), float(np.quantile(nonzero, 0.25)))

    # sigma 영향
    print(f"\n=== sigma sweep (min_dist=22cm, h=q25, no prom) ===")
    for sig in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
        p_s = g1d(profile, sigma=sig)
        nz = p_s[p_s > 0]
        sweep(f"sigma={sig}", p_s, min_dist_px, np.quantile(nz, 0.25), None)

    # min_dist 영향
    print(f"\n=== min_dist sweep (sigma=2, h=q25, no prom) ===")
    p_s = g1d(profile, sigma=2.0)
    nz = p_s[p_s > 0]
    for sp_cm in [15, 18, 20, 22, 25, 28]:
        d = max(5, int(sp_cm / 100 / gsd_m))
        sweep(f"dist={sp_cm}cm({d}px)", p_s, d, np.quantile(nz, 0.25), None)

    # 한 라인 개체 검출 sweep — 중앙 두둑 하나 선택
    from emergence_lib import detect_row_lines_v2
    profile_s2 = g1d(profile, sigma=3.0)
    nz2 = profile_s2[profile_s2 > 0]
    peaks, _ = fp(profile_s2,
                  distance=int(0.22/gsd_m),
                  height=np.quantile(nz2, 0.25),
                  prominence=nz2.mean()*0.057)
    print(f"\n=== Plant detection sweep (use middle row line) ===")
    if len(peaks) > 0:
        # 중간 정도 위치 라인 선택
        mid_line_y = int(peaks[len(peaks)//2])
        band_half_px = int(0.12 / gsd_m)  # ±12cm
        r0 = max(0, mid_line_y - band_half_px)
        r1 = min(rot.shape[0], mid_line_y + band_half_px + 1)
        band = rot[r0:r1]
        col_profile = band.astype(np.float32).sum(axis=0)
        col_profile_s = g1d(col_profile, sigma=1.5)
        nz_col = col_profile_s[col_profile_s > 0]
        print(f"  line y={mid_line_y}, band {r0}-{r1}, col_profile len={len(col_profile_s)}")
        print(f"  col: min={nz_col.min():.0f}, max={nz_col.max():.0f}, "
              f"mean={nz_col.mean():.0f}, q25={np.quantile(nz_col,0.25):.0f}, "
              f"q50={np.quantile(nz_col,0.50):.0f}")
        min_plant_dist = int(0.13 / gsd_m)  # 13cm = 25px
        for label, h, p in [
            ("no_thr", None, None),
            ("h=q25", np.quantile(nz_col, 0.25), None),
            ("h=mean", float(nz_col.mean()), None),
            ("h=mean + prom=mean*0.1", float(nz_col.mean()), float(nz_col.mean()*0.10)),
            ("h=mean + prom=mean*0.15", float(nz_col.mean()), float(nz_col.mean()*0.15)),
            ("h=mean*0.5 + prom=mean*0.15", float(nz_col.mean()*0.5), float(nz_col.mean()*0.15)),
            ("h=q25 + prom=mean*0.10", np.quantile(nz_col, 0.25), float(nz_col.mean()*0.10)),
            ("h=q25 + prom=mean*0.20", np.quantile(nz_col, 0.25), float(nz_col.mean()*0.20)),
        ]:
            pp, _ = fp(col_profile_s, distance=min_plant_dist, height=h, prominence=p)
            mean_jugan = (np.diff(pp).mean() * gsd_m * 100) if len(pp) > 1 else float('nan')
            print(f"    {label}: plants={len(pp)}, mean_jugan={mean_jugan:.1f}cm")


if __name__ == "__main__":
    main()
