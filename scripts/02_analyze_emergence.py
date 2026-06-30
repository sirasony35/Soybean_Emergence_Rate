"""
[02] 필지별 콩 입모율 분석 (농진청 흐름 적용)

흐름 (각 타일 단위):
  1) ExG 식생지수 (캐시 활용)
  2) 3배 다운샘플 (~15mm GSD) — 약한 식생 신호 강화
  3) Otsu 단독 이진화 (엄격) + opening + size 필터
  4) 타일별 Radon 두둑각 추정
  5) 글로벌 dominant 각도 SNAP (도로/이상 타일 제외)
  6) 회전 → 가로축 누적 1D 히스토그램
  7) 1차 두둑 검출 (피치 ≥ 50cm)
  8) 2차 각 두둑 안 dual-row (±35cm, 30cm 간격)
  9) 각 줄 따라 1D 카운팅 → 콩 개체
  10) 결주 = 인접 개체 간격 > 1.5 × median(주간)
"""
from __future__ import annotations
from pathlib import Path
import time
import csv
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import Affine
from shapely.geometry import Point, LineString
from skimage.filters import threshold_otsu
from skimage.morphology import opening, remove_small_objects, disk
from scipy.ndimage import rotate as ndi_rotate, gaussian_filter1d
from scipy.signal import find_peaks

from emergence_lib import (
    compute_exg, detrend_local,
    estimate_row_angle, refine_mask_row_aware,
    find_dominant_angles, snap_to_dominant,
    rotated_to_original_coords, pixel_to_geo,
)


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELDS_DIR = ROOT / "result" / "fields"
SHP_DIR = ROOT / "shapefile"
OUT_DIR = ROOT / "result" / "emergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 농학 제원 ===
TILE_SIZE_M           = 15.0   # 타일 크기 (혼합 두둑 방향 자동 처리)
DS_FACTOR             = 3      # 다운샘플 배수 (5.25 → 15.75 mm GSD)

RIDGE_PITCH_MIN_M     = 0.50   # 두둑 최소 간격 (50cm; 좁은 두둑 + 행간 합)
INTRA_RIDGE_BAND_M    = 0.35   # 두둑 안 dual-row 검출 반경 (±35cm)
INTRA_LINE_MIN_M      = 0.15   # 두둑 안 두 줄 간 최소 간격 (15cm)
STD_PLANT_SPACING_M   = 0.20   # 주간 표준
MIN_PLANT_SPACING_M   = 0.13   # 개체 검출 최소 간격
BAND_HALF_M           = 0.10   # 줄 ± 밴드 (개체 카운팅용)
DOMINANT_SNAP_DEG     = 15.0   # 타일 각도 SNAP 허용 오차

EXG_DETREND_SIGMA     = 150


def _smooth(arr: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    return gaussian_filter1d(arr, sigma=sigma)


def detect_ridges_in_profile(profile_s: np.ndarray, gsd_m: float,
                              min_pitch_m: float = RIDGE_PITCH_MIN_M,
                              height_q: float = 0.30,
                              prom_factor: float = 0.10):
    """1단계 두둑 검출 — 큰 주기 피크 (≥50cm 간격).
       반환: ridge_peaks (Y px array)
    """
    nz = profile_s[profile_s > 0]
    if nz.size < 20:
        return np.array([], dtype=int)
    h_thr = float(np.quantile(nz, height_q))
    prom = max(1.0, float(nz.mean()) * prom_factor)
    min_dist_px = max(5, int(min_pitch_m / gsd_m))
    peaks, _ = find_peaks(profile_s, distance=min_dist_px,
                           height=h_thr, prominence=prom)
    return peaks


def detect_dual_rows_per_ridge(profile_s: np.ndarray, ridge_peaks: np.ndarray,
                                gsd_m: float,
                                band_half_m: float = INTRA_RIDGE_BAND_M,
                                intra_min_m: float = INTRA_LINE_MIN_M,
                                prom_factor: float = 0.05,
                                max_lines_per_ridge: int = 2):
    """2단계 각 두둑 안 dual-row 검출.
       반환: list[dict{ridge_idx, line_y_px, line_idx_in_ridge}]
    """
    band_half_px = max(3, int(band_half_m / gsd_m))
    intra_min_px = max(3, int(intra_min_m / gsd_m))
    out = []
    for ridge_idx, rp in enumerate(ridge_peaks):
        y_lo = max(0, rp - band_half_px)
        y_hi = min(len(profile_s), rp + band_half_px + 1)
        sub = profile_s[y_lo:y_hi]
        sub_nz = sub[sub > 0]
        if sub_nz.size < 5:
            continue
        sub_prom = max(1.0, float(sub_nz.mean()) * prom_factor)
        sub_peaks, _ = find_peaks(sub, distance=intra_min_px, prominence=sub_prom)
        if len(sub_peaks) == 0:
            # 폴백: 두둑 중심 자체를 1줄로
            out.append({"ridge_idx": ridge_idx, "line_y_px": int(rp),
                        "line_idx_in_ridge": 0})
            continue
        # 상위 N개 (height 기준)
        if len(sub_peaks) > max_lines_per_ridge:
            heights = sub[sub_peaks]
            top = np.argsort(-heights)[:max_lines_per_ridge]
            sub_peaks = np.sort(sub_peaks[top])
        for i, sp in enumerate(sub_peaks):
            out.append({"ridge_idx": ridge_idx,
                        "line_y_px": int(y_lo + sp),
                        "line_idx_in_ridge": i})
    return out


def count_plants_along_line(rot_bw: np.ndarray, line_y: int, gsd_m: float,
                             band_half_px: int,
                             min_plant_spacing_m: float = MIN_PLANT_SPACING_M,
                             smooth_sigma: float = 1.5,
                             prom_factor: float = 0.10,
                             h_q: float = 0.25):
    """줄 한 개에 대해 가로 누적 1D 프로파일 → find_peaks."""
    H, W = rot_bw.shape
    r0 = max(0, line_y - band_half_px)
    r1 = min(H, line_y + band_half_px + 1)
    band = rot_bw[r0:r1]
    col_profile = band.astype(np.float32).sum(axis=0)
    col_profile_s = gaussian_filter1d(col_profile, sigma=smooth_sigma)
    nz = col_profile_s[col_profile_s > 0]
    if nz.size < 5:
        return col_profile_s, np.array([], dtype=int)
    h_thr = float(np.quantile(nz, h_q))
    prom = max(0.5, float(nz.mean()) * prom_factor)
    min_dist_px = max(3, int(min_plant_spacing_m / gsd_m))
    peaks, _ = find_peaks(col_profile_s, distance=min_dist_px,
                           height=h_thr, prominence=prom)
    return col_profile_s, peaks


def analyze_tile(tile_bw: np.ndarray, gsd_m: float, force_angle: float | None):
    """단일 타일 처리 — 두둑→dual-row→개체."""
    if tile_bw.sum() < 500:
        return None
    angle = float(force_angle) if force_angle is not None else estimate_row_angle(tile_bw)
    rot_cleaned, _ = refine_mask_row_aware(tile_bw, angle, gsd_m,
                                            close_len_m=0.04,
                                            open_radius=1, min_size=20)
    if rot_cleaned.sum() < 300:
        return None
    profile = rot_cleaned.astype(np.float32).sum(axis=1)
    profile_s = _smooth(profile, sigma=4.0)

    # 1차 두둑 검출
    ridge_peaks = detect_ridges_in_profile(profile_s, gsd_m)
    if len(ridge_peaks) < 2:
        return None

    # 2차 두둑 안 dual-row
    line_info = detect_dual_rows_per_ridge(profile_s, ridge_peaks, gsd_m)
    if not line_info:
        return None
    line_ys = np.array(sorted(set(L["line_y_px"] for L in line_info)))
    line_to_ridge = {L["line_y_px"]: L["ridge_idx"] for L in line_info}

    # 3차 줄별 개체 카운팅
    band_half_px = max(3, int(BAND_HALF_M / gsd_m))
    per_row = []
    for ly in line_ys:
        col_prof, plants_x = count_plants_along_line(
            rot_cleaned, int(ly), gsd_m, band_half_px,
        )
        gaps = []
        med_spacing_px = float("nan")
        if len(plants_x) >= 2:
            d = np.diff(plants_x)
            med_spacing_px = float(np.median(d))
            thr = med_spacing_px * 1.5
            for i, dd in enumerate(d):
                if dd > thr:
                    gaps.append((int(plants_x[i]), int(plants_x[i+1]), int(dd)))
        per_row.append({
            "line_y_px": int(ly),
            "ridge_idx": int(line_to_ridge[int(ly)]),
            "plants_x": plants_x,
            "n_plants": int(len(plants_x)),
            "median_spacing_px": med_spacing_px,
            "gaps": gaps,
        })
    return {
        "angle": angle,
        "rot_shape": rot_cleaned.shape,
        "orig_shape": tile_bw.shape,
        "ridge_peaks_y": ridge_peaks,
        "line_ys": line_ys,
        "per_row": per_row,
    }


def estimate_tile_angle_only(tile_bw: np.ndarray, gsd_m: float):
    if tile_bw.sum() < 500:
        return None
    return estimate_row_angle(tile_bw)


def analyze_field(tif_path: Path) -> dict:
    field_name = tif_path.stem
    print(f"\n========== {field_name} ==========")
    t0 = time.time()

    with rasterio.open(tif_path) as src:
        gsd_orig = abs(src.transform.a)
        transform_orig = src.transform
        crs = src.crs
        H_orig, W_orig = src.height, src.width
        print(f"  CRS: {crs}, GSD: {gsd_orig*1000:.3f} mm/px, shape: {H_orig}×{W_orig}")

        # 캐시 로드
        cache_exg = FIELDS_DIR / f"_cache_{field_name}_exg.npy"
        cache_valid = FIELDS_DIR / f"_cache_{field_name}_valid.npy"
        if cache_exg.exists() and cache_valid.exists():
            t1 = time.time()
            exg_full = np.load(cache_exg).astype(np.float32)
            valid_full = np.load(cache_valid).astype(bool)
            print(f"  [cache] ExG+valid 로드 {time.time()-t1:.1f}s")
        else:
            t1 = time.time()
            rgb = src.read(indexes=[1,2,3]).astype(np.float32)
            alpha = src.read(4) if src.count >= 4 else None
            valid_full = (alpha > 0) if alpha is not None else np.ones((H_orig, W_orig), dtype=bool)
            exg_full = compute_exg(rgb)
            exg_full = detrend_local(exg_full, sigma=EXG_DETREND_SIGMA)
            exg_full[~valid_full] = 0
            del rgb
            np.save(cache_exg, exg_full.astype(np.float32))
            np.save(cache_valid, valid_full.astype(np.uint8))
            print(f"  [ExG] {time.time()-t1:.1f}s + cache 저장")

    # === 다운샘플 (DS_FACTOR배) ===
    ds = DS_FACTOR
    gsd = gsd_orig * ds
    exg_ds = exg_full[::ds, ::ds]
    valid_ds = valid_full[::ds, ::ds]
    H, W = exg_ds.shape
    transform = transform_orig * Affine.scale(ds, ds)
    print(f"  [ds] {ds}배 다운샘플 → {gsd*1000:.1f}mm/px, shape: {H}×{W}")

    # === 1차 이진화 (Otsu × 0.6 완화, 약한 새싹 포함) ===
    t1 = time.time()
    otsu_raw = float(threshold_otsu(exg_ds[valid_ds]))
    thr = otsu_raw * 0.6   # 임계 완화
    bw = (exg_ds > thr) & valid_ds
    bw = opening(bw, disk(1))
    bw = remove_small_objects(bw, min_size=4)
    veg_pct = bw.sum() / valid_ds.sum() * 100 if valid_ds.sum() else 0
    print(f"  [bin] {time.time()-t1:.1f}s  Otsu_raw={otsu_raw:.4f}, 사용 임계={thr:.4f}, "
          f"식생 {veg_pct:.2f}%")

    valid_area_m2 = valid_ds.sum() * gsd * gsd

    # === 타일별 처리 ===
    tile_px = max(1, int(TILE_SIZE_M / gsd))
    n_ty = (H + tile_px - 1) // tile_px
    n_tx = (W + tile_px - 1) // tile_px
    print(f"  [tile] 타일 {tile_px}px ({TILE_SIZE_M}m), 격자 {n_ty}×{n_tx}")

    # 1차 패스: 각도 수집
    t1 = time.time()
    angle_records = []  # (ty, tx, y0, x0, y1, x1, angle)
    for ty in range(n_ty):
        for tx in range(n_tx):
            y0 = ty*tile_px; x0 = tx*tile_px
            y1 = min(y0+tile_px, H); x1 = min(x0+tile_px, W)
            if valid_ds[y0:y1, x0:x1].sum() < 0.3 * (y1-y0)*(x1-x0):
                continue
            a = estimate_tile_angle_only(bw[y0:y1, x0:x1], gsd)
            if a is not None:
                angle_records.append((ty, tx, y0, x0, y1, x1, a))
    print(f"  [pass1] {time.time()-t1:.1f}s  타일 각도 수집 {len(angle_records)}")

    angles = [r[-1] for r in angle_records]
    dominant = find_dominant_angles(angles, bin_width=5.0,
                                     min_prominence_ratio=0.25, max_modes=2)
    print(f"           dominant: {dominant}")

    # 2차 패스: 타일별 두둑→dual-row→개체
    t1 = time.time()
    all_plants_geo = []
    all_lines = []
    all_gaps = []
    tile_results = []
    n_used = 0; n_skipped = 0
    total_plants = 0; total_expected = 0

    for ty, tx, y0, x0, y1, x1, orig_a in angle_records:
        snapped = snap_to_dominant(orig_a, dominant, DOMINANT_SNAP_DEG)
        if snapped is None:
            n_skipped += 1
            continue
        tile_bw = bw[y0:y1, x0:x1]
        tile_valid = valid_ds[y0:y1, x0:x1]

        res = analyze_tile(tile_bw, gsd, force_angle=snapped)
        if res is None:
            n_skipped += 1
            continue
        n_used += 1

        angle = res["angle"]
        rot_shape = res["rot_shape"]
        tile_orig_shape = res["orig_shape"]
        per_row = res["per_row"]
        ridge_peaks = res["ridge_peaks_y"]

        # 회전된 valid (라인 유효 길이 계산용)
        rot_valid = ndi_rotate(tile_valid.astype(np.uint8),
                                angle=-angle, reshape=True, order=0).astype(bool)
        min_h = min(rot_valid.shape[0], rot_shape[0])
        min_w = min(rot_valid.shape[1], rot_shape[1])
        rot_valid = rot_valid[:min_h, :min_w]

        # plants 좌표 변환
        plant_rot_yx = []
        for r in per_row:
            for px in r["plants_x"]:
                plant_rot_yx.append([r["line_y_px"], int(px)])
        if plant_rot_yx:
            plant_rot_yx = np.array(plant_rot_yx)
            plant_tile_orig = rotated_to_original_coords(
                plant_rot_yx, angle, tile_orig_shape, rot_shape
            )
            field_y = plant_tile_orig[:, 0] + y0
            field_x = plant_tile_orig[:, 1] + x0
            geos = pixel_to_geo(np.column_stack([field_y, field_x]), transform)
            for g in geos:
                all_plants_geo.append((g[0], g[1]))

        # 라인 + 결주 좌표
        for line_idx, r in enumerate(per_row):
            ly = r["line_y_px"]
            if ly >= rot_valid.shape[0]:
                continue
            rv = rot_valid[ly]
            if not rv.any():
                continue
            vidx = np.where(rv)[0]
            lx0, lx1 = int(vidx.min()), int(vidx.max())
            ep_rot = np.array([[ly, lx0], [ly, lx1]])
            ep_tile = rotated_to_original_coords(ep_rot, angle, tile_orig_shape, rot_shape)
            ep_field = np.column_stack([ep_tile[:,0]+y0, ep_tile[:,1]+x0])
            ep_geo = pixel_to_geo(ep_field, transform)
            valid_len_m = int(rv.sum()) * gsd
            exp_n = max(1, int(round(valid_len_m / STD_PLANT_SPACING_M)))
            emerg = r["n_plants"] / exp_n * 100 if exp_n > 0 else 0
            all_lines.append({
                "tile_y": ty, "tile_x": tx,
                "line_idx_in_tile": line_idx,
                "ridge_idx_in_tile": r["ridge_idx"],
                "angle_deg": angle,
                "n_plants": r["n_plants"],
                "expected_n": exp_n,
                "row_emergence_pct": emerg,
                "valid_len_m": valid_len_m,
                "geometry": LineString(ep_geo.tolist()),
            })
            total_plants += r["n_plants"]
            total_expected += exp_n

            for g0, g1, dd in r["gaps"]:
                ep = np.array([[ly, g0], [ly, g1]])
                ept = rotated_to_original_coords(ep, angle, tile_orig_shape, rot_shape)
                epf = np.column_stack([ept[:,0]+y0, ept[:,1]+x0])
                epg = pixel_to_geo(epf, transform)
                all_gaps.append({
                    "tile_y": ty, "tile_x": tx,
                    "line_idx_in_tile": line_idx,
                    "gap_len_m": dd * gsd,
                    "geometry": LineString(epg.tolist()),
                })

        tile_results.append({
            "tile_y": ty, "tile_x": tx,
            "angle": angle,
            "n_ridges": len(ridge_peaks),
            "n_lines": len(per_row),
            "n_plants": sum(r["n_plants"] for r in per_row),
        })
    print(f"  [pass2] {time.time()-t1:.1f}s  사용 {n_used}/스킵 {n_skipped}")
    print(f"  [agg] 라인 {len(all_lines)}, 개체 {len(all_plants_geo)}, 결주 {len(all_gaps)}")

    field_emerg = total_plants / total_expected * 100 if total_expected else float("nan")
    print(f"  필지 입모율 = {total_plants}/{total_expected} = {field_emerg:.1f}%")

    # === 저장 ===
    t1 = time.time()
    if all_plants_geo:
        plant_gdf = gpd.GeoDataFrame(
            {"x_geo": [p[0] for p in all_plants_geo],
             "y_geo": [p[1] for p in all_plants_geo]},
            geometry=[Point(p) for p in all_plants_geo],
            crs=crs,
        )
        plant_gdf.to_file(OUT_DIR / f"{field_name}_plants.gpkg", driver="GPKG")
    if all_lines:
        gpd.GeoDataFrame(all_lines, crs=crs).to_file(
            OUT_DIR / f"{field_name}_rows.gpkg", driver="GPKG"
        )
    if all_gaps:
        gpd.GeoDataFrame(all_gaps, crs=crs).to_file(
            OUT_DIR / f"{field_name}_gaps.gpkg", driver="GPKG"
        )
    rows_csv = OUT_DIR / f"{field_name}_rows.csv"
    with rows_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tile_y","tile_x","line_idx_in_tile","ridge_idx_in_tile",
                    "angle_deg","valid_len_m","n_plants","expected_n","row_emergence_pct"])
        for L in all_lines:
            w.writerow([L["tile_y"], L["tile_x"], L["line_idx_in_tile"],
                        L["ridge_idx_in_tile"], round(L["angle_deg"], 2),
                        round(L["valid_len_m"], 2),
                        L["n_plants"], L["expected_n"], round(L["row_emergence_pct"], 1)])
    tiles_csv = OUT_DIR / f"{field_name}_tiles.csv"
    with tiles_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tile_y","tile_x","angle_deg","n_ridges","n_lines","n_plants"])
        for r in tile_results:
            w.writerow([r["tile_y"], r["tile_x"], round(r["angle"], 2),
                        r["n_ridges"], r["n_lines"], r["n_plants"]])
    print(f"  [save] {time.time()-t1:.1f}s")

    dt = time.time() - t0
    print(f"  [완료] {dt:.1f}s")

    return {
        "field": field_name,
        "area_ha": valid_area_m2 / 10000,
        "ds_factor": ds,
        "ds_gsd_mm": gsd*1000,
        "otsu_threshold": thr,
        "vegetation_pct": veg_pct,
        "n_tiles_used": n_used,
        "n_tiles_skipped": n_skipped,
        "n_lines": len(all_lines),
        "n_plants": total_plants,
        "expected_plants": total_expected,
        "emergence_pct": field_emerg,
        "n_gaps": len(all_gaps),
        "elapsed_s": dt,
    }


def main():
    tif_paths = sorted(FIELDS_DIR.glob("GJSM-*.tif"))
    print(f"분석할 필지 {len(tif_paths)}개:")
    for p in tif_paths: print(f"  - {p.name}")
    results = []
    t_total = time.time()
    for p in tif_paths:
        try:
            r = analyze_field(p)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ {p.name}: {e}")
    summary = OUT_DIR / "emergence_summary.csv"
    if results:
        with summary.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results: w.writerow(r)
    print(f"\n전체 완료: {(time.time()-t_total)/60:.1f}분")
    print(f"요약: {summary}\n=== 필지별 요약 ===")
    for r in results:
        print(f"  {r['field']:>12s}: {r['area_ha']:.3f}ha  타일 {r['n_tiles_used']:3d}, "
              f"라인 {r['n_lines']:>5,}, 개체 {r['n_plants']:>6,}, "
              f"입모율 {r['emergence_pct']:5.1f}%")


if __name__ == "__main__":
    main()
