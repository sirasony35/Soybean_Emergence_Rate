"""
[02] 필지별 콩 입모율 분석

각 필지 TIF에 대해:
  1) RGB 로드, alpha 마스크로 유효 영역 분리
  2) ExG + detrend → 1차 이진화 → Radon 두둑각
  3) 두둑방향 closing 정리 → 두둑 라인 검출
  4) 두둑별 1D 피크로 개체 검출 → 결주 구간 추출
  5) 회전→원본→지리 좌표 역변환
  6) 산출:
     - {필지}_plants.gpkg    개체 포인트
     - {필지}_rows.gpkg      두둑 중심선 (라인스트링)
     - {필지}_gaps.gpkg      결주 구간 (라인스트링)
     - {필지}_overview.png   시각화 (다운샘플)
     - {필지}_rows.csv       두둑별 통계
  7) 모든 필지 통합 요약: emergence_summary.csv
"""

from __future__ import annotations
from pathlib import Path
import time
import csv
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt

from emergence_lib import (
    STD_PLANT_SPACING_M, STD_INTRA_RIDGE_LINE_CM, STD_INTER_RIDGE_CM,
    INTRA_RIDGE_MAX_CM, MIN_LINE_SPACING_M, MIN_PLANT_SPACING_M,
    ROW_TOLERANCE_CM, PLANT_TOLERANCE_CM, INTER_RIDGE_TOL_CM,
    compute_exg, detrend_local,
    binarize_vegetation_raw,
    analyze_tile, cluster_lines_to_ridges, compute_line_spacings_split,
    evaluate_spacing_quality,
    rotated_to_original_coords, pixel_to_geo,
    estimate_tile_angle_only, find_dominant_angles, snap_to_dominant,
)

TILE_SIZE_M = 15.0   # 타일 크기 (m). 30cm 행간 가정 시 ~50 라인 들어옴

ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELDS_DIR = ROOT / "result" / "fields"
OUT_DIR = ROOT / "result" / "emergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 두둑 라인 길이로 결주 비율을 계산할 때 한 두둑의 유효 픽셀 수가 이만큼은 되어야 함.
MIN_ROW_VALID_PX = 100

# 시각화 다운샘플 크기 (긴 변)
VIS_LONG_SIDE_PX = 2200


def downsample_for_vis(img: np.ndarray, target_long: int) -> tuple[np.ndarray, int]:
    """대용량 영상 다운샘플(블록 평균). 반환: (다운영상, 스텝)."""
    long_side = max(img.shape[-2:])
    step = max(1, long_side // target_long)
    if img.ndim == 3:
        return img[:, ::step, ::step], step
    return img[::step, ::step], step


def _load_cached_result_DEPRECATED(field_name: str) -> dict | None:
    """이미 분석된 필지의 산출물이 있으면 캐시 결과 dict 반환, 없으면 None."""
    needed = [
        OUT_DIR / f"{field_name}_plants.gpkg",
        OUT_DIR / f"{field_name}_rows.gpkg",
        OUT_DIR / f"{field_name}_gaps.gpkg",
        OUT_DIR / f"{field_name}_rows.csv",
    ]
    if not all(p.exists() for p in needed):
        return None
    # CSV에서 통계 복원
    try:
        rows_csv = OUT_DIR / f"{field_name}_rows.csv"
        import csv as _csv
        rows = []
        with rows_csv.open(encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                rows.append(row)
        if not rows:
            return None
        n_rows = len(rows)
        total_plants = sum(int(r["n_plants"]) for r in rows)
        total_expected = sum(int(r["expected_n"]) for r in rows)
        # 추정 입모율
        emergence_pct = total_plants / total_expected * 100 if total_expected > 0 else 0
        total_gaps = sum(int(r["n_gaps"]) for r in rows)
        # 면적: TIF에서 다시 계산
        tif_path = FIELDS_DIR / f"{field_name}.tif"
        with rasterio.open(tif_path) as src:
            gsd_m = abs(src.transform.a)
            alpha = src.read(4) if src.count >= 4 else None
        if alpha is not None:
            area_ha = (alpha > 0).sum() * gsd_m ** 2 / 10000
        else:
            area_ha = 0.0
        # 주간/행간 stats — 캐시에 없으므로 GPKG 다시 읽어 산출
        plants_gdf = gpd.read_file(OUT_DIR / f"{field_name}_plants.gpkg")
        rows_gdf = gpd.read_file(OUT_DIR / f"{field_name}_rows.gpkg")
        # 행간 — 라인 중심점들 인접 거리
        if len(rows_gdf) >= 2:
            l0 = rows_gdf.geometry.iloc[0]
            (x0, y0), (x1, y1) = l0.coords[0], l0.coords[-1]
            dx, dy = x1 - x0, y1 - y0
            L = np.hypot(dx, dy)
            ux, uy = dx/L, dy/L
            nx, ny = -uy, ux
            cent = rows_gdf.geometry.centroid
            cx_arr = np.array([c.x for c in cent])
            cy_arr = np.array([c.y for c in cent])
            perp = cx_arr * nx + cy_arr * ny
            d = np.diff(np.sort(perp))
            if len(d) > 10:
                d = d[d <= np.percentile(d, 95) * 1.5]
            row_spacing_mean_cm = float(d.mean()) * 100 if len(d) else np.nan
            row_in_range = float(np.mean(np.abs(d*100 - 30) <= 7)) * 100 if len(d) else np.nan
        else:
            row_spacing_mean_cm = np.nan; row_in_range = np.nan
        # 주간 — 라인 따라 인접거리
        all_int = []
        for _, r in rows_gdf.iterrows():
            coords = list(r.geometry.coords)
            if len(coords) < 2: continue
            (X0, Y0), (X1, Y1) = coords[0], coords[-1]
            dx, dy = X1-X0, Y1-Y0; L = np.hypot(dx, dy)
            if L == 0: continue
            ux, uy = dx/L, dy/L; nx, ny = -uy, ux
            rel_x = plants_gdf.geometry.x.values - X0
            rel_y = plants_gdf.geometry.y.values - Y0
            proj = rel_x*ux + rel_y*uy
            perp = rel_x*nx + rel_y*ny
            mask = (np.abs(perp) <= 0.12) & (proj >= 0) & (proj <= L)
            if mask.sum() < 2: continue
            all_int.append(np.diff(np.sort(proj[mask])))
        if all_int:
            arr = np.concatenate(all_int) * 100  # cm
            plant_mean = float(arr.mean())
            plant_cv = float(arr.std()/arr.mean()) if arr.mean() > 0 else np.nan
            plant_in_range = float(np.mean(np.abs(arr - 20) <= 5)) * 100
        else:
            plant_mean = np.nan; plant_cv = np.nan; plant_in_range = np.nan

        # row_angle — rows_gdf의 라인 기울기에서 직접 산출 (정확)
        if len(rows_gdf) > 0:
            angs = []
            for _, r in rows_gdf.iterrows():
                c = list(r.geometry.coords)
                if len(c) < 2: continue
                dx_a = c[-1][0] - c[0][0]; dy_a = c[-1][1] - c[0][1]
                ang = np.degrees(np.arctan2(dy_a, dx_a))
                # -90~90 정규화
                ang = (ang + 90) % 180 - 90
                angs.append(ang)
            row_angle = float(np.median(angs)) if angs else 0.0
        else:
            row_angle = 0.0

        return {
            "field": field_name,
            "area_ha": area_ha,
            "row_angle_deg": row_angle,
            "n_rows": n_rows,
            "row_spacing_mean_cm": row_spacing_mean_cm,
            "row_spacing_in_range_pct": row_in_range,
            "n_plants": total_plants,
            "expected_plants": total_expected,
            "emergence_pct": emergence_pct,
            "plant_spacing_mean_cm": plant_mean,
            "plant_spacing_cv": plant_cv,
            "plant_spacing_in_range_pct": plant_in_range,
            "n_gaps": total_gaps,
            "elapsed_s": 0.0,
        }
    except Exception as e:
        print(f"  [캐시 복원 실패] {field_name}: {e}")
        return None


def analyze_field(tif_path: Path) -> dict:
    field_name = tif_path.stem
    print(f"\n========== {field_name} ==========")
    t0 = time.time()

    with rasterio.open(tif_path) as src:
        gsd_m = abs(src.transform.a)
        transform = src.transform
        crs = src.crs
        print(f"  CRS: {crs}, GSD: {gsd_m*1000:.3f} mm/px")
        print(f"  shape: {src.height} × {src.width}")
        data = src.read()  # (4, H, W)
        rgb = data[:3].astype(np.float32)
        alpha = data[3] if data.shape[0] >= 4 else np.full(rgb.shape[1:], 255, dtype=np.uint8)
        del data
    print(f"  [load] {time.time()-t0:.1f}s,  RGB shape={rgb.shape}")

    valid_mask = alpha > 0
    H, W = valid_mask.shape
    valid_area_m2 = valid_mask.sum() * gsd_m ** 2

    # ExG/이진화 결과 캐시 (반복 테스트 가속)
    cache_bw = FIELDS_DIR / f"_cache_{field_name}_bw.npy"
    cache_valid = FIELDS_DIR / f"_cache_{field_name}_valid.npy"
    if cache_bw.exists() and cache_valid.exists():
        t1 = time.time()
        bw_raw = np.load(cache_bw).astype(bool)
        valid_mask = np.load(cache_valid).astype(bool)
        print(f"  [cache] bw + valid 로드 {time.time()-t1:.1f}s")
    else:
        # 1) ExG + detrend
        t1 = time.time()
        exg = compute_exg(rgb)
        exg = detrend_local(exg, sigma=150)
        exg[~valid_mask] = 0
        print(f"  [ExG]  {time.time()-t1:.1f}s")

        # 2) 1차 이진화
        t1 = time.time()
        bw_raw, otsu_t = binarize_vegetation_raw(exg, valid_mask)
        raw_pct = bw_raw.sum() / valid_mask.sum() * 100
        print(f"  [bin]  {time.time()-t1:.1f}s,  Otsu={otsu_t:.4f}, 식생(노이즈포함)={raw_pct:.2f}%")
        # 캐시 저장
        np.save(cache_bw, bw_raw.astype(np.uint8))
        np.save(cache_valid, valid_mask.astype(np.uint8))
        print(f"  [cache] 저장 → {cache_bw.name}")
    del rgb, alpha

    # 3) 타일별 분석 (혼합 두둑 방향 자동 처리)
    t1 = time.time()
    tile_px = int(TILE_SIZE_M / gsd_m)
    n_tiles_y = (H + tile_px - 1) // tile_px
    n_tiles_x = (W + tile_px - 1) // tile_px
    print(f"  [tile] 타일 크기 {TILE_SIZE_M}m({tile_px}px), 격자 {n_tiles_y}×{n_tiles_x}")

    # 3a) 1차 패스 — 모든 타일에서 각도만 빠르게 수집
    tile_angles = []  # (ty, tx, y0, x0, y1, x1, angle)
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0 = ty * tile_px; x0 = tx * tile_px
            y1 = min(y0 + tile_px, H); x1 = min(x0 + tile_px, W)
            tile_valid = valid_mask[y0:y1, x0:x1]
            if tile_valid.sum() < 0.3 * tile_valid.size:
                continue
            tile_bw = bw_raw[y0:y1, x0:x1]
            ang, veg = estimate_tile_angle_only(tile_bw, gsd_m)
            if ang is not None:
                tile_angles.append((ty, tx, y0, x0, y1, x1, ang))
    print(f"  [pass1] 타일 각도 수집: {len(tile_angles)}개")

    # 3b) 글로벌 dominant 각도 모드 검출
    angles_only = [a[-1] for a in tile_angles]
    dominant = find_dominant_angles(angles_only, bin_width=5.0,
                                     min_prominence_ratio=0.25, max_modes=2)
    print(f"  [pass1] dominant 각도 모드: {dominant}")

    # 3c) 2차 패스 — 각 타일 SNAP된 각도로 처리
    all_plants_geo = []
    all_lines = []
    all_gaps = []
    tile_results = []
    n_tiles_used = 0
    n_tiles_skipped = 0
    total_plants = 0
    total_expected = 0

    for ty, tx, y0, x0, y1, x1, orig_ang in tile_angles:
        snapped = snap_to_dominant(orig_ang, dominant, max_dist_deg=15.0)
        if snapped is None:
            n_tiles_skipped += 1
            continue
        tile_bw = bw_raw[y0:y1, x0:x1]
        tile_valid = valid_mask[y0:y1, x0:x1]

        res = analyze_tile(tile_bw, gsd_m,
                           close_len_m=0.06,
                           min_line_spacing_m=MIN_LINE_SPACING_M,
                           band_half_m=0.12,
                           min_plant_spacing_m=MIN_PLANT_SPACING_M,
                           force_angle=snapped)
        if res is None:
            n_tiles_skipped += 1
            continue

        n_tiles_used += 1
        angle = res["angle"]
        rot_shape = res["rot_shape"]
        tile_orig_shape = res["orig_shape"]
        peaks_y = res["peaks_y"]
        per_row = res["per_row"]

        # 타일 내 line→ridge 클러스터
        ridges_in_tile = cluster_lines_to_ridges(peaks_y, gsd_m,
                                                  intra_max_cm=INTRA_RIDGE_MAX_CM)
        line_to_ridge_tile = {}
        for ridx, lines_in_ridge in enumerate(ridges_in_tile):
            for ry_in in lines_in_ridge:
                line_to_ridge_tile[ry_in] = ridx

        # plants 회전→타일원본→필지절대→geo
        plant_rot_yx = []
        for r in per_row:
            for px in r["plants_x"]:
                plant_rot_yx.append([r["row_pix"], int(px)])
        if plant_rot_yx:
            plant_rot_yx = np.array(plant_rot_yx)
            plant_tile_orig = rotated_to_original_coords(
                plant_rot_yx, angle, tile_orig_shape, rot_shape
            )
            plant_field_y = plant_tile_orig[:, 0] + y0
            plant_field_x = plant_tile_orig[:, 1] + x0
            field_pixels = np.column_stack([plant_field_y, plant_field_x])
            plant_geos = pixel_to_geo(field_pixels, transform)
            for g in plant_geos:
                all_plants_geo.append((g[0], g[1]))

        # 라인 끝점 (회전된 valid 마스크에서 가로 유효 범위)
        from scipy.ndimage import rotate as ndi_rotate
        rot_valid_tile = ndi_rotate(tile_valid.astype(np.uint8),
                                     angle=-angle, reshape=True, order=0).astype(bool)
        min_h = min(rot_valid_tile.shape[0], rot_shape[0])
        min_w = min(rot_valid_tile.shape[1], rot_shape[1])
        rot_valid_tile = rot_valid_tile[:min_h, :min_w]

        for line_idx, r in enumerate(per_row):
            ry = r["row_pix"]
            if ry >= rot_valid_tile.shape[0]:
                continue
            rv = rot_valid_tile[ry]
            if not rv.any():
                continue
            vidx = np.where(rv)[0]
            lx0, lx1 = int(vidx.min()), int(vidx.max())
            endpoints_rot = np.array([[ry, lx0], [ry, lx1]])
            endpoints_tile_orig = rotated_to_original_coords(
                endpoints_rot, angle, tile_orig_shape, rot_shape
            )
            endpoints_field = np.column_stack([
                endpoints_tile_orig[:, 0] + y0,
                endpoints_tile_orig[:, 1] + x0,
            ])
            endpoints_geo = pixel_to_geo(endpoints_field, transform)
            line_valid_len_px = int(rv.sum())
            line_valid_len_m = line_valid_len_px * gsd_m
            line_expected = max(1, int(round(line_valid_len_m / STD_PLANT_SPACING_M)))
            line_emergence = r["n_plants"] / line_expected * 100 if line_expected > 0 else 0
            all_lines.append({
                "tile_y": ty, "tile_x": tx,
                "line_idx_in_tile": line_idx,
                "ridge_idx_in_tile": line_to_ridge_tile.get(ry, -1),
                "angle_deg": angle,
                "n_plants": r["n_plants"],
                "expected_n": line_expected,
                "row_emergence_pct": line_emergence,
                "valid_len_m": line_valid_len_m,
                "geometry": LineString(endpoints_geo.tolist()),
            })
            total_plants += r["n_plants"]
            total_expected += line_expected

            # 결주 구간
            for (g0, g1, dd) in r["gaps"]:
                gap_rot = np.array([[ry, g0], [ry, g1]])
                gap_tile_orig = rotated_to_original_coords(
                    gap_rot, angle, tile_orig_shape, rot_shape
                )
                gap_field = np.column_stack([
                    gap_tile_orig[:, 0] + y0,
                    gap_tile_orig[:, 1] + x0,
                ])
                gap_geo = pixel_to_geo(gap_field, transform)
                all_gaps.append({
                    "tile_y": ty, "tile_x": tx,
                    "line_idx_in_tile": line_idx,
                    "gap_len_m": dd * gsd_m,
                    "geometry": LineString(gap_geo.tolist()),
                })

        tile_results.append({
            "tile_y": ty, "tile_x": tx,
            "angle": angle,
            "n_lines": len(peaks_y),
            "n_ridges": len(ridges_in_tile),
            "n_plants": sum(r["n_plants"] for r in per_row),
        })

    print(f"  [tile] {time.time()-t1:.1f}s,  사용 {n_tiles_used}타일 / 스킵 {n_tiles_skipped}")
    print(f"  [agg]  라인 {len(all_lines)}, 개체 {len(all_plants_geo)}, 결주 {len(all_gaps)}")

    field_emergence = total_plants / total_expected * 100 if total_expected > 0 else np.nan
    print(f"  필지 입모율 = {total_plants}/{total_expected} = {field_emergence:.1f}%")

    # 4) 통계 계산
    # 조간/행간: 각 타일별 인접 라인 거리 모음
    # (타일 회전 좌표에서 peaks_y 차이 → cm)
    all_intra = []; all_inter = []
    for tr in tile_results:
        # tr에는 peaks가 없으니 — all_lines에서 같은 (tile_y, tile_x) 의 라인들 골라
        same_tile_lines = [L for L in all_lines if L["tile_y"]==tr["tile_y"] and L["tile_x"]==tr["tile_x"]]
        if len(same_tile_lines) < 2:
            continue
        # 라인 중심 perpendicular 좌표로 정렬
        # 각도 angle을 사용해 normal 방향 계산
        ang_rad = np.deg2rad(tr["angle"])
        # 라인 방향 = (cos, sin)
        nx, ny = -np.sin(ang_rad), np.cos(ang_rad)
        # 라인 중심점 사용
        centers = []
        for L in same_tile_lines:
            cx = (L["geometry"].coords[0][0] + L["geometry"].coords[-1][0]) / 2
            cy = (L["geometry"].coords[0][1] + L["geometry"].coords[-1][1]) / 2
            perp = cx * nx + cy * ny
            centers.append(perp)
        centers = np.sort(np.array(centers))
        gaps_cm = np.diff(centers) * 100
        for g in gaps_cm:
            if g < INTRA_RIDGE_MAX_CM:
                all_intra.append(g)
            else:
                all_inter.append(g)
    all_intra = np.array(all_intra); all_inter = np.array(all_inter)

    if len(all_intra):
        intra_q = {"mean_cm": float(all_intra.mean()),
                   "cv": float(all_intra.std()/max(0.1, all_intra.mean())),
                   "in_range_ratio": float(np.mean(np.abs(all_intra - STD_INTRA_RIDGE_LINE_CM) <= ROW_TOLERANCE_CM))}
    else:
        intra_q = {"mean_cm": np.nan, "cv": np.nan, "in_range_ratio": np.nan}
    if len(all_inter):
        inter_q = {"mean_cm": float(all_inter.mean()),
                   "cv": float(all_inter.std()/max(0.1, all_inter.mean())),
                   "in_range_ratio": float(np.mean(np.abs(all_inter - STD_INTER_RIDGE_CM) <= INTER_RIDGE_TOL_CM))}
    else:
        inter_q = {"mean_cm": np.nan, "cv": np.nan, "in_range_ratio": np.nan}

    print(f"  조간(intra-ridge): 평균 {intra_q['mean_cm']:.1f}cm (n={len(all_intra)})")
    print(f"  행간(inter-ridge): 평균 {inter_q['mean_cm']:.1f}cm (n={len(all_inter)})")

    # 주간: 모든 plants를 line별 정렬해 인접거리
    # all_lines와 all_plants_geo로 재계산 — 라인 따라 plant 모음 → 정렬 → diff
    jugan_intervals_cm = []
    plants_arr = np.array(all_plants_geo)  # (N, 2) X,Y
    for L in all_lines:
        coords = list(L["geometry"].coords)
        if len(coords) < 2:
            continue
        (X0, Y0), (X1, Y1) = coords[0], coords[-1]
        dX, dY = X1 - X0, Y1 - Y0
        L_geo = np.hypot(dX, dY)
        if L_geo == 0:
            continue
        ux, uy = dX/L_geo, dY/L_geo
        nx, ny = -uy, ux
        rel_x = plants_arr[:, 0] - X0
        rel_y = plants_arr[:, 1] - Y0
        proj = rel_x*ux + rel_y*uy
        perp = rel_x*nx + rel_y*ny
        m = (np.abs(perp) <= 0.12) & (proj >= 0) & (proj <= L_geo)
        if m.sum() >= 2:
            jugan_intervals_cm.extend(np.diff(np.sort(proj[m])).tolist())
    jugan_intervals_cm = np.array(jugan_intervals_cm) * 100
    if len(jugan_intervals_cm):
        plant_q = {"mean_cm": float(jugan_intervals_cm.mean()),
                   "cv": float(jugan_intervals_cm.std()/max(0.1, jugan_intervals_cm.mean())),
                   "in_range_ratio": float(np.mean(np.abs(jugan_intervals_cm - STD_PLANT_SPACING_M*100) <= PLANT_TOLERANCE_CM))}
    else:
        plant_q = {"mean_cm": np.nan, "cv": np.nan, "in_range_ratio": np.nan}
    print(f"  주간(jugan): 평균 {plant_q['mean_cm']:.1f}cm (n={len(jugan_intervals_cm)})")

    # 5) GPKG 저장
    t1 = time.time()
    if all_plants_geo:
        plant_gdf = gpd.GeoDataFrame(
            {"x_geo": [g[0] for g in all_plants_geo],
             "y_geo": [g[1] for g in all_plants_geo]},
            geometry=[Point(g) for g in all_plants_geo],
            crs=crs,
        )
        plant_gdf.to_file(OUT_DIR / f"{field_name}_plants.gpkg", driver="GPKG")

    if all_lines:
        rows_gdf = gpd.GeoDataFrame(all_lines, crs=crs)
        rows_gdf.to_file(OUT_DIR / f"{field_name}_rows.gpkg", driver="GPKG")

    if all_gaps:
        gaps_gdf = gpd.GeoDataFrame(all_gaps, crs=crs)
        gaps_gdf.to_file(OUT_DIR / f"{field_name}_gaps.gpkg", driver="GPKG")
    print(f"  [save GPKG]{time.time()-t1:.1f}s")

    # 6) 라인별 CSV
    rows_csv = OUT_DIR / f"{field_name}_rows.csv"
    with rows_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tile_y", "tile_x", "line_idx_in_tile", "ridge_idx_in_tile",
                    "angle_deg", "valid_len_m", "n_plants", "expected_n",
                    "row_emergence_pct"])
        for L in all_lines:
            w.writerow([L["tile_y"], L["tile_x"], L["line_idx_in_tile"],
                        L["ridge_idx_in_tile"], round(L["angle_deg"], 2),
                        round(L["valid_len_m"], 2),
                        L["n_plants"], L["expected_n"],
                        round(L["row_emergence_pct"], 1)])

    # 타일 메타 CSV
    tiles_csv = OUT_DIR / f"{field_name}_tiles.csv"
    with tiles_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tile_y", "tile_x", "angle_deg", "n_lines", "n_ridges", "n_plants"])
        for tr in tile_results:
            w.writerow([tr["tile_y"], tr["tile_x"], round(tr["angle"], 2),
                        tr["n_lines"], tr["n_ridges"], tr["n_plants"]])

    dt = time.time() - t0
    print(f"  [완료]  {dt:.1f}s")

    return {
        "field": field_name,
        "area_ha": valid_area_m2 / 10000,
        "n_tiles_used": n_tiles_used,
        "n_lines": len(all_lines),
        "n_plants": total_plants,
        "expected_plants": total_expected,
        "emergence_pct": field_emergence,
        "jogan_mean_cm": intra_q["mean_cm"],
        "jogan_cv": intra_q["cv"],
        "jogan_in_range_pct": intra_q["in_range_ratio"] * 100 if intra_q["in_range_ratio"] == intra_q["in_range_ratio"] else np.nan,
        "haengan_mean_cm": inter_q["mean_cm"],
        "haengan_cv": inter_q["cv"],
        "haengan_in_range_pct": inter_q["in_range_ratio"] * 100 if inter_q["in_range_ratio"] == inter_q["in_range_ratio"] else np.nan,
        "jugan_mean_cm": plant_q["mean_cm"],
        "jugan_cv": plant_q["cv"],
        "jugan_in_range_pct": plant_q["in_range_ratio"] * 100 if plant_q["in_range_ratio"] == plant_q["in_range_ratio"] else np.nan,
        "n_gaps": len(all_gaps),
        "elapsed_s": dt,
    }


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    tif_paths = sorted(FIELDS_DIR.glob("GJSM-*.tif"))
    if not tif_paths:
        print(f"필지 TIF가 없습니다: {FIELDS_DIR}")
        return
    print(f"분석할 필지 {len(tif_paths)}개:")
    for p in tif_paths:
        print(f"  - {p.name}")

    results = []
    t_total = time.time()
    for p in tif_paths:
        try:
            r = analyze_field(p)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ {p.name} 실패: {e}")

    # 통합 요약 CSV
    summary_csv = OUT_DIR / "emergence_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results:
                w.writerow(r)
    print(f"\n전체 완료: {time.time()-t_total:.1f}s")
    print(f"요약: {summary_csv}")
    print("\n=== 필지별 요약 ===")
    for r in results:
        jg = r.get('jogan_mean_cm', float('nan'))
        hg = r.get('haengan_mean_cm', float('nan'))
        ju = r.get('jugan_mean_cm', float('nan'))
        print(f"  {r['field']:>12s}: 타일 {r.get('n_tiles_used',0):3d} / 라인 {r['n_lines']:4d}, "
              f"개체 {r['n_plants']:>6d}/{r['expected_plants']:>6d}, "
              f"입모율 {r['emergence_pct']:5.1f}%, "
              f"조간 {jg:4.1f}cm / 행간 {hg:4.1f}cm / 주간 {ju:4.1f}cm")


if __name__ == "__main__":
    main()
