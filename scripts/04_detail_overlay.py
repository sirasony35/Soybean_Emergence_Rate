"""
[04] RGB 상세 오버레이 + 조간/주간 CSV 보강

산출:
  (A) result/emergence/{필지}_detail.png — 작은 영역(약 6m × 5m) RGB 위에:
        · 두둑 라인 (노란선)
        · 콩 개체 (빨간 원)
        · 좌상단: "total=N  lines=M  area=A m²"
        · 영역이 두둑과 평행하도록 회전 (라인이 수직 표시)
        · GJSM-1-1은 zone별로 추가 — _detail_위_일반.png, _detail_아래_스마트.png
  (B) emergence_summary.csv 에 조간/주간 명시 컬럼 추가
  (C) GJSM-1-1_seeder_stats.csv 에 zone별 조간/주간 추가
"""

from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box as shp_box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import rotate as ndi_rotate
import sys

sys.path.insert(0, str(Path(__file__).parent))
from emergence_lib import STD_ROW_SPACING_M, STD_PLANT_SPACING_M, load_polygon_zip

ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELDS_DIR = ROOT / "result" / "fields"
EM_DIR = ROOT / "result" / "emergence"
SHP_DIR = ROOT / "shapefile"

DETAIL_WIN_M = (6.0, 5.0)   # (along-row, across-rows) in meters


# ============================================================
#  (A) 상세 오버레이
# ============================================================
def _compute_intervals_along_rows_cm(plants_geos, rows_lines, band_width_m=0.12):
    """plants_geos: (N,2) 지리좌표. rows_lines: list of (p1, p2) tuples (지리좌표).
       각 라인 ±band_width 내 plants를 따라 정렬 후 인접거리(cm) 산출.
       전체 인접거리 1D 배열 반환.
    """
    px = plants_geos[:, 0]; py = plants_geos[:, 1]
    out = []
    for (x0, y0), (x1, y1) in rows_lines:
        dx, dy = x1 - x0, y1 - y0
        L = float(np.hypot(dx, dy))
        if L <= 0:
            continue
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        rel_x = px - x0; rel_y = py - y0
        proj = rel_x * ux + rel_y * uy
        perp = rel_x * nx + rel_y * ny
        mask = (np.abs(perp) <= band_width_m) & (proj >= 0) & (proj <= L)
        if mask.sum() < 2:
            continue
        proj_sorted = np.sort(proj[mask])
        out.append(np.diff(proj_sorted) * 100)
    return np.concatenate(out) if out else np.array([])


def make_detail_overlay(field_name: str, center_xy_geo: tuple[float, float],
                         row_angle_deg: float,
                         out_path: Path,
                         zone_filter=None,
                         title_suffix: str = ""):
    """필지의 특정 위치에서 작은 윈도우를 잘라 회전(라인 수직) + 오버레이 → PNG."""
    tif_path = FIELDS_DIR / f"{field_name}.tif"
    plants_gpkg = EM_DIR / f"{field_name}_plants.gpkg"
    rows_gpkg = EM_DIR / f"{field_name}_rows.gpkg"
    plants_gdf = gpd.read_file(plants_gpkg)
    rows_gdf = gpd.read_file(rows_gpkg)

    with rasterio.open(tif_path) as src:
        gsd_m = abs(src.transform.a)
        transform = src.transform
        crs = src.crs
        # 회전 여유분 포함한 큰 윈도우 (along-row, across-rows의 √2배)
        margin = max(DETAIL_WIN_M) * 1.6
        win_pix = int(margin / gsd_m)
        cx, cy = center_xy_geo
        row_c, col_c = src.index(cx, cy)
        col_off = max(0, col_c - win_pix // 2)
        row_off = max(0, row_c - win_pix // 2)
        window = Window(col_off=col_off, row_off=row_off,
                       width=min(win_pix, src.width - col_off),
                       height=min(win_pix, src.height - row_off))
        data = src.read(window=window)
        win_transform = src.window_transform(window)
        win_bounds = rasterio.windows.bounds(window, src.transform)  # (l,b,r,t)

    rgb = data[:3]
    alpha = data[3] if data.shape[0] >= 4 else np.full(rgb.shape[1:], 255, dtype=np.uint8)

    # 회전: 두둑이 영상의 세로축(Y)이 되도록.
    # 분석에서 `-row_angle_deg` 회전 시 두둑이 가로축이 됨. 세로로 만들려면 +90° 더.
    rot_angle = -row_angle_deg + 90.0
    rgb_rot = np.stack([ndi_rotate(rgb[i], angle=rot_angle, reshape=True, order=1)
                       for i in range(3)], axis=0)
    alpha_rot = ndi_rotate(alpha, angle=rot_angle, reshape=True, order=0)

    rot_H, rot_W = rgb_rot.shape[1:]
    orig_H, orig_W = rgb.shape[1:]

    # 윈도우 내 plants/rows 필터: 윈도우 지리 박스
    win_box = shp_box(win_bounds[0], win_bounds[1], win_bounds[2], win_bounds[3])
    plants_in = plants_gdf[plants_gdf.geometry.within(win_box)]
    if zone_filter is not None and "zone" in plants_in.columns:
        plants_in = plants_in[plants_in.zone == zone_filter]
    # rows: 라인이 박스와 교차하는 것만
    rows_in = rows_gdf[rows_gdf.geometry.intersects(win_box)]

    # 지리 → 윈도우 픽셀(원본 비회전) → 회전 픽셀로 변환
    def geo_to_rot_pix(xs_geo, ys_geo):
        col_orig = (xs_geo - win_transform.c) / win_transform.a
        row_orig = (ys_geo - win_transform.f) / win_transform.e
        A = np.deg2rad(rot_angle)
        cos_A, sin_A = np.cos(A), np.sin(A)
        cy_o = orig_H/2.0; cx_o = orig_W/2.0
        cy_r = rot_H/2.0; cx_r = rot_W/2.0
        dy = row_orig - cy_o; dx = col_orig - cx_o
        y_rot = cos_A * dy - sin_A * dx + cy_r
        x_rot = sin_A * dy + cos_A * dx + cx_r
        return x_rot, y_rot

    # 회전 후 사용자가 원하는 박스(DETAIL_WIN_M)에 맞춰 자르기
    # 회전 영상에서 라인이 수직이므로 along-row=세로(Y), across-rows=가로(X)
    win_along_px = int(DETAIL_WIN_M[0] / gsd_m)
    win_across_px = int(DETAIL_WIN_M[1] / gsd_m)
    cy_r = rot_H // 2; cx_r = rot_W // 2
    y0 = max(0, cy_r - win_along_px // 2); y1 = min(rot_H, y0 + win_along_px)
    x0 = max(0, cx_r - win_across_px // 2); x1 = min(rot_W, x0 + win_across_px)
    rgb_crop = rgb_rot[:, y0:y1, x0:x1]
    crop_disp = np.transpose(rgb_crop.astype(np.uint8), (1, 2, 0))

    # plants 회전 픽셀 → 크롭 픽셀
    px_in_arr = plants_in.geometry.x.values
    py_in_arr = plants_in.geometry.y.values
    px_rot, py_rot = geo_to_rot_pix(px_in_arr, py_in_arr)
    # 크롭 내 필터
    crop_mask = (px_rot >= x0) & (px_rot < x1) & (py_rot >= y0) & (py_rot < y1)
    px_c = px_rot[crop_mask] - x0
    py_c = py_rot[crop_mask] - y0
    n_plants_visible = int(crop_mask.sum())

    # rows 끝점 → 크롭 픽셀, 그리고 크롭 박스와 클립
    line_segs = []
    for _, r in rows_in.iterrows():
        coords = list(r.geometry.coords)
        if len(coords) < 2:
            continue
        (X0g, Y0g), (X1g, Y1g) = coords[0], coords[-1]
        ax_, ay_ = geo_to_rot_pix(np.array([X0g, X1g]), np.array([Y0g, Y1g]))
        # 두 점 모두 크롭 박스 안인지 (혹은 일부) 확인
        seg = ((ax_[0]-x0, ay_[0]-y0), (ax_[1]-x0, ay_[1]-y0))
        line_segs.append(seg)
    # 보이는 라인 수 = 라인이 크롭 박스와 실제 교차하는 수 (Shapely clip)
    from shapely.geometry import LineString, box as shp_box2
    crop_H = y1 - y0; crop_W = x1 - x0
    crop_geom = shp_box2(0, 0, crop_W, crop_H)
    visible_lines = 0
    visible_line_segs = []
    for seg in line_segs:
        ls = LineString(seg)
        clipped = ls.intersection(crop_geom)
        if clipped.is_empty or clipped.length < 10:
            continue
        visible_lines += 1
        visible_line_segs.append(seg)

    # ---- 그리기 ----
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    ax.imshow(crop_disp)
    for (ax0, ay0), (ax1, ay1) in visible_line_segs:
        ax.plot([ax0, ax1], [ay0, ay1], color="#ffd000", lw=1.0, alpha=0.95)
    if n_plants_visible > 0:
        ax.scatter(px_c, py_c, s=80, facecolors="none", edgecolors="#ff2020",
                  lw=1.4, alpha=0.95)
    ax.set_xlim(0, crop_W); ax.set_ylim(crop_H, 0)
    ax.axis("off")
    # 좌상단 텍스트
    txt = f"total = {n_plants_visible}    lines = {visible_lines}\n" \
          f"area = {DETAIL_WIN_M[0]:.1f}m × {DETAIL_WIN_M[1]:.1f}m"
    ax.text(0.015, 0.985, txt, transform=ax.transAxes,
           fontsize=13, fontweight="bold",
           verticalalignment="top", horizontalalignment="left",
           color="white",
           bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.6))
    ttl = f"{field_name} 상세 오버레이"
    if title_suffix:
        ttl += f" — {title_suffix}"
    plt.title(ttl, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → {out_path.name}  (plants={n_plants_visible}, lines={visible_lines})")


# ============================================================
#  (B) 조간/주간 통계 강화
# ============================================================
def compute_zone_stats_with_spacings(field_name: str):
    """각 필지의 조간(인접 두둑선 간), 주간(인접 개체 간) 통계 산출.
       GJSM-1-1은 zone별로도 산출.
       반환: dict {zone_or_field: {조간_평균_cm, 조간_CV, 조간_적합pct, 주간_평균_cm, 주간_CV, 주간_적합pct}}
       범위: 조간 30±7cm, 주간 20±5cm
    """
    plants_gpkg = EM_DIR / f"{field_name}_plants.gpkg"
    rows_gpkg = EM_DIR / f"{field_name}_rows.gpkg"
    plants_gdf = gpd.read_file(plants_gpkg)
    rows_gdf = gpd.read_file(rows_gpkg)

    def _compute_for(p_gdf, r_gdf):
        # 조간: 라인 중심점들의 인접 거리(라인 방향 직교축으로 정렬)
        if len(r_gdf) >= 2:
            # 라인 방향 벡터: 첫 라인 기준
            l0 = r_gdf.geometry.iloc[0]
            (x0, y0), (x1, y1) = l0.coords[0], l0.coords[-1]
            dx = x1 - x0; dy = y1 - y0
            L = float(np.hypot(dx, dy))
            ux, uy = dx/L, dy/L
            nx, ny = -uy, ux
            cent = r_gdf.geometry.centroid
            cx_arr = np.array([c.x for c in cent])
            cy_arr = np.array([c.y for c in cent])
            # normal축 좌표
            perp_coord = cx_arr * nx + cy_arr * ny
            perp_sorted = np.sort(perp_coord)
            row_int = np.diff(perp_sorted) * 100  # cm
            # 너무 큰 간격(영역 경계) 제거: 99 percentile 위는 outlier
            if len(row_int) > 10:
                upper = np.percentile(row_int, 95) * 1.5
                row_int = row_int[row_int <= upper]
        else:
            row_int = np.array([])

        # 주간: 라인 따라 인접 개체 거리
        rows_lines = []
        for _, r in r_gdf.iterrows():
            c = list(r.geometry.coords)
            if len(c) >= 2:
                rows_lines.append((c[0], c[-1]))
        plant_geos = np.column_stack([p_gdf.geometry.x.values, p_gdf.geometry.y.values])
        plant_int = _compute_intervals_along_rows_cm(plant_geos, rows_lines, band_width_m=0.12)

        def stat_block(arr, std_cm, tol_cm):
            if len(arr) == 0:
                return {"평균_cm": np.nan, "CV": np.nan, "적합pct": np.nan, "표본수": 0}
            m = float(np.mean(arr)); s = float(np.std(arr))
            inr = float(np.mean(np.abs(arr - std_cm) <= tol_cm)) * 100
            return {"평균_cm": m, "CV": s/m if m > 0 else np.nan,
                    "적합pct": inr, "표본수": int(arr.size)}

        return {
            "조간": stat_block(row_int, STD_ROW_SPACING_M*100, 7.0),
            "주간": stat_block(plant_int, STD_PLANT_SPACING_M*100, 5.0),
        }

    out = {}
    out["전체"] = _compute_for(plants_gdf, rows_gdf)

    if field_name == "GJSM-1-1":
        # zone 컬럼이 GPKG에 없으면 03 script의 사선 경계 검출 재현
        if "zone" not in plants_gdf.columns:
            import importlib
            mod3 = importlib.import_module("03_postprocess_visualize")
            tif_path = FIELDS_DIR / f"{field_name}.tif"
            # row_angle 가져오기
            with (EM_DIR / "emergence_summary.csv").open(encoding="utf-8-sig") as f:
                row_angle = None
                for r in csv.DictReader(f):
                    if r["field"] == field_name:
                        row_angle = float(r["row_angle_deg"]); break
            det = mod3.detect_slanted_boundary(tif_path, row_angle, search_band=(0.15, 0.75))
            with rasterio.open(tif_path) as src:
                crs = src.crs
            poly = load_polygon_zip(SHP_DIR / f"{field_name}_Boundary.zip", crs).geometry.iloc[0]
            p1, p2 = mod3._slanted_boundary_endpoints(det, poly.bounds)
            (gx1, gy1), (gx2, gy2) = p1, p2
            L_g = float(np.hypot(gx2-gx1, gy2-gy1))
            ux_, uy_ = (gx2-gx1)/L_g, (gy2-gy1)/L_g
            nx_, ny_ = -uy_, ux_
            sign_up_ = 1.0 if ny_ >= 0 else -1.0
            # plants
            sd_p = ((plants_gdf.geometry.x.values - gx1)*nx_ + (plants_gdf.geometry.y.values - gy1)*ny_) * sign_up_
            plants_gdf = plants_gdf.copy()
            plants_gdf["zone"] = np.where(sd_p > 0, "위_일반국내", "아래_스마트")
            # rows
            rc = rows_gdf.geometry.centroid
            sd_r = ((np.array([c.x for c in rc]) - gx1)*nx_ + (np.array([c.y for c in rc]) - gy1)*ny_) * sign_up_
            rows_gdf = rows_gdf.copy()
            rows_gdf["zone"] = np.where(sd_r > 0, "위_일반국내", "아래_스마트")
        for zone in ["위_일반국내", "아래_스마트"]:
            p_z = plants_gdf[plants_gdf.zone == zone]
            r_z = rows_gdf[rows_gdf.zone == zone] if "zone" in rows_gdf.columns else rows_gdf
            out[zone] = _compute_for(p_z, r_z)
    return out


def augment_emergence_summary_csv():
    """emergence_summary.csv 에 조간/주간 한국어 컬럼을 추가하여 in-place 갱신."""
    src = EM_DIR / "emergence_summary.csv"
    if not src.exists():
        print(f"  [skip] {src.name} 없음"); return
    rows = []
    with src.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            field = r["field"]
            stats = compute_zone_stats_with_spacings(field)
            jo = stats["전체"]["조간"]; ju = stats["전체"]["주간"]
            r.update({
                "조간_평균_cm": round(jo["평균_cm"], 2) if jo["평균_cm"] == jo["평균_cm"] else "",
                "조간_CV": round(jo["CV"], 3) if jo["CV"] == jo["CV"] else "",
                "조간_적합pct": round(jo["적합pct"], 1) if jo["적합pct"] == jo["적합pct"] else "",
                "조간_표본수": jo["표본수"],
                "주간_평균_cm": round(ju["평균_cm"], 2) if ju["평균_cm"] == ju["평균_cm"] else "",
                "주간_CV": round(ju["CV"], 3) if ju["CV"] == ju["CV"] else "",
                "주간_적합pct": round(ju["적합pct"], 1) if ju["적합pct"] == ju["적합pct"] else "",
                "주간_표본수": ju["표본수"],
            })
            rows.append(r)
    if not rows:
        return
    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with src.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  → {src.name} 갱신")


def augment_seeder_stats_csv():
    """GJSM-1-1_seeder_stats.csv 에 zone별 조간/주간 추가."""
    src = EM_DIR / "GJSM-1-1_seeder_stats.csv"
    if not src.exists():
        print(f"  [skip] {src.name} 없음"); return
    stats = compute_zone_stats_with_spacings("GJSM-1-1")
    rows = []
    with src.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            zone = r["zone"]
            if zone in stats:
                jo = stats[zone]["조간"]; ju = stats[zone]["주간"]
                r.update({
                    "조간_평균_cm": round(jo["평균_cm"], 2) if jo["평균_cm"] == jo["평균_cm"] else "",
                    "조간_CV": round(jo["CV"], 3) if jo["CV"] == jo["CV"] else "",
                    "조간_적합pct": round(jo["적합pct"], 1) if jo["적합pct"] == jo["적합pct"] else "",
                    "조간_표본수": jo["표본수"],
                    "주간_평균_cm": round(ju["평균_cm"], 2) if ju["평균_cm"] == ju["평균_cm"] else "",
                    "주간_CV": round(ju["CV"], 3) if ju["CV"] == ju["CV"] else "",
                    "주간_적합pct": round(ju["적합pct"], 1) if ju["적합pct"] == ju["적합pct"] else "",
                    "주간_표본수": ju["표본수"],
                })
            rows.append(r)
    # 모든 row의 키 합집합으로 fieldnames 구성
    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with src.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  → {src.name} 갱신")


# ============================================================
#  main
# ============================================================
def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # row_angle_deg per field
    angles = {}
    with (EM_DIR / "emergence_summary.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            angles[r["field"]] = float(r["row_angle_deg"])

    # 각 필지 — 폴리곤 무게중심에서 상세 오버레이
    print("\n===== (A) 상세 RGB 오버레이 =====")
    for field_name, row_angle in angles.items():
        tif_path = FIELDS_DIR / f"{field_name}.tif"
        if not tif_path.exists():
            continue
        # 필지 폴리곤 무게중심
        shp_zip = SHP_DIR / f"{field_name}_Boundary.zip"
        with rasterio.open(tif_path) as src:
            crs = src.crs
        poly_gdf = load_polygon_zip(shp_zip, crs)
        poly = poly_gdf.geometry.iloc[0]
        center = (poly.centroid.x, poly.centroid.y)
        out_png = EM_DIR / f"{field_name}_detail.png"
        try:
            make_detail_overlay(field_name, center, row_angle, out_png,
                               title_suffix="필지 중심부")
        except Exception as e:
            print(f"  ❌ {field_name}: {e}")

    # GJSM-1-1 zone별 상세 — zone은 plants_gpkg에 저장 안 되어 있으므로 03 로직 재현
    print("\n===== (A2) GJSM-1-1 zone별 오버레이 =====")
    plants_gpkg = EM_DIR / "GJSM-1-1_plants.gpkg"
    if plants_gpkg.exists():
        plants_gdf = gpd.read_file(plants_gpkg)
        # 03_postprocess_visualize 의 사선 분할 재계산
        import importlib
        sys.path.insert(0, str(Path(__file__).parent))
        mod3 = importlib.import_module("03_postprocess_visualize")
        tif_path = FIELDS_DIR / "GJSM-1-1.tif"
        row_angle_11 = angles["GJSM-1-1"]
        det = mod3.detect_slanted_boundary(tif_path, row_angle_11, search_band=(0.15, 0.75))
        with rasterio.open(tif_path) as src:
            crs = src.crs
        poly = load_polygon_zip(SHP_DIR / "GJSM-1-1_Boundary.zip", crs).geometry.iloc[0]
        boundary_p1, boundary_p2 = mod3._slanted_boundary_endpoints(det, poly.bounds)
        (gx1, gy1), (gx2, gy2) = boundary_p1, boundary_p2
        L_g = np.hypot(gx2-gx1, gy2-gy1)
        ux_, uy_ = (gx2-gx1)/L_g, (gy2-gy1)/L_g
        nx_, ny_ = -uy_, ux_
        sign_up_ = 1.0 if ny_ >= 0 else -1.0
        pxs = plants_gdf.geometry.x.values; pys = plants_gdf.geometry.y.values
        sd = ((pxs - gx1) * nx_ + (pys - gy1) * ny_) * sign_up_
        plants_gdf["zone"] = np.where(sd > 0, "위_일반국내", "아래_스마트")
        # rows_gdf 도 zone 부여 — make_detail_overlay 안에서 다시 로드되므로 zone_filter 가 작동 안 함
        # 대신 zone별 중심점 계산해서 거기서 상세 보기
        for zone, label in [("위_일반국내", "위_일반국내파종기"),
                            ("아래_스마트", "아래_스마트파종기")]:
            z_plants = plants_gdf[plants_gdf.zone == zone]
            if len(z_plants) == 0:
                continue
            cx = float(z_plants.geometry.x.mean())
            cy = float(z_plants.geometry.y.mean())
            out_png = EM_DIR / f"GJSM-1-1_detail_{label}.png"
            try:
                make_detail_overlay("GJSM-1-1", (cx, cy), row_angle_11, out_png,
                                   title_suffix=zone.replace("_", " "))
            except Exception as e:
                print(f"  ❌ {label}: {e}")

    print("\n===== (B) emergence_summary 조간/주간 컬럼 추가 =====")
    augment_emergence_summary_csv()

    print("\n===== (C) GJSM-1-1_seeder_stats 조간/주간 컬럼 추가 =====")
    augment_seeder_stats_csv()


if __name__ == "__main__":
    main()
