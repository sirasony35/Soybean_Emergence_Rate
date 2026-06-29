"""
[03] 후처리·시각화 강화 + GJSM-1-1 상하 분할 비교

02 단계에서 생성된 plants/rows/gaps GPKG와 rows.csv를 다시 로드해서:
  (A) 각 필지 overview PNG를 PPT용 설명 캡션 + 큰 폰트로 재생성
  (B) GJSM-1-1 은 필지 중심 가로선 기준 상하 분할:
      - 위쪽(북쪽, 큰 Y) = 일반 국내 파종기
      - 아래쪽(남쪽, 작은 Y) = 스마트 파종기 실증
      각 영역별 개체수·평균 주간·결주율·입모율 비교

산출:
  result/emergence/{필지}_overview_ppt.png       — 캡션 강화 PNG
  result/emergence/GJSM-1-1_seeder_compare.png   — 상하 비교 시각화
  result/emergence/GJSM-1-1_seeder_stats.csv     — 상하 비교 통계 CSV
"""

from __future__ import annotations
from pathlib import Path
import csv
import zipfile
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from emergence_lib import STD_ROW_SPACING_M, STD_PLANT_SPACING_M, load_polygon_zip

ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELDS_DIR = ROOT / "result" / "fields"
EM_DIR = ROOT / "result" / "emergence"
SHP_DIR = ROOT / "shapefile"


# =============================================================
#  (A) PPT용 overview PNG 재생성
# =============================================================

PANEL_EXPLANATIONS = {
    "rgb": (
        "[1] 원본 RGB 정사영상\n"
        "드론 촬영(GSD 5.25mm/px) → 필지 SHP 경계로 클립.\n"
        "두둑 위에 파종된 콩 새싹이 점 형태로 분포."
    ),
    "mask_rows": (
        "[2] 식생 마스크 + 두둑 라인 검출\n"
        "ExG(2g−r−b) 색지수 → Otsu+Sauvola 이진화로 식생 픽셀 추출 →\n"
        "Radon 변환으로 두둑 각도 자동 검출 → 두둑 방향 closing 정리.\n"
        "빨간 가로선이 검출된 두둑 중심선."
    ),
    "row_emergence": (
        "[3] 두둑별 입모율 (%)\n"
        "각 두둑별로 (검출된 개체수 / 기대 개체수) ×100.\n"
        "기대 개체수 = 두둑 유효 길이 / 표준 주간 20cm.\n"
        "빨간 점선이 필지 전체 평균. 막대가 평균보다 낮으면 결주·미파종 두둑."
    ),
    "plant_dist": (
        "[4] 주간 분포 히스토그램\n"
        "검출된 인접 개체 간 거리 분포(cm).\n"
        "빨간 점선이 표준 주간 20cm. 분홍 영역이 적합 범위(±5cm).\n"
        "분포가 표준 근처에 모이면 정상 파종, 우측 꼬리(40cm+)는 결주를 의미."
    ),
}


def downsample_rgb(rgb_path: Path, target_long: int = 2200):
    """RGBA TIF에서 RGB 다운샘플."""
    with rasterio.open(rgb_path) as src:
        long_side = max(src.width, src.height)
        step = max(1, long_side // target_long)
        rgb = src.read(out_shape=(src.count, src.height // step, src.width // step))
        gsd_m = abs(src.transform.a) * step
        # downsample된 transform 계산
        new_transform = src.transform * src.transform.scale(step, step)
        crs = src.crs
        bounds = src.bounds
    rgb_disp = np.transpose(rgb[:3], (1, 2, 0))
    if rgb.shape[0] >= 4:
        alpha = rgb[3]
        rgb_disp = rgb_disp.copy()
        rgb_disp[alpha == 0] = 0
    return rgb_disp, new_transform, crs, bounds, gsd_m


def compute_per_row_intervals_cm(plants_gdf: gpd.GeoDataFrame,
                                  rows_gdf: gpd.GeoDataFrame,
                                  band_width_m: float = 0.12) -> np.ndarray:
    """각 row 라인에 ±band_width_m 내 plants를 모아 라인 따라 정렬 후 인접거리(cm) 산출.
       반환: (총인접쌍수,) cm 배열.
    """
    if len(plants_gdf) == 0 or len(rows_gdf) == 0:
        return np.array([])
    px = plants_gdf.geometry.x.values
    py = plants_gdf.geometry.y.values
    all_intervals = []
    for _, r in rows_gdf.iterrows():
        coords = list(r.geometry.coords)
        if len(coords) < 2:
            continue
        (x0, y0), (x1, y1) = coords[0], coords[-1]
        # 라인 방향 단위벡터, 라인 길이
        dx, dy = x1 - x0, y1 - y0
        L = float(np.hypot(dx, dy))
        if L <= 0:
            continue
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux  # 법선
        rel_x = px - x0; rel_y = py - y0
        proj = rel_x * ux + rel_y * uy
        perp = rel_x * nx + rel_y * ny
        mask = (np.abs(perp) <= band_width_m) & (proj >= 0) & (proj <= L)
        if mask.sum() < 2:
            continue
        proj_sorted = np.sort(proj[mask])
        intervals = np.diff(proj_sorted)
        all_intervals.append(intervals)
    if not all_intervals:
        return np.array([])
    return np.concatenate(all_intervals) * 100  # m → cm


def plot_overview_ppt(field_name: str):
    """단일 필지의 PPT 친화적 overview PNG 재생성."""
    tif_path = FIELDS_DIR / f"{field_name}.tif"
    plants_gpkg = EM_DIR / f"{field_name}_plants.gpkg"
    rows_gpkg = EM_DIR / f"{field_name}_rows.gpkg"
    gaps_gpkg = EM_DIR / f"{field_name}_gaps.gpkg"
    rows_csv = EM_DIR / f"{field_name}_rows.csv"

    if not all(p.exists() for p in [tif_path, plants_gpkg, rows_gpkg, rows_csv]):
        print(f"[skip] {field_name}: 입력 파일 없음")
        return None

    print(f"\n== {field_name} PPT overview 생성 ==")
    rgb_disp, _, crs, bounds, gsd_m = downsample_rgb(tif_path)

    plants_gdf = gpd.read_file(plants_gpkg)
    rows_gdf = gpd.read_file(rows_gpkg)

    # rows.csv 로드 — 라인/두둑 별도 카운트
    row_emergence = []
    n_plants_list = []
    expected_list = []
    ridge_idx_list = []
    with rows_csv.open(encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row_emergence.append(float(row["row_emergence_pct"]))
                n_plants_list.append(int(row["n_plants"]))
                expected_list.append(int(row["expected_n"]))
                ridge_idx_list.append(int(row.get("ridge_idx", -1)))
            except (ValueError, KeyError):
                pass

    n_lines = len(row_emergence)
    n_ridges = len(set(r for r in ridge_idx_list if r >= 0)) if ridge_idx_list else 0
    total_plants = sum(n_plants_list)
    total_expected = sum(expected_list)
    field_emergence = total_plants / total_expected * 100 if total_expected > 0 else np.nan

    # 폴리곤 실제 면적
    field_area_m2 = (bounds.right - bounds.left) * (bounds.top - bounds.bottom)
    shp_zip = SHP_DIR / f"{field_name}_Boundary.zip"
    if shp_zip.exists():
        poly_gdf = load_polygon_zip(shp_zip, crs)
        field_area_m2 = float(poly_gdf.geometry.iloc[0].area)

    # 라인 따라 plant 간 거리(cm) 산출 — 진짜 주간 분포
    intervals_cm = compute_per_row_intervals_cm(plants_gdf, rows_gdf, band_width_m=0.12)
    print(f"  인접 거리 표본: {len(intervals_cm):,}쌍")

    # ---- 그림 (필지 가로:세로 비율에 따라 패널 비율 조정) ----
    aspect = (bounds.right - bounds.left) / (bounds.top - bounds.bottom)
    # 위쪽 두 패널은 영상이라 영상 비율을 따르고, 아래쪽 두 패널은 그래프(가로 넓게)
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.5, 2.5],
        hspace=0.42, wspace=0.15,
        left=0.04, right=0.97, top=0.92, bottom=0.05,
    )

    H_px, W_px = rgb_disp.shape[:2]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_disp)
    ax1.axis("off")
    ax1.set_title("(1) 원본 RGB 정사영상", fontsize=15, fontweight="bold")
    ax1.text(0.02, -0.06, PANEL_EXPLANATIONS["rgb"], transform=ax1.transAxes,
            fontsize=10.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc", alpha=0.95))

    # 패널 2: 라인 오버레이 (RGB 위에) — 두둑(ridge)별 색 구분
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_disp)
    # ridge_idx에 따라 교차 색상 (홀수=노랑, 짝수=빨강) — 두둑 페어 구분 시각화
    line_palette = ["#ff2020", "#ffd000"]   # ridge별 alternating
    for _, row in rows_gdf.iterrows():
        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue
        ridx = int(row.get("ridge_idx", 0)) if "ridge_idx" in rows_gdf.columns else 0
        color = line_palette[ridx % 2] if ridx >= 0 else "red"
        (x0, y0), (x1, y1) = coords[0], coords[-1]
        col0 = (x0 - bounds.left) / (bounds.right - bounds.left) * W_px
        col1 = (x1 - bounds.left) / (bounds.right - bounds.left) * W_px
        row0 = (bounds.top - y0) / (bounds.top - bounds.bottom) * H_px
        row1 = (bounds.top - y1) / (bounds.top - bounds.bottom) * H_px
        ax2.plot([col0, col1], [row0, row1], color=color, lw=0.35, alpha=0.7)
    ax2.axis("off")
    ax2.set_title(f"(2) 파종 라인 검출  (두둑 {n_ridges}개 / 라인 {n_lines}줄, dual-row)",
                  fontsize=15, fontweight="bold")
    ax2.text(0.02, -0.06, PANEL_EXPLANATIONS["mask_rows"], transform=ax2.transAxes,
            fontsize=10.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc", alpha=0.95))

    # 패널 3: 두둑별 입모율
    ax3 = fig.add_subplot(gs[1, 0])
    em_arr = np.array(row_emergence)
    mean_em = float(em_arr.mean())
    colors = ["#d33" if e < 70 else "#dc3" if e < 90 else "#3a3" for e in em_arr]
    ax3.bar(range(len(em_arr)), em_arr, color=colors, width=1.0)
    ax3.axhline(mean_em, color="black", ls="--", lw=1.5,
               label=f"필지 평균 {mean_em:.1f}%")
    ax3.axhline(100, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax3.set_xlabel("두둑 index (회전영상 기준)")
    ax3.set_ylabel("입모율 (%)")
    ax3.set_title(f"(3) 두둑별 입모율  (필지 평균 {mean_em:.1f}%)",
                 fontsize=15, fontweight="bold")
    ax3.set_ylim(0, max(140, em_arr.max() * 1.05))
    ax3.legend(loc="upper right")
    # 범례 색
    legend2 = [mpatches.Patch(color="#3a3", label="≥90% (정상)"),
              mpatches.Patch(color="#dc3", label="70–90% (주의)"),
              mpatches.Patch(color="#d33", label="<70% (저조)")]
    ax3.legend(handles=legend2 + [plt.Line2D([0], [0], color="black", ls="--",
                                            label=f"평균 {mean_em:.1f}%")],
              loc="upper right", fontsize=9)
    ax3.text(0.02, -0.30, PANEL_EXPLANATIONS["row_emergence"], transform=ax3.transAxes,
            fontsize=10.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc", alpha=0.95))

    # 패널 4: 주간 분포 (라인 따라 모든 인접 거리)
    ax4 = fig.add_subplot(gs[1, 1])
    if len(intervals_cm) > 0:
        bins = np.arange(0, 65, 1.0)
        ax4.hist(intervals_cm, bins=bins, color="#3a3", alpha=0.75, edgecolor="white")
        ax4.axvline(20, color="red", ls="--", lw=1.5, label="표준 20cm")
        ax4.axvspan(15, 25, color="red", alpha=0.10, label="적합 범위 ±5cm")
        mean_sp = float(np.mean(intervals_cm))
        med_sp = float(np.median(intervals_cm))
        in_range = float(np.mean((intervals_cm >= 15) & (intervals_cm <= 25))) * 100
        ax4.axvline(mean_sp, color="blue", ls=":", lw=1.5, label=f"실측 평균 {mean_sp:.1f}cm")
        ax4.set_xlabel("개체 간 주간 (cm)")
        ax4.set_ylabel(f"빈도 (총 {len(intervals_cm):,}쌍)")
        ax4.set_title(f"(4) 주간 분포  (평균 {mean_sp:.1f}cm, 중앙값 {med_sp:.1f}cm, "
                     f"적합 {in_range:.1f}%)", fontsize=15, fontweight="bold")
        ax4.set_xlim(0, 60)
        ax4.legend(loc="upper right", fontsize=9)
    ax4.text(0.02, -0.30, PANEL_EXPLANATIONS["plant_dist"], transform=ax4.transAxes,
            fontsize=10.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc", alpha=0.95))

    # 전체 헤더
    title = (f"{field_name} 콩 입모율 분석 결과\n"
            f"면적 {field_area_m2/10000:.3f} ha  ·  두둑 {n_ridges}개 ({n_lines}줄)  ·  "
            f"개체 {total_plants:,}개  ·  입모율 {field_emergence:.1f}%")
    fig.suptitle(title, fontsize=18, fontweight="bold")

    out_png = EM_DIR / f"{field_name}_overview_ppt.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → {out_png.name}")
    return {
        "field": field_name,
        "area_ha": field_area_m2 / 10000,
        "n_lines": n_lines,
        "n_ridges": n_ridges,
        "n_plants": total_plants,
        "emergence_pct": field_emergence,
    }


# =============================================================
#  (B) GJSM-1-1 상하 분할 비교 (위 = 일반 국내 파종기, 아래 = 스마트 파종기)
# =============================================================

def detect_slanted_boundary(tif_path: Path, row_angle_deg: float,
                            search_band: tuple[float, float] = (0.2, 0.8)):
    """두둑 방향과 평행한 사선 경계 검출.
       전제: 필지 안 두둑 사이에 두둑 방향과 평행한 경계 통로(미파종)가 있다.

       방법:
         1) 다운샘플 후 ExG 마스크 + valid 마스크 생성
         2) 두둑이 가로축이 되도록 회전 (angle = -row_angle_deg)
         3) 회전 프레임에서 가로축 합산 → 1D Y 프로파일 (식생 밀도)
         4) 가장 깊은 골(valley)이 경계 위치 (rot Y 좌표)

       반환:
         boundary_y_rot      회전영상 픽셀 Y
         original_shape      (H, W) 다운샘플 원본
         rotated_shape       (H_rot, W_rot)
         transform           다운샘플 원본 transform
         density_s           스무딩된 1D 밀도 프로파일 (회전 프레임)
         band_y0, band_y1    탐색 범위
    """
    with rasterio.open(tif_path) as src:
        step = max(1, max(src.width, src.height) // 1500)
        rgb = src.read(out_shape=(src.count, src.height // step, src.width // step))
        transform = src.transform * src.transform.scale(step, step)
    H_orig, W_orig = rgb.shape[1:]

    if rgb.shape[0] < 3:
        return None
    rgb_f = rgb[:3].astype(np.float32)
    s = rgb_f.sum(axis=0) + 1e-6
    g = rgb_f[1] / s; r = rgb_f[0] / s; b = rgb_f[2] / s
    exg = 2.0*g - r - b
    valid = (rgb[3] if rgb.shape[0] >= 4 else (s > 30)) > 0
    veg = (exg > 0.05) & valid

    # 두둑이 가로축이 되도록 회전
    from scipy.ndimage import rotate as ndi_rotate
    from scipy.ndimage import gaussian_filter1d as g1d
    veg_rot = ndi_rotate(veg.astype(np.uint8), angle=-row_angle_deg, reshape=True, order=0).astype(bool)
    valid_rot = ndi_rotate(valid.astype(np.uint8), angle=-row_angle_deg, reshape=True, order=0).astype(bool)
    H_rot, W_rot = veg_rot.shape

    width_per_row = valid_rot.sum(axis=1).astype(np.float32)
    veg_per_row = veg_rot.sum(axis=1).astype(np.float32)
    density = np.where(width_per_row > 50, veg_per_row / np.maximum(width_per_row, 1), 1.0)
    density_s = g1d(density, sigma=8)

    valid_y = np.where(width_per_row > 50)[0]
    if len(valid_y) == 0:
        return None
    y0_rot, y1_rot = int(valid_y[0]), int(valid_y[-1])
    band_y0 = int(y0_rot + (y1_rot - y0_rot) * search_band[0])
    band_y1 = int(y0_rot + (y1_rot - y0_rot) * search_band[1])

    sub = density_s[band_y0:band_y1].copy()
    if sub.size == 0:
        return None
    boundary_y_rot = int(band_y0 + np.argmin(sub))
    print(f"  사선 경계: rot Y={boundary_y_rot} (탐색 {band_y0}~{band_y1}, "
          f"density={density_s[boundary_y_rot]:.3f})")
    return {
        "boundary_y_rot": boundary_y_rot,
        "original_shape": (H_orig, W_orig),
        "rotated_shape": (H_rot, W_rot),
        "transform": transform,
        "density_s": density_s,
        "band_y0": band_y0,
        "band_y1": band_y1,
        "row_angle_deg": row_angle_deg,
    }


def split_rows_by_slanted_boundary(rows_gdf, transform, original_shape, rotated_shape,
                                   row_angle_deg, boundary_y_rot):
    """rows_gdf 라인스트링의 중심점을 사선 경계로 분류."""
    centroids = rows_gdf.geometry.centroid
    H_orig, W_orig = original_shape; H_rot, W_rot = rotated_shape
    cx_geo = np.asarray([c.x for c in centroids])
    cy_geo = np.asarray([c.y for c in centroids])
    col = (cx_geo - transform.c) / transform.a
    row = (cy_geo - transform.f) / transform.e
    A = np.deg2rad(-row_angle_deg)
    cos_A, sin_A = np.cos(A), np.sin(A)
    cy_o = H_orig/2.0; cx_o = W_orig/2.0
    cy_r = H_rot/2.0; cx_r = W_rot/2.0
    dy = row - cy_o; dx = col - cx_o
    y_rot = cos_A * dy - sin_A * dx + cy_r
    return y_rot < boundary_y_rot


def split_plants_by_slanted_boundary(plants_gdf, transform, original_shape, rotated_shape,
                                     row_angle_deg, boundary_y_rot):
    """각 plant의 지리좌표 → 다운샘플 픽셀 → 회전 픽셀 → boundary_y_rot 와 비교.
       반환: bool 배열 (True=위쪽=일반국내, False=아래쪽=스마트)
    """
    H_orig, W_orig = original_shape
    H_rot, W_rot = rotated_shape

    plants_x_geo = plants_gdf.geometry.x.values
    plants_y_geo = plants_gdf.geometry.y.values
    # geo → 다운샘플 원본 pixel (행=Y감소, 열=X증가)
    plants_col = (plants_x_geo - transform.c) / transform.a
    plants_row = (plants_y_geo - transform.f) / transform.e

    # 원본 픽셀 → 회전 픽셀 (ndi_rotate(angle=-row_angle_deg) 의 forward)
    A = np.deg2rad(-row_angle_deg)
    cos_A, sin_A = np.cos(A), np.sin(A)
    cy_orig = H_orig / 2.0; cx_orig = W_orig / 2.0
    cy_rot = H_rot / 2.0; cx_rot = W_rot / 2.0
    dy = plants_row - cy_orig
    dx = plants_col - cx_orig
    plants_y_rot = cos_A * dy - sin_A * dx + cy_rot
    # plants_x_rot 도 필요할 일 있을 때만
    return plants_y_rot < boundary_y_rot   # True = 위쪽 = 일반국내


def compare_seeders_GJSM_1_1():
    field_name = "GJSM-1-1"
    print(f"\n== {field_name} 상하 분할 비교 (위=일반국내, 아래=스마트) ==")

    tif_path = FIELDS_DIR / f"{field_name}.tif"
    plants_gpkg = EM_DIR / f"{field_name}_plants.gpkg"
    rows_gpkg = EM_DIR / f"{field_name}_rows.gpkg"

    if not (plants_gpkg.exists() and rows_gpkg.exists()):
        print(f"[skip] {field_name}: GPKG 없음")
        return

    rgb_disp, _, crs, bounds, gsd_m = downsample_rgb(tif_path)
    plants_gdf = gpd.read_file(plants_gpkg)
    rows_gdf = gpd.read_file(rows_gpkg)

    # 02_analyze_emergence.py 출력에서 두둑각 가져오기
    import csv as _csv
    row_angle_deg = None
    with (EM_DIR / "emergence_summary.csv").open(encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            if row["field"] == field_name:
                row_angle_deg = float(row["row_angle_deg"])
                break
    if row_angle_deg is None:
        print("  [fallback] emergence_summary.csv 에서 row_angle_deg 못 찾음 — -22.5° 가정")
        row_angle_deg = -22.5
    print(f"  두둑 각도: {row_angle_deg:.2f}°")

    # 사선 경계 자동 검출 (두둑 방향과 평행)
    det = detect_slanted_boundary(tif_path, row_angle_deg, search_band=(0.15, 0.75))
    shp_zip = SHP_DIR / f"{field_name}_Boundary.zip"
    poly_gdf = load_polygon_zip(shp_zip, crs)
    poly = poly_gdf.geometry.iloc[0]
    if det is None:
        print(f"  [fallback] 자동 검출 실패 — 무게중심에서 수평 분할")
        cy = poly.centroid.y
        plants_gdf["zone"] = np.where(plants_gdf.geometry.y >= cy, "위_일반국내", "아래_스마트")
        rows_gdf["zone"] = np.where(rows_gdf.geometry.centroid.y >= cy, "위_일반국내", "아래_스마트")
        boundary_geo_p1 = (poly.bounds[0], cy); boundary_geo_p2 = (poly.bounds[2], cy)
    else:
        # 사선 경계선의 두 끝점 지리좌표
        boundary_geo_p1, boundary_geo_p2 = _slanted_boundary_endpoints(det, poly.bounds)
        # 지리좌표 normal로 분할 (큰 Y쪽 = 위 = 일반국내)
        (gx1, gy1), (gx2, gy2) = boundary_geo_p1, boundary_geo_p2
        L_g = np.hypot(gx2-gx1, gy2-gy1)
        ux_, uy_ = (gx2-gx1)/L_g, (gy2-gy1)/L_g
        nx_, ny_ = -uy_, ux_
        sign_up_ = 1.0 if ny_ >= 0 else -1.0
        # plants
        pxs = plants_gdf.geometry.x.values
        pys = plants_gdf.geometry.y.values
        sd_plants = ((pxs - gx1) * nx_ + (pys - gy1) * ny_) * sign_up_
        plants_gdf["zone"] = np.where(sd_plants > 0, "위_일반국내", "아래_스마트")
        # rows (라인 중심 기준)
        row_cent = rows_gdf.geometry.centroid
        rxs = np.asarray([c.x for c in row_cent])
        rys = np.asarray([c.y for c in row_cent])
        sd_rows = ((rxs - gx1) * nx_ + (rys - gy1) * ny_) * sign_up_
        rows_gdf["zone"] = np.where(sd_rows > 0, "위_일반국내", "아래_스마트")

    # 각 zone별 통계
    stats = {}
    for zone in ["위_일반국내", "아래_스마트"]:
        z_plants = plants_gdf[plants_gdf.zone == zone]
        z_rows = rows_gdf[rows_gdf.zone == zone]

        n_plants = len(z_plants)
        n_rows_z = len(z_rows)
        total_valid_len = float(z_rows["valid_len_m"].sum()) if n_rows_z else 0
        expected = total_valid_len / STD_PLANT_SPACING_M if total_valid_len > 0 else 0
        emergence = n_plants / expected * 100 if expected > 0 else np.nan
        row_emergence_mean = float(z_rows["row_emergence_pct"].mean()) if n_rows_z else np.nan
        # 사선 경계로 폴리곤을 분할해 zone 면적 산출
        zone_poly = _split_polygon_by_slanted_line(poly, boundary_geo_p1, boundary_geo_p2, zone)
        zone_area_m2 = float(zone_poly.area) if zone_poly is not None else 0.0

        stats[zone] = {
            "n_rows": n_rows_z,
            "total_row_length_m": total_valid_len,
            "n_plants": n_plants,
            "expected_plants": int(expected),
            "emergence_pct_area": emergence,
            "row_emergence_mean_pct": row_emergence_mean,
            "zone_area_ha": zone_area_m2 / 10000,
            "plants_per_ha": n_plants / (zone_area_m2/10000) if zone_area_m2 > 0 else np.nan,
        }
        print(f"  {zone}: 라인 {n_rows_z}줄, 개체 {n_plants}개, "
              f"입모율 (면적기반) {emergence:.1f}%, "
              f"라인평균 입모율 {row_emergence_mean:.1f}%")

    # CSV 저장
    out_csv = EM_DIR / f"{field_name}_seeder_stats.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        keys = list(next(iter(stats.values())).keys())
        w = csv.writer(f)
        w.writerow(["zone"] + keys)
        for zone, s in stats.items():
            w.writerow([zone] + [s[k] for k in keys])
    print(f"  통계 CSV: {out_csv.name}")

    # ---- 시각화 ----
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2],
                         left=0.04, right=0.97, top=0.92, bottom=0.06,
                         wspace=0.22, hspace=0.32)

    # 상단 — RGB + 사선 분할선 + 영역 색조 오버레이 (반투명)
    ax_main = fig.add_subplot(gs[0, :2])
    ax_main.imshow(rgb_disp)
    H_px, W_px = rgb_disp.shape[:2]

    # 사선 경계의 두 끝점을 디스플레이 픽셀 좌표로
    (gx1, gy1), (gx2, gy2) = boundary_geo_p1, boundary_geo_p2
    px1 = (gx1 - bounds.left) / (bounds.right - bounds.left) * W_px
    py1 = (bounds.top - gy1) / (bounds.top - bounds.bottom) * H_px
    px2 = (gx2 - bounds.left) / (bounds.right - bounds.left) * W_px
    py2 = (bounds.top - gy2) / (bounds.top - bounds.bottom) * H_px

    # 사선 방향 + normal (지리좌표 기준)
    dx_geo = gx2 - gx1; dy_geo = gy2 - gy1
    L_geo = np.hypot(dx_geo, dy_geo)
    ux, uy = dx_geo/L_geo, dy_geo/L_geo
    nx, ny = -uy, ux

    # 정의: 위 = 큰 Y(북쪽). 그래서 normal 방향 부호를 ny 부호에 맞춤.
    # 부호 sign_up: +normal 방향이 북쪽일 때 +1, 남쪽일 때 -1
    sign_up = 1.0 if ny >= 0 else -1.0

    # 각 픽셀에 대한 위/아래 분류 (지리좌표 부호로 직접 비교)
    yy, xx = np.mgrid[0:H_px, 0:W_px]
    geo_x = bounds.left + (xx + 0.5) / W_px * (bounds.right - bounds.left)
    geo_y = bounds.top - (yy + 0.5) / H_px * (bounds.top - bounds.bottom)
    sign_map = ((geo_x - gx1) * nx + (geo_y - gy1) * ny) * sign_up
    # sign_map > 0 = 위쪽(북) = 일반국내, < 0 = 아래쪽(남) = 스마트

    overlay = np.zeros((H_px, W_px, 4), dtype=np.float32)
    overlay[sign_map > 0, :3] = (0.0, 0.7, 0.9)
    overlay[sign_map > 0, 3] = 0.18
    overlay[sign_map < 0, :3] = (0.9, 0.0, 0.7)
    overlay[sign_map < 0, 3] = 0.18
    ax_main.imshow(overlay)
    ax_main.plot([px1, px2], [py1, py2], color="yellow", lw=3, ls="--",
                label="분할 사선 (두둑 방향과 평행)")

    # 영역 라벨 위치 — 위쪽은 +normal*sign_up, 아래쪽은 그 반대
    mask_up = plants_gdf.zone.values == "위_일반국내"
    mask_dn = ~mask_up
    offset = L_geo * 0.18
    cx_up = (gx1 + gx2)/2 + nx * offset * sign_up
    cy_up = (gy1 + gy2)/2 + ny * offset * sign_up
    cx_dn = (gx1 + gx2)/2 - nx * offset * sign_up
    cy_dn = (gy1 + gy2)/2 - ny * offset * sign_up
    px_up = (cx_up - bounds.left) / (bounds.right - bounds.left) * W_px
    py_up = (bounds.top - cy_up) / (bounds.top - bounds.bottom) * H_px
    px_dn = (cx_dn - bounds.left) / (bounds.right - bounds.left) * W_px
    py_dn = (bounds.top - cy_dn) / (bounds.top - bounds.bottom) * H_px

    ax_main.text(px_up, py_up,
                f"위 (일반 국내 파종기)\n개체 {mask_up.sum():,}개",
                ha="center", va="center", fontsize=18, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#0072A8", alpha=0.85))
    ax_main.text(px_dn, py_dn,
                f"아래 (스마트 파종기 실증)\n개체 {mask_dn.sum():,}개",
                ha="center", va="center", fontsize=18, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#A8007D", alpha=0.85))
    ax_main.set_title("(1) GJSM-1-1 분할 — 위: 일반 국내 파종기 / 아래: 스마트 파종기 실증\n"
                     f"(분할선은 두둑 방향 {row_angle_deg:.1f}°와 평행한 사선)",
                     fontsize=15, fontweight="bold")
    ax_main.set_xlim(0, W_px); ax_main.set_ylim(H_px, 0)
    ax_main.axis("off")

    # 상단 우측 — 비교 막대 그래프 (입모율)
    ax_bar = fig.add_subplot(gs[0, 2])
    zones_kr = ["위\n(일반국내)", "아래\n(스마트)"]
    em_vals = [stats["위_일반국내"]["emergence_pct_area"],
              stats["아래_스마트"]["emergence_pct_area"]]
    bars = ax_bar.bar(zones_kr, em_vals, color=["#5c9", "#c59"], width=0.55,
                     edgecolor="black")
    for b, v in zip(bars, em_vals):
        ax_bar.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%",
                   ha="center", fontsize=14, fontweight="bold")
    ax_bar.set_ylabel("입모율 (%)")
    ax_bar.set_title("(2) 파종기별 면적기반 입모율 비교", fontsize=15, fontweight="bold")
    ax_bar.set_ylim(0, max(em_vals)*1.20 if em_vals[0] else 100)
    ax_bar.grid(axis="y", alpha=0.3)

    # 하단 — 비교 표 (텍스트)
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    diff_em = em_vals[1] - em_vals[0]
    table_text = [
        ["지표", "위 (일반 국내 파종기)", "아래 (스마트 파종기 실증)", "차이 (스마트-일반)"],
        ["면적 (ha)",
         f"{stats['위_일반국내']['zone_area_ha']:.3f}",
         f"{stats['아래_스마트']['zone_area_ha']:.3f}",
         f"{stats['아래_스마트']['zone_area_ha']-stats['위_일반국내']['zone_area_ha']:+.3f}"],
        ["검출 라인 수",
         f"{stats['위_일반국내']['n_rows']:,}",
         f"{stats['아래_스마트']['n_rows']:,}",
         f"{stats['아래_스마트']['n_rows']-stats['위_일반국내']['n_rows']:+,}"],
        ["라인 총 길이 (m)",
         f"{stats['위_일반국내']['total_row_length_m']:.1f}",
         f"{stats['아래_스마트']['total_row_length_m']:.1f}",
         f"{stats['아래_스마트']['total_row_length_m']-stats['위_일반국내']['total_row_length_m']:+.1f}"],
        ["검출 개체 수",
         f"{stats['위_일반국내']['n_plants']:,}",
         f"{stats['아래_스마트']['n_plants']:,}",
         f"{stats['아래_스마트']['n_plants']-stats['위_일반국내']['n_plants']:+,}"],
        ["기대 개체 수 (20cm 기준)",
         f"{stats['위_일반국내']['expected_plants']:,}",
         f"{stats['아래_스마트']['expected_plants']:,}",
         f"{stats['아래_스마트']['expected_plants']-stats['위_일반국내']['expected_plants']:+,}"],
        ["면적기반 입모율 (%)",
         f"{stats['위_일반국내']['emergence_pct_area']:.1f}",
         f"{stats['아래_스마트']['emergence_pct_area']:.1f}",
         f"{diff_em:+.1f}%p"],
        ["두둑 평균 입모율 (%)",
         f"{stats['위_일반국내']['row_emergence_mean_pct']:.1f}",
         f"{stats['아래_스마트']['row_emergence_mean_pct']:.1f}",
         f"{stats['아래_스마트']['row_emergence_mean_pct']-stats['위_일반국내']['row_emergence_mean_pct']:+.1f}%p"],
        ["ha당 개체수",
         f"{stats['위_일반국내']['plants_per_ha']:.0f}",
         f"{stats['아래_스마트']['plants_per_ha']:.0f}",
         f"{stats['아래_스마트']['plants_per_ha']-stats['위_일반국내']['plants_per_ha']:+.0f}"],
    ]
    tbl = ax_table.table(cellText=table_text, loc="center", cellLoc="center",
                         colWidths=[0.25, 0.25, 0.25, 0.25])
    tbl.auto_set_font_size(False); tbl.set_fontsize(12)
    tbl.scale(1, 1.8)
    for j in range(4):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", weight="bold")
    for i in range(1, len(table_text)):
        if i == 6:  # 면적기반 입모율 강조
            for j in range(4):
                tbl[(i, j)].set_facecolor("#fff3cd")
                tbl[(i, j)].set_text_props(weight="bold")

    fig.suptitle("GJSM-1-1 파종기 비교 — 일반 국내 vs 스마트 파종기 실증",
                fontsize=18, fontweight="bold", y=0.98)

    out_png = EM_DIR / f"{field_name}_seeder_compare.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  비교 시각화: {out_png.name}")


def _slanted_boundary_endpoints(det, poly_bounds):
    """det(detect_slanted_boundary 결과)로부터 사선 경계의 양 끝점 지리좌표 계산.
       회전영상에서 y=boundary_y_rot, x=0 ~ W_rot 인 두 점을 원본 픽셀 → 지리좌표로 변환.
    """
    H_orig, W_orig = det["original_shape"]
    H_rot, W_rot = det["rotated_shape"]
    A = np.deg2rad(-det["row_angle_deg"])
    cos_A, sin_A = np.cos(A), np.sin(A)
    cy_rot = H_rot/2.0; cx_rot = W_rot/2.0
    cy_o = H_orig/2.0; cx_o = W_orig/2.0
    by = det["boundary_y_rot"]
    # 회전영상에서 두 끝점 (회전→원본 역변환)
    pts_geo = []
    for bx in [0, W_rot - 1]:
        y_rel = by - cy_rot; x_rel = bx - cx_rot
        # 역회전: cos_A*y_rel + sin_A*x_rel = orig_y_rel ; -sin_A*y_rel + cos_A*x_rel = orig_x_rel
        oy = cos_A * y_rel + sin_A * x_rel + cy_o
        ox = -sin_A * y_rel + cos_A * x_rel + cx_o
        # 다운샘플 원본 픽셀 → 지리좌표
        t = det["transform"]
        gx = t.a * (ox + 0.5) + t.b * (oy + 0.5) + t.c
        gy = t.d * (ox + 0.5) + t.e * (oy + 0.5) + t.f
        pts_geo.append((gx, gy))
    return pts_geo[0], pts_geo[1]


def _split_polygon_by_slanted_line(poly, p1, p2, zone):
    """폴리곤을 사선(p1→p2 통과 무한선)으로 분할해 zone에 해당하는 부분 반환.
       정의: 위_일반국내 = 사선보다 UTM Y(북쪽) 큰 쪽,  아래_스마트 = 작은 쪽.
       normal n=(-uy, ux); n의 y성분 부호(=ux=(x2-x1)/L)로 +normal이 북쪽인지 판정.
    """
    from shapely.geometry import Polygon
    (x1, y1), (x2, y2) = p1, p2
    dx = x2 - x1; dy = y2 - y1
    ld = float(np.hypot(dx, dy))
    if ld == 0:
        return poly
    ux, uy = dx/ld, dy/ld
    nx, ny = -uy, ux                 # 단위 normal
    sign_up = 1.0 if ny >= 0 else -1.0   # +1: +normal=북, -1: -normal=북

    minx, miny, maxx, maxy = poly.bounds
    L = max(maxx - minx, maxy - miny) * 5
    # +normal 방향 반평면 폴리곤
    half_plane_pos = Polygon([
        (x1 + nx*L - ux*L, y1 + ny*L - uy*L),
        (x2 + nx*L + ux*L, y2 + ny*L + uy*L),
        (x2 + ux*L,         y2 + uy*L),
        (x1 - ux*L,         y1 - uy*L),
    ])
    pos_part = poly.intersection(half_plane_pos)
    # 위(북) = sign_up>0 이면 pos_part, 아니면 그 반대
    if zone == "위_일반국내":
        return pos_part if sign_up > 0 else poly.difference(pos_part)
    else:
        return poly.difference(pos_part) if sign_up > 0 else pos_part


# =============================================================
#  main
# =============================================================
def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    field_names = sorted(p.stem for p in FIELDS_DIR.glob("GJSM-*.tif"))
    print(f"PPT overview 재생성 대상: {field_names}\n")
    for fname in field_names:
        plot_overview_ppt(fname)

    # GJSM-1-1 파종기 비교
    if (EM_DIR / "GJSM-1-1_plants.gpkg").exists():
        compare_seeders_GJSM_1_1()


if __name__ == "__main__":
    main()
