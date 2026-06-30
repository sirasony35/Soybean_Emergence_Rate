"""
콩 입모율(Soybean Emergence Rate) — 소규모 검증 스크립트

목적: 한 필지(GJSM-1-1)의 작은 서브 윈도우에서 아래 파이프라인을 시각화 검증.
  1) SHP 경계 재투영(4326 -> 32652) 후 윈도우 단위 크롭
  2) RGB -> ExG(2G - R - B) + 조명 보정(local detrend)
  3) Otsu + adaptive 합성 이진화로 식생 마스크
  4) Radon transform으로 두둑(파종 라인) 방향 자동 추정
  5) 두둑 직교방향 ExG 프로파일 -> 두둑 중심선 좌표 추출
  6) 각 두둑을 따라 1D ExG 프로파일 -> find_peaks 로 개체 카운팅
  7) 결주 = 라인 위 피크 간격 > 1.5 * median(주간) 인 구간

출력: result/verify/ 에 시각화 PNG + 통계 텍스트
"""

from __future__ import annotations

from pathlib import Path
import zipfile
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
import geopandas as gpd
from shapely.geometry import Point
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.transform import radon
from skimage.morphology import remove_small_objects, opening, closing as morph_closing, disk
from skimage.color import rgb2lab
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, rotate as ndi_rotate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------- 경로 설정 ----------
ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
ORTHO = ROOT / "data" / "간척지 초고해상도 정사영상-orthomosaic.tiff"
SHP_ZIP = ROOT / "shapefile" / "GJSM-1-1_Boundary.zip"
OUT_DIR = ROOT / "result" / "verify"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 검증 파라미터 ----------
TILE_PIX = 2000          # 서브타일 크기(px). 5.25mm/px * 2000 = 10.5m -> 두둑 ~15개 들어옴
N_SAMPLE_TILES = 5       # 폴리곤 내부에서 자동 선택할 타일 수 (가운데 + 4개 모서리 근처)

# 작물 재배 표준 (논콩) — 자동 추정의 시드값
EXPECTED_ROW_SPACING_M = 0.65   # 60-70cm 권장 행간
EXPECTED_PLANT_SPACING_M = 0.17  # 15-20cm 권장 주간


# ---------- 1. SHP 로드 + 재투영 ----------
def load_field_polygon(zip_path: Path, target_crs: str | int):
    with zipfile.ZipFile(zip_path) as z:
        shp_name = next(n for n in z.namelist() if n.endswith(".shp"))
    gdf = gpd.read_file(f"zip://{zip_path}!{shp_name}")
    gdf = gdf.to_crs(target_crs)
    return gdf


def sample_tile_centers_inside_polygon(poly, gsd_m: float, tile_pix: int, n: int = 5):
    """폴리곤 안쪽으로 tile_pix/2 만큼 buffer한 영역에서 중심 후보 추출.
       중심 1개 + 내부 4사분면 각 1개 = 5개 위치 반환.
    """
    half_m = tile_pix * gsd_m / 2
    inner = poly.buffer(-half_m)
    if inner.is_empty:
        # 폴리곤이 너무 좁으면 더 적은 마진으로
        inner = poly.buffer(-half_m * 0.5)
    if inner.is_empty:
        return [("center", *poly.centroid.coords[0])]

    centroid = inner.centroid
    cx0, cy0 = centroid.x, centroid.y
    bounds = inner.bounds  # (minx, miny, maxx, maxy)

    positions = [("center", cx0, cy0)]
    # 내부 4사분면: 중심에서 각 모서리쪽으로 절반 지점들 — 폴리곤 안인지 확인 후 채택
    # 안 되면 점 점 안쪽으로 후퇴
    quad_names = ["TL", "TR", "BL", "BR"]
    quad_offsets = [(-0.35, -0.35), (0.35, -0.35), (-0.35, 0.35), (0.35, 0.35)]
    bx, by = bounds[2] - bounds[0], bounds[3] - bounds[1]
    for name, (rx, ry) in zip(quad_names, quad_offsets):
        for shrink in [1.0, 0.7, 0.4, 0.2]:
            cx = cx0 + rx * bx / 2 * shrink
            cy = cy0 + ry * by / 2 * shrink
            if inner.contains(Point(cx, cy)):
                positions.append((name, cx, cy))
                break
        else:
            positions.append((name, cx0, cy0))  # fallback
    return positions[:n]


# ---------- 2. 크롭 ----------
def read_tile_at_center(src: rasterio.io.DatasetReader, cx: float, cy: float, tile_pix: int):
    """필지 중심 (m, m)에서 tile_pix × tile_pix 윈도우 읽기. RGBA -> (RGB, alpha)."""
    row_center, col_center = src.index(cx, cy)
    half = tile_pix // 2
    row_off = max(0, row_center - half)
    col_off = max(0, col_center - half)
    window = Window(col_off=col_off, row_off=row_off, width=tile_pix, height=tile_pix)
    data = src.read(window=window)  # (4, H, W) uint8
    if data.shape[0] >= 4:
        rgb = data[:3].astype(np.float32)
        alpha = data[3]
    else:
        rgb = data[:3].astype(np.float32)
        alpha = np.full(rgb.shape[1:], 255, dtype=np.uint8)
    tile_transform = src.window_transform(window)
    return rgb, alpha, tile_transform, window


# ---------- 3. 색지수 + 조명 보정 ----------
def compute_indices(rgb: np.ndarray):
    """식생 분리에 효과적인 다중 지수 산출.
       - ExG  : 2g - r - b
       - ExGR : ExG - 1.4r + g  (Meyer 2008) — 토양 노이즈 더 잘 거름
       - a*    : CIE Lab a 채널 (식생: 음수, 사질토: ~0)
    """
    s = rgb.sum(axis=0) + 1e-6
    r = rgb[0] / s
    g = rgb[1] / s
    b = rgb[2] / s
    exg = 2.0 * g - r - b
    exgr = exg - 1.4 * r + g
    # rgb2lab은 (H,W,3) float [0,1] 입력
    rgb01 = np.transpose(rgb / 255.0, (1, 2, 0)).astype(np.float32)
    lab = rgb2lab(rgb01)
    a_star = lab[..., 1]
    return exg, exgr, a_star


def detrend_local(arr: np.ndarray, sigma: int = 200) -> np.ndarray:
    """큰 시그마 가우시안 빼서 광 그라데이션 제거."""
    from scipy.ndimage import gaussian_filter
    bg = gaussian_filter(arr, sigma=sigma)
    return arr - bg


# ---------- 4. 이진화 + 두둑방향 형태학 정리 ----------
def binarize_vegetation_raw(exg: np.ndarray, valid_mask: np.ndarray):
    """1차 이진화 (노이즈 허용): ExG Otsu OR Sauvola.
       이 단계는 두둑 위 약한 새싹까지 전부 살리는 게 목표. 노이즈는 다음 단계에서 정리.
    """
    otsu_t = threshold_otsu(exg[valid_mask])
    sauvola_t = threshold_sauvola(exg, window_size=51, k=0.2)
    bw = ((exg > otsu_t) | (exg > sauvola_t)) & valid_mask
    return bw, otsu_t


def refine_mask_row_aware(bw_noisy: np.ndarray, row_angle_deg: float, gsd_m: float,
                          close_len_m: float = 0.06, open_radius: int = 1,
                          min_size: int = 30):
    """두둑 방향 인지 형태학 정리:
       1) 두둑 방향이 가로축이 되도록 회전
       2) 가로방향 closing (커널 길이 ≈ 새싹 크기의 2~3배) — 두둑 위 인접 식생 픽셀 연결,
          따라서 노이즈가 진짜 식생 옆이면 살리고 단독이면 그대로 둠.
       3) opening(1)로 단독 점 노이즈 제거 (사질토 입자)
       4) min_size로 컴포넌트 단위 잡노이즈 추가 제거
       반환: 회전된 정리 마스크 (이후 단계가 회전 좌표계에서 동작).
    """
    rot = ndi_rotate(bw_noisy.astype(np.uint8), angle=-row_angle_deg, reshape=True, order=0)
    rot = rot.astype(bool)
    close_px = max(3, int(close_len_m / gsd_m))   # 0.06m / 0.00525m ≈ 11 px
    h_kernel = np.ones((1, close_px), dtype=bool)
    closed = morph_closing(rot, footprint=h_kernel)
    opened = opening(closed, footprint=disk(open_radius))
    cleaned = remove_small_objects(opened, min_size=min_size)
    return cleaned, rot


# ---------- 5. 두둑(파종 라인) 방향 추정 ----------
def estimate_row_angle(bw: np.ndarray, theta_step: float = 0.5) -> float:
    """Radon transform으로 가장 강한 두둑 각도(도) 추정.
    반환: deg in [-90, 90), 0 = 영상 가로(x축)와 평행, 양수 = 시계 반대방향.
    """
    # 다운샘플로 속도 ↑
    step = max(1, bw.shape[0] // 600)
    bw_ds = bw[::step, ::step].astype(np.float32)
    thetas = np.arange(-90, 90, theta_step)
    sino = radon(bw_ds, theta=thetas, circle=False)
    # 각 angle별 분산(=라인이 평행이면 분산 큼)
    var = sino.var(axis=0)
    best_idx = int(np.argmax(var))
    angle_proj = thetas[best_idx]      # projection axis = 라인의 수직축
    row_angle = (angle_proj + 90.0) % 180.0 - 90.0  # 라인 자체의 방향
    return row_angle, thetas, var


# ---------- 6. 두둑 라인 좌표 추출 (이미 회전된 마스크 입력 받음) ----------
def extract_row_centers_from_rot(rot_bw: np.ndarray, gsd_m: float,
                                 expected_spacing_m: float):
    """회전된 마스크의 가로방향 누적 → 직교축 1D 프로파일 → 두둑 중심 픽셀행 검출."""
    row_profile = rot_bw.astype(np.float32).sum(axis=1)
    row_profile_s = gaussian_filter1d(row_profile, sigma=3)
    min_dist_px = int(expected_spacing_m / gsd_m * 0.6)
    height_thr = np.percentile(row_profile_s, 50)
    peaks, _ = find_peaks(row_profile_s, distance=min_dist_px, height=height_thr)
    return row_profile_s, peaks


# ---------- 7. 두둑별 1D 피크 = 개체 카운팅 + 결주 검출 ----------
def count_plants_per_row(rot_bw: np.ndarray, peaks_rows: np.ndarray, gsd_m: float,
                         expected_plant_spacing_m: float, row_half_width_px: int = 30):
    """각 두둑 라인 부근(±half_width)을 잘라 1D 합산 프로파일 -> find_peaks."""
    H, W = rot_bw.shape
    min_dist_px = max(3, int(expected_plant_spacing_m / gsd_m * 0.6))
    per_row_results = []
    for r in peaks_rows:
        r0 = max(0, r - row_half_width_px)
        r1 = min(H, r + row_half_width_px + 1)
        band = rot_bw[r0:r1]
        col_profile = band.sum(axis=0).astype(np.float32)
        col_profile_s = gaussian_filter1d(col_profile, sigma=1.5)
        thr = max(1.0, np.percentile(col_profile_s, 60))
        peaks, _ = find_peaks(col_profile_s, distance=min_dist_px, height=thr)
        # 결주 구간 = 인접 피크 간 픽셀 거리 > 1.5 * median
        gaps = []
        if len(peaks) >= 3:
            d = np.diff(peaks)
            med = float(np.median(d))
            for i, dd in enumerate(d):
                if dd > med * 1.5:
                    gaps.append((peaks[i], peaks[i + 1], dd))
        per_row_results.append({
            "row_pix": int(r),
            "profile": col_profile_s,
            "plants_x": peaks,
            "n_plants": len(peaks),
            "median_spacing_px": float(np.median(np.diff(peaks))) if len(peaks) >= 2 else np.nan,
            "gaps": gaps,
        })
    return per_row_results


# ---------- 8. 시각화 ----------
def visualize(rgb: np.ndarray, exg: np.ndarray,
              bw_raw: np.ndarray, rot_raw: np.ndarray, rot_cleaned: np.ndarray,
              row_profile: np.ndarray, row_peaks: np.ndarray,
              per_row: list, row_angle_deg: float, gsd_m: float, out_png: Path):
    rgb_disp = np.transpose(rgb.astype(np.uint8), (1, 2, 0))
    fig, axes = plt.subplots(3, 3, figsize=(20, 19))

    axes[0, 0].imshow(rgb_disp)
    axes[0, 0].set_title("(1) 원본 RGB 서브타일")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(exg, cmap="RdYlGn", vmin=-0.05, vmax=0.15)
    axes[0, 1].set_title("(2) ExG (조명보정 후)")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.04)

    raw_pct = bw_raw.sum() / bw_raw.size * 100
    axes[0, 2].imshow(bw_raw, cmap="Greens")
    axes[0, 2].set_title(f"(3) 1차 마스크 ExG Otsu+Sauvola ({raw_pct:.1f}%)\n"
                         "노이즈 허용 — 약한 새싹 보존")
    axes[0, 2].axis("off")

    rot_raw_pct = rot_raw.sum() / max(1, rot_raw.size) * 100
    axes[1, 0].imshow(rot_raw, cmap="Greens")
    axes[1, 0].set_title(f"(4) 두둑각 보정 회전 ({row_angle_deg:.2f}°)\n원본 노이즈 마스크")
    axes[1, 0].axis("off")

    cleaned_pct = rot_cleaned.sum() / max(1, rot_cleaned.size) * 100
    axes[1, 1].imshow(rot_cleaned, cmap="Greens")
    axes[1, 1].set_title(f"(5) 두둑방향 closing + opening + size필터\n"
                         f"정리 후: {cleaned_pct:.1f}% (노이즈 {raw_pct-cleaned_pct:+.1f}%p 감소)")
    axes[1, 1].axis("off")

    # 정리 + 두둑 라인
    axes[1, 2].imshow(rot_cleaned, cmap="Greens")
    for r in row_peaks:
        axes[1, 2].axhline(r, color="red", lw=0.4, alpha=0.7)
    axes[1, 2].set_title(f"(6) 두둑 라인 검출: {len(row_peaks)}행")
    axes[1, 2].axis("off")

    axes[2, 0].plot(row_profile, color="green")
    axes[2, 0].scatter(row_peaks, row_profile[row_peaks], color="red", s=10, zorder=3)
    axes[2, 0].set_title("(7) 직교축 두둑 프로파일 + 피크")
    axes[2, 0].set_xlabel("회전영상 row index")
    axes[2, 0].set_ylabel("식생 픽셀 수")

    # 두둑별 개체수 분포
    n_per_row = [r["n_plants"] for r in per_row]
    if n_per_row:
        axes[2, 1].bar(range(len(n_per_row)), n_per_row, color="seagreen")
        med_n = np.median(n_per_row)
        axes[2, 1].axhline(med_n, color="red", lw=1, ls="--",
                          label=f"중앙값 {int(med_n)}")
        axes[2, 1].set_title(f"(8) 두둑별 개체수 (총 {sum(n_per_row)}개)")
        axes[2, 1].set_xlabel("두둑 index")
        axes[2, 1].set_ylabel("개체 수")
        axes[2, 1].legend()

    if per_row:
        sample_idx = len(per_row) // 2
        sample = per_row[sample_idx]
        axes[2, 2].plot(sample["profile"], color="darkgreen")
        axes[2, 2].scatter(sample["plants_x"], sample["profile"][sample["plants_x"]],
                          color="red", s=12, zorder=3, label=f"개체 {sample['n_plants']}개")
        for g0, g1, dd in sample["gaps"]:
            axes[2, 2].axvspan(g0, g1, color="orange", alpha=0.3)
        spacing_cm = sample["median_spacing_px"] * gsd_m * 100
        axes[2, 2].set_title(
            f"(9) 샘플 두둑(중앙) 1D 프로파일\n개체수={sample['n_plants']}, "
            f"median 주간={spacing_cm:.1f}cm, 결주={len(sample['gaps'])}구간"
        )
        axes[2, 2].set_xlabel("회전영상 col index")
        axes[2, 2].set_ylabel("식생 픽셀 수")
        axes[2, 2].legend()

    plt.suptitle("콩 입모율 검증 (두둑방향 closing) — GJSM-1-1 서브타일", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


# ---------- 단일 타일 처리 ----------
def process_tile(src, poly, gsd_m, tile_name: str, cx: float, cy: float):
    rgb, alpha, tile_transform, window = read_tile_at_center(src, cx, cy, TILE_PIX)
    valid_mask = alpha > 0
    valid_ratio = valid_mask.sum() / valid_mask.size

    result = {
        "tile": tile_name,
        "center_x": cx, "center_y": cy,
        "valid_ratio": valid_ratio,
        "raw_pct": np.nan, "cleaned_pct": np.nan,
        "row_angle": np.nan, "n_rows": 0,
        "row_spacing_cm": np.nan,
        "n_plants": 0, "plant_spacing_cm": np.nan,
        "n_gaps": 0, "emergence_est_pct": np.nan,
        "n_empty_rows": 0,
    }

    if valid_ratio < 0.5:
        print(f"  [{tile_name}] 유효픽셀 {valid_ratio*100:.1f}% — 스킵")
        return result, None

    exg_raw, _, _ = compute_indices(rgb)
    exg = detrend_local(exg_raw, sigma=150)
    exg[~valid_mask] = 0

    bw_raw, otsu_t = binarize_vegetation_raw(exg, valid_mask)
    raw_pct = bw_raw.sum() / valid_mask.sum() * 100
    result["raw_pct"] = raw_pct

    row_angle, _, _ = estimate_row_angle(bw_raw)
    result["row_angle"] = row_angle

    rot_cleaned, rot_raw = refine_mask_row_aware(
        bw_raw, row_angle, gsd_m,
        close_len_m=0.06, open_radius=1, min_size=30,
    )
    cleaned_pct = rot_cleaned.sum() / max(1, rot_cleaned.size) * 100
    result["cleaned_pct"] = cleaned_pct

    row_profile, row_peaks = extract_row_centers_from_rot(
        rot_cleaned, gsd_m, EXPECTED_ROW_SPACING_M
    )
    result["n_rows"] = len(row_peaks)
    if len(row_peaks) >= 2:
        med_row_spacing_m = float(np.median(np.diff(row_peaks))) * gsd_m
        result["row_spacing_cm"] = med_row_spacing_m * 100
    else:
        med_row_spacing_m = EXPECTED_ROW_SPACING_M

    per_row = count_plants_per_row(
        rot_cleaned, row_peaks, gsd_m, EXPECTED_PLANT_SPACING_M,
        row_half_width_px=int(med_row_spacing_m / gsd_m * 0.35),
    )
    total_plants = sum(r["n_plants"] for r in per_row)
    total_gaps = sum(len(r["gaps"]) for r in per_row)
    n_per_row = [r["n_plants"] for r in per_row]
    n_empty_rows = sum(1 for n in n_per_row if n <= 3)  # ≤3 개체는 사실상 미파종

    if per_row:
        med_plant_spacing_cm = np.nanmedian([r["median_spacing_px"] for r in per_row]) * gsd_m * 100
    else:
        med_plant_spacing_cm = np.nan

    # 추정 입모율 = 검출개체 / (검출두둑수 × 기대두둑당개체)
    if len(row_peaks) >= 1:
        tile_len_m = TILE_PIX * gsd_m
        expected_per_row = tile_len_m / EXPECTED_PLANT_SPACING_M
        emergence_est = total_plants / (len(row_peaks) * expected_per_row) * 100
    else:
        emergence_est = np.nan

    result.update({
        "n_plants": total_plants,
        "plant_spacing_cm": med_plant_spacing_cm,
        "n_gaps": total_gaps,
        "emergence_est_pct": emergence_est,
        "n_empty_rows": n_empty_rows,
    })

    out_png = OUT_DIR / f"GJSM-1-1_{tile_name}_verify.png"
    visualize(rgb, exg, bw_raw, rot_raw, rot_cleaned, row_profile, row_peaks,
              per_row, row_angle, gsd_m, out_png)
    return result, per_row


# ---------- main ----------
def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    print("[1] SHP 로드 + 재투영")
    with rasterio.open(ORTHO) as src:
        target_crs = src.crs
        gsd_m = abs(src.transform.a)
        gdf = load_field_polygon(SHP_ZIP, target_crs)
        poly = gdf.geometry.iloc[0]
        minx, miny, maxx, maxy = poly.bounds
        print(f"     필지 면적: {poly.area/10000:.3f} ha,  GSD: {gsd_m*1000:.3f} mm/px")
        print(f"     bounds (m): {minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}\n")

        positions = sample_tile_centers_inside_polygon(poly, gsd_m, TILE_PIX, N_SAMPLE_TILES)
        print(f"     폴리곤 내부 샘플 위치 {len(positions)}개:")
        for name, cx, cy in positions:
            print(f"       {name}: ({cx:.1f}, {cy:.1f})")
        print()

        results = []
        for tile_name, cx, cy in positions:
            print(f"[{tile_name}] cx={cx:.1f}, cy={cy:.1f}")
            r, _ = process_tile(src, poly, gsd_m, tile_name, cx, cy)
            results.append(r)
            if not np.isnan(r["emergence_est_pct"]):
                print(f"   → 두둑 {r['n_rows']}행({r['row_spacing_cm']:.1f}cm), "
                      f"개체 {r['n_plants']}개({r['plant_spacing_cm']:.1f}cm), "
                      f"결주구간 {r['n_gaps']}, 사실상 빈두둑 {r['n_empty_rows']}, "
                      f"입모율 {r['emergence_est_pct']:.1f}%")

    # 요약 CSV
    summary_path = OUT_DIR / "GJSM-1-1_multi_tile_summary.csv"
    import csv
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n요약 CSV: {summary_path}")
    print("PNG 위치:", OUT_DIR)


if __name__ == "__main__":
    main()
