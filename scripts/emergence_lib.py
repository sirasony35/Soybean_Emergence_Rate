"""
콩 입모율 분석 공통 알고리즘 모듈

검증 스크립트(verify_emergence_tile.py)에서 검증된 핵심 알고리즘을 모아둠.
01_crop_fields.py 와 02_analyze_emergence.py 에서 import 해 사용.

알고리즘 흐름:
  RGB → ExG → 1차 이진화(노이즈 허용) → Radon 두둑각 →
  두둑방향 closing → 두둑 라인 검출 → 두둑별 1D 피크(개체) → 결주 구간
"""

from __future__ import annotations
from pathlib import Path
import zipfile
import numpy as np
import geopandas as gpd
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.transform import radon
from skimage.morphology import remove_small_objects, opening, closing as morph_closing, disk
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, gaussian_filter1d, rotate as ndi_rotate


# -------- 표준 재배 제원 (사용자 확정) --------
# 새 흐름: 두둑(ridge) → 파종 라인(line) → 개체(plant)
#   두둑 피치 (ridge pitch)   = 65 cm  (고랑 포함 두둑 간 중심거리)
#   조간 (한 두둑 안 dual-row) = 30 cm  (있을 경우)
#   주간 (in-line plant)       = 20 cm
STD_RIDGE_PITCH_M         = 0.65    # 두둑 중심 간 거리 (= "행간")
STD_INTRA_RIDGE_LINE_CM   = 30.0    # 두둑 안 두 줄 간격 (조간, dual-row 시)
STD_INTER_RIDGE_CM        = 65.0    # = 두둑 피치 × 100 (호환)
STD_PLANT_SPACING_M       = 0.20    # 주간

# 적합 판정 허용 오차 (±cm)
RIDGE_PITCH_TOL_CM   = 12.0  # 두둑 피치 적합 ±12cm
ROW_TOLERANCE_CM     = 7.0   # (조간) 적합 ±7cm
PLANT_TOLERANCE_CM   = 5.0   # 주간 적합 ±5cm
INTER_RIDGE_TOL_CM   = 12.0  # 호환

# 두둑 안 분류 임계
INTRA_RIDGE_MAX_CM        = 42.0   # 두둑 안 두 줄로 인식할 최대 간격

# 라인/개체 검출 최소 간격
MIN_LINE_SPACING_M   = 0.22   # 22 cm: 조간 30cm 잡되 노이즈 분리
MIN_PLANT_SPACING_M  = 0.13   # 13 cm: 17~20cm 주간 잡되 false-positive 방지

# Backward-compatibility
STD_ROW_SPACING_M    = STD_INTRA_RIDGE_LINE_CM / 100   # 0.30 (조간)


# -------- SHP 로드 --------
def load_polygon_zip(zip_path: Path, target_crs):
    with zipfile.ZipFile(zip_path) as z:
        shp_name = next(n for n in z.namelist() if n.endswith(".shp"))
    gdf = gpd.read_file(f"zip://{zip_path}!{shp_name}")
    return gdf.to_crs(target_crs)


# -------- 색지수 + 조명 보정 --------
def compute_exg(rgb: np.ndarray) -> np.ndarray:
    """ExG = 2g - r - b (정규화). rgb shape: (3, H, W) float."""
    s = rgb.sum(axis=0) + 1e-6
    r = rgb[0] / s
    g = rgb[1] / s
    b = rgb[2] / s
    return 2.0 * g - r - b


def detrend_local(arr: np.ndarray, sigma: int = 150) -> np.ndarray:
    """대 sigma 가우시안 빼서 광 그라데이션 제거."""
    bg = gaussian_filter(arr, sigma=sigma)
    return arr - bg


# -------- 이진화 --------
def binarize_vegetation_raw(exg: np.ndarray, valid_mask: np.ndarray):
    """1차 이진화(노이즈 허용): ExG Otsu OR Sauvola. 약한 새싹까지 포착이 목적."""
    otsu_t = threshold_otsu(exg[valid_mask])
    sauvola_t = threshold_sauvola(exg, window_size=51, k=0.2)
    bw = ((exg > otsu_t) | (exg > sauvola_t)) & valid_mask
    return bw, otsu_t


# -------- 두둑각 추정 (수정) --------
def estimate_row_angle(bw: np.ndarray, theta_step: float = 0.5) -> float:
    """Radon transform으로 두둑 방향(deg, [-90, 90)) 추정.

    skimage.transform.radon: 이미지를 theta만큼 회전 후 axis=0으로 합.
    수직 라인이 강하면 theta=0에서 max variance → 라인 방향 = -90° (수직).
    수평 라인이 강하면 theta=90에서 max variance → 라인 방향 = 0° (수평).
    즉 Radon이 반환하는 angle_proj은 라인의 PERPENDICULAR 방향.
    실제 라인 방향 = angle_proj - 90° (혹은 + 90°, mod 180 normalization).
    """
    step = max(1, max(bw.shape) // 600)
    bw_ds = bw[::step, ::step].astype(np.float32)
    thetas = np.arange(-90, 90, theta_step)
    sino = radon(bw_ds, theta=thetas, circle=False)
    best_idx = int(np.argmax(sino.var(axis=0)))
    angle_proj = thetas[best_idx]   # Radon에서 max variance인 projection 방향
    # 라인 방향은 projection 방향에서 90° 회전
    row_angle = angle_proj - 90.0
    # [-90, 90)로 normalize
    while row_angle < -90:
        row_angle += 180
    while row_angle >= 90:
        row_angle -= 180
    return float(row_angle)


# -------- 두둑방향 형태학 정리 --------
def refine_mask_row_aware(bw_noisy: np.ndarray, row_angle_deg: float, gsd_m: float,
                          close_len_m: float = 0.06, open_radius: int = 1,
                          min_size: int = 30):
    """두둑이 가로축이 되도록 회전 → 가로방향 closing → opening → size 필터."""
    rot = ndi_rotate(bw_noisy.astype(np.uint8), angle=-row_angle_deg, reshape=True, order=0).astype(bool)
    close_px = max(3, int(close_len_m / gsd_m))
    h_kernel = np.ones((1, close_px), dtype=bool)
    closed = morph_closing(rot, footprint=h_kernel)
    opened = opening(closed, footprint=disk(open_radius))
    cleaned = remove_small_objects(opened, min_size=min_size)
    return cleaned, rot


# -------- 두둑 라인 검출 (구버전 호환용) --------
def extract_row_centers_from_rot(rot_bw: np.ndarray, gsd_m: float,
                                 expected_spacing_m: float = STD_ROW_SPACING_M):
    """[deprecated] v2 사용 권장."""
    return detect_row_lines_v2(rot_bw, gsd_m)


# -------- 두둑 라인 검출 v2 (튜닝됨, 진단으로 확정) --------
def detect_row_lines_v2(rot_bw: np.ndarray, gsd_m: float,
                        min_line_spacing_m: float = MIN_LINE_SPACING_M,
                        smooth_sigma: float = 3.0,
                        height_quantile: float = 0.25,
                        prominence_factor_of_mean: float = 0.057):
    """라인 검출. 진단 sweep으로 확정한 설정:
       - height_thr = quantile(nonzero, 0.25)
       - prominence = mean(nonzero) × 0.057  (GJSM-1-1에서 prom=500 ≈ mean×0.057)
       이 조합으로 조간 평균 30cm 표준과 잘 일치.
       반환: (row_profile_smoothed, peaks_y_px)
    """
    row_profile = rot_bw.astype(np.float32).sum(axis=1)
    row_profile_s = gaussian_filter1d(row_profile, sigma=smooth_sigma)
    nonzero = row_profile_s[row_profile_s > 0]
    if nonzero.size < 20:
        return row_profile_s, np.array([], dtype=int)
    height_thr = float(np.quantile(nonzero, height_quantile))
    prominence_val = max(1.0, float(nonzero.mean()) * prominence_factor_of_mean)
    min_dist_px = max(5, int(min_line_spacing_m / gsd_m))
    peaks, _ = find_peaks(row_profile_s,
                          distance=min_dist_px,
                          height=height_thr,
                          prominence=prominence_val)
    return row_profile_s, peaks


def cluster_lines_to_ridges(peaks_y_px: np.ndarray, gsd_m: float,
                            intra_max_cm: float = INTRA_RIDGE_MAX_CM):
    """라인을 두둑(ridge) 페어로 클러스터.
       반환: list of list — 각 두둑에 속하는 라인 y_px 인덱스들.
    """
    peaks_y_px = np.asarray(peaks_y_px, dtype=int)
    if peaks_y_px.size == 0:
        return []
    intra_max_px = intra_max_cm / 100 / gsd_m
    ridges = []
    current = [int(peaks_y_px[0])]
    for i in range(1, len(peaks_y_px)):
        gap = peaks_y_px[i] - peaks_y_px[i-1]
        if gap < intra_max_px:
            current.append(int(peaks_y_px[i]))
        else:
            ridges.append(current)
            current = [int(peaks_y_px[i])]
    ridges.append(current)
    return ridges


def compute_line_spacings_split(peaks_y_px: np.ndarray, gsd_m: float,
                                 intra_max_cm: float = INTRA_RIDGE_MAX_CM):
    """라인 간 인접 거리를 조간(intra-ridge) vs 행간(inter-ridge)으로 분리.
       반환: (intra_cm_array, inter_cm_array)
    """
    peaks_y_px = np.asarray(peaks_y_px)
    if peaks_y_px.size < 2:
        return np.array([]), np.array([])
    gaps_cm = np.diff(peaks_y_px) * gsd_m * 100
    intra = gaps_cm[gaps_cm < intra_max_cm]
    inter = gaps_cm[gaps_cm >= intra_max_cm]
    return intra, inter


# -------- 타일 각도만 빠르게 추정 (1-pass) --------
def estimate_tile_angle_only(tile_bw: np.ndarray, gsd_m: float):
    """타일에서 각도만 빠르게 검출 (회전/검출 생략).
       반환: (angle_deg or None, veg_count)
    """
    veg = int(tile_bw.sum())
    if veg < 1000:
        return None, veg
    return estimate_row_angle(tile_bw), veg


def find_dominant_angles(angles: list, bin_width: float = 5.0,
                         min_prominence_ratio: float = 0.2,
                         max_modes: int = 2):
    """전체 타일 각도 분포에서 dominant 모드 찾기 (히스토그램 + find_peaks).
       angle은 ±90° 범위, 원형 보정 — 80°와 -80°는 가깝다.
       반환: list of dominant angles (deg)
    """
    if len(angles) < 3:
        return [float(np.median(angles))] if angles else []
    # 원형 거리: 각도를 unit vector로 변환해 평균 후 다시 각도화
    # 그러나 dual-direction(line)이므로 2*angle을 사용 (라인 방향은 ±180 동치)
    angles_arr = np.array(angles)
    angles2 = (angles_arr * 2)  # 2배로 unwrap (라인 방향)
    # mod 360, range [0, 360)
    angles2 = np.mod(angles2 + 360, 360)
    bins = np.arange(0, 365, bin_width)
    hist, edges = np.histogram(angles2, bins=bins)
    if hist.max() == 0:
        return [float(np.median(angles_arr))]
    # 원형 평활화 (히스토그램 wrap)
    from scipy.signal import find_peaks
    hist_padded = np.concatenate([hist[-3:], hist, hist[:3]])
    peaks, _ = find_peaks(hist_padded, distance=int(20/bin_width))
    # 원본 인덱스로 환원
    peaks = [p - 3 for p in peaks if 3 <= p < 3 + len(hist)]
    if not peaks:
        return [float(np.median(angles_arr))]
    # prominence 기준 — 최대 모드의 min_prominence_ratio 이상만
    max_h = max(hist[p] for p in peaks)
    strong = [p for p in peaks if hist[p] >= max_h * min_prominence_ratio]
    strong.sort(key=lambda p: -hist[p])
    strong = strong[:max_modes]
    # 각 강한 모드 → 모드 주변 ±bin_width 안의 각도들의 평균
    centers2 = (edges[:-1] + edges[1:]) / 2  # 2배 공간
    modes = []
    for p in strong:
        c2 = centers2[p]
        # ±10° (2배 공간에서 ±20°) 안의 원본 각도 골라 평균
        mask = np.abs(np.mod(angles2 - c2 + 180, 360) - 180) <= 20
        if mask.sum() > 0:
            ang2_mean = np.mod(angles2[mask].mean(), 360)
            modes.append(ang2_mean / 2)  # 원래 공간으로 환원
        else:
            modes.append(c2 / 2)
    # [-90, 90)로 normalize
    modes_norm = []
    for m in modes:
        while m >= 90: m -= 180
        while m < -90: m += 180
        modes_norm.append(float(m))
    return modes_norm


def angular_dist(a: float, b: float) -> float:
    """각도 a와 b의 직선 방향 거리 (라인은 ±180° 동치). 반환 ≥ 0."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def snap_to_dominant(angle: float, modes: list, max_dist_deg: float = 15.0):
    """angle을 가장 가까운 mode로 SNAP. max_dist 이상이면 None."""
    if not modes:
        return None
    dists = [(m, angular_dist(angle, m)) for m in modes]
    best_m, best_d = min(dists, key=lambda x: x[1])
    if best_d > max_dist_deg:
        return None
    return best_m


# ================================================================
# 새 흐름: 두둑 → 파종 라인 → 개체 (2-stage)
# ================================================================

def detect_lines_and_cluster_to_ridges(rot_bw: np.ndarray, gsd_m: float,
                                        min_line_spacing_m: float = 0.20,
                                        smooth_sigma: float = 5.0,
                                        height_quantile: float = 0.25,
                                        prominence_factor: float = 0.05,
                                        intra_ridge_max_cm: float = 75.0):
    """새 흐름: 모든 줄을 관대하게 검출 → 인접 줄을 두둑으로 클러스터.
       두둑 = 인접 줄 간격 ≤ intra_ridge_max_cm 인 줄 그룹 (최대 2줄/두둑).

       반환:
         profile_s            : 회전영상 가로축 합 (smoothed)
         line_ys              : 모든 검출 줄 Y (오름차순)
         line_to_ridge        : dict {line_y: ridge_idx}
         ridges               : list[list[int]]  — 각 두둑의 줄 Y 리스트
    """
    profile = rot_bw.astype(np.float32).sum(axis=1)
    profile_s = gaussian_filter1d(profile, sigma=smooth_sigma)
    nz = profile_s[profile_s > 0]
    if nz.size < 20:
        return profile_s, np.array([], dtype=int), {}, []
    height_thr = float(np.quantile(nz, height_quantile))
    prominence_val = max(1.0, float(nz.mean()) * prominence_factor)
    min_dist_px = max(5, int(min_line_spacing_m / gsd_m))
    peaks, _ = find_peaks(profile_s,
                          distance=min_dist_px,
                          height=height_thr,
                          prominence=prominence_val)
    if len(peaks) == 0:
        return profile_s, np.array([], dtype=int), {}, []

    # 두둑 클러스터링 — 인접 줄 간격 ≤ intra_ridge_max_cm 면 같은 두둑
    intra_max_px = intra_ridge_max_cm / 100 / gsd_m
    line_ys = peaks  # 이미 오름차순
    ridges = []
    current = [int(line_ys[0])]
    for i in range(1, len(line_ys)):
        gap_px = line_ys[i] - line_ys[i-1]
        # 두둑당 최대 2줄로 제한 (3줄째는 새 두둑 시작)
        if gap_px <= intra_max_px and len(current) < 2:
            current.append(int(line_ys[i]))
        else:
            ridges.append(current)
            current = [int(line_ys[i])]
    ridges.append(current)

    line_to_ridge = {}
    for ridge_idx, lines_in_ridge in enumerate(ridges):
        for ly in lines_in_ridge:
            line_to_ridge[int(ly)] = ridge_idx
    return profile_s, line_ys, line_to_ridge, ridges


# -------- 타일별 분석 (혼합 두둑 방향 처리) --------
def analyze_tile(tile_bw: np.ndarray, gsd_m: float,
                 close_len_m: float = 0.06,
                 min_line_spacing_m: float = MIN_LINE_SPACING_M,
                 band_half_m: float = 0.12,
                 min_plant_spacing_m: float = MIN_PLANT_SPACING_M,
                 force_angle: float | None = None):
    """단일 타일 분석: 각도 검출(또는 강제) → 회전 → 라인 검출 → 개체 검출.
       반환: {
         'angle': local row angle (deg),
         'rot_cleaned': rotated cleaned mask,
         'rot_shape': rotated shape,
         'orig_shape': original tile shape,
         'peaks_y': row line peaks (y in rotated),
         'per_row': list of {row_pix, plants_x, n_plants, gaps},
       }
    """
    # 타일 내 식생 충분한지 확인
    if tile_bw.sum() < 1000:
        return None

    local_angle = float(force_angle) if force_angle is not None else estimate_row_angle(tile_bw)
    rot_cleaned, _ = refine_mask_row_aware(
        tile_bw, local_angle, gsd_m,
        close_len_m=close_len_m, open_radius=1, min_size=30,
    )
    if rot_cleaned.sum() < 500:
        return None

    # 1단계 + 2단계 통합: 모든 줄 검출 → 두둑 클러스터링 (dual-row)
    profile_s, line_ys, line_to_ridge, ridges = detect_lines_and_cluster_to_ridges(
        rot_cleaned, gsd_m,
        min_line_spacing_m=0.20,
        intra_ridge_max_cm=75.0,
    )
    if len(line_ys) < 2 or not ridges:
        return None
    # 두둑 중심 = 두 줄의 평균 (또는 single이면 그 줄)
    ridge_peaks_y = np.array([int(np.mean(r)) for r in ridges], dtype=int)
    line_ridge_idx_list = [line_to_ridge.get(int(y), -1) for y in line_ys]

    # 3단계: 각 파종 라인 따라 개체 검출
    per_row = count_plants_per_row_v2(
        rot_cleaned, line_ys, gsd_m,
        band_half_m=band_half_m,
        min_plant_spacing_m=min_plant_spacing_m,
    )

    return {
        "angle": local_angle,
        "rot_cleaned": rot_cleaned,
        "rot_shape": rot_cleaned.shape,
        "orig_shape": tile_bw.shape,
        "ridge_peaks_y": ridge_peaks_y,           # 두둑 중심 Y들
        "ridges": ridges,                          # 두둑별 줄 Y 리스트
        "peaks_y": line_ys,                        # 파종 라인 Y들
        "line_ridge_idx": line_ridge_idx_list,     # 각 라인이 속한 두둑 idx
        "per_row": per_row,
    }


# -------- 라인별 개체 카운팅 v2 (엄격, false-positive 억제) --------
def count_plants_along_line(rot_bw: np.ndarray, peak_row_y: int, gsd_m: float,
                            band_half_px: int,
                            min_plant_spacing_m: float = MIN_PLANT_SPACING_M,
                            smooth_sigma: float = 1.5,
                            prominence_factor_of_mean: float = 0.10,
                            height_quantile: float = 0.25):
    """라인 한 줄에 대해 개체 검출. 진단 sweep으로 확정:
       - height_thr = quantile(nonzero, 0.25)
       - prominence = mean(nonzero) × 0.10
       반환: (col_profile_smoothed, peaks_x_px, info_dict)
    """
    H, W = rot_bw.shape
    r0 = max(0, peak_row_y - band_half_px)
    r1 = min(H, peak_row_y + band_half_px + 1)
    band = rot_bw[r0:r1]
    col_profile = band.astype(np.float32).sum(axis=0)
    col_profile_s = gaussian_filter1d(col_profile, sigma=smooth_sigma)
    nonzero = col_profile_s[col_profile_s > 0]
    if nonzero.size < 5:
        return col_profile_s, np.array([], dtype=int), {"valid_len_px": 0}
    valid_len_px = int(nonzero.size)
    height_thr = float(np.quantile(nonzero, height_quantile))
    prominence_val = max(0.5, float(nonzero.mean()) * prominence_factor_of_mean)
    min_dist_px = max(3, int(min_plant_spacing_m / gsd_m))
    peaks, _ = find_peaks(col_profile_s,
                          distance=min_dist_px,
                          height=height_thr,
                          prominence=prominence_val)
    return col_profile_s, peaks, {"valid_len_px": valid_len_px}


def count_plants_per_row_v2(rot_bw: np.ndarray, peaks_rows: np.ndarray, gsd_m: float,
                            expected_plant_spacing_m: float = STD_PLANT_SPACING_M,
                            band_half_m: float = 0.12,
                            min_plant_spacing_m: float = MIN_PLANT_SPACING_M,
                            prominence_factor_of_mean: float = 0.10):
    """v2: 각 라인별 엄격한 카운팅 + 결주 구간 검출.
       band_half_m: 라인 ±폭. 조간 30cm 의 ~40%(=12cm)면 인접 라인과 겹치지 않음.
    """
    band_half_px = max(3, int(band_half_m / gsd_m))
    results = []
    for r in peaks_rows:
        col_prof, plants_x, info = count_plants_along_line(
            rot_bw, int(r), gsd_m, band_half_px,
            min_plant_spacing_m=min_plant_spacing_m,
            prominence_factor_of_mean=prominence_factor_of_mean,
        )
        gaps = []
        med_spacing_px = float("nan")
        if len(plants_x) >= 2:
            d = np.diff(plants_x)
            med_spacing_px = float(np.median(d))
            # 결주 = median * 1.5 이상
            thr = med_spacing_px * 1.5
            for i, dd in enumerate(d):
                if dd > thr:
                    gaps.append((int(plants_x[i]), int(plants_x[i+1]), int(dd)))
        results.append({
            "row_pix": int(r),
            "profile": col_prof,
            "plants_x": plants_x,
            "n_plants": int(len(plants_x)),
            "median_spacing_px": med_spacing_px,
            "gaps": gaps,
            "valid_len_px": info["valid_len_px"],
        })
    return results


# -------- 두둑별 개체 카운팅 + 결주 검출 (구버전 호환) --------
def count_plants_per_row(rot_bw: np.ndarray, peaks_rows: np.ndarray, gsd_m: float,
                         expected_plant_spacing_m: float = STD_PLANT_SPACING_M,
                         row_half_width_m: float | None = None,
                         row_spacing_m: float | None = None):
    """각 두둑 라인 부근 밴드를 잘라 1D ExG 누적 → find_peaks로 개체 검출.
       반환: per_row [{row_pix, profile, plants_x, n_plants, median_spacing_px, gaps}]
       gaps = [(start_px, end_px, gap_px), ...]
    """
    if row_half_width_m is None:
        if row_spacing_m is None:
            row_spacing_m = STD_ROW_SPACING_M
        # 두둑 간격의 40% — 인접 두둑과 겹치지 않게
        row_half_width_m = row_spacing_m * 0.4
    H, _ = rot_bw.shape
    half_px = max(3, int(row_half_width_m / gsd_m))
    min_dist_px = max(3, int(expected_plant_spacing_m / gsd_m * 0.6))
    gap_factor = 1.5
    results = []
    for r in peaks_rows:
        r0 = max(0, r - half_px); r1 = min(H, r + half_px + 1)
        band = rot_bw[r0:r1]
        col_profile = band.sum(axis=0).astype(np.float32)
        col_profile_s = gaussian_filter1d(col_profile, sigma=1.5)
        thr = max(1.0, np.percentile(col_profile_s, 60))
        peaks, _ = find_peaks(col_profile_s, distance=min_dist_px, height=thr)
        gaps = []
        med_spacing_px = np.nan
        if len(peaks) >= 2:
            d = np.diff(peaks)
            med_spacing_px = float(np.median(d))
            for i, dd in enumerate(d):
                if dd > med_spacing_px * gap_factor:
                    gaps.append((int(peaks[i]), int(peaks[i + 1]), int(dd)))
        results.append({
            "row_pix": int(r),
            "profile": col_profile_s,
            "plants_x": peaks,
            "n_plants": int(len(peaks)),
            "median_spacing_px": med_spacing_px,
            "gaps": gaps,
        })
    return results


# -------- 간격 적합성 평가 --------
def evaluate_spacing_quality(per_row: list, gsd_m: float,
                             std_plant_cm: float = STD_PLANT_SPACING_M * 100,
                             plant_tol_cm: float = PLANT_TOLERANCE_CM):
    """모든 두둑의 주간 간격 분포에서 표준 간격 적합도 계산.
       반환:
         - mean_cm, std_cm, cv: 평균 / 표준편차 / 변동계수
         - in_range_ratio: 표준±허용 범위 안 비율 (0~1)
         - n_intervals: 간격 표본수
    """
    all_intervals_cm = []
    for r in per_row:
        if len(r["plants_x"]) >= 2:
            d_cm = np.diff(r["plants_x"]) * gsd_m * 100
            all_intervals_cm.extend(d_cm.tolist())
    if not all_intervals_cm:
        return {"mean_cm": np.nan, "std_cm": np.nan, "cv": np.nan,
                "in_range_ratio": np.nan, "n_intervals": 0}
    arr = np.asarray(all_intervals_cm)
    in_range = np.sum(np.abs(arr - std_plant_cm) <= plant_tol_cm) / len(arr)
    mean_cm = float(arr.mean())
    std_cm = float(arr.std())
    return {
        "mean_cm": mean_cm,
        "std_cm": std_cm,
        "cv": std_cm / mean_cm if mean_cm > 0 else np.nan,
        "in_range_ratio": float(in_range),
        "n_intervals": int(arr.size),
    }


def evaluate_row_spacing_quality(row_peaks_px: np.ndarray, gsd_m: float,
                                 std_row_cm: float = STD_ROW_SPACING_M * 100,
                                 row_tol_cm: float = ROW_TOLERANCE_CM):
    """두둑 라인 간격의 표준 적합도. row_peaks_px는 회전영상에서 검출된 라인 row index."""
    if len(row_peaks_px) < 2:
        return {"mean_cm": np.nan, "std_cm": np.nan, "cv": np.nan,
                "in_range_ratio": np.nan, "n_intervals": 0}
    d_cm = np.diff(np.asarray(row_peaks_px)) * gsd_m * 100
    in_range = np.sum(np.abs(d_cm - std_row_cm) <= row_tol_cm) / len(d_cm)
    mean_cm = float(d_cm.mean()); std_cm = float(d_cm.std())
    return {
        "mean_cm": mean_cm,
        "std_cm": std_cm,
        "cv": std_cm / mean_cm if mean_cm > 0 else np.nan,
        "in_range_ratio": float(in_range),
        "n_intervals": int(d_cm.size),
    }


# -------- 회전 좌표 → 원본 좌표 변환 --------
def rotated_to_original_coords(rot_yx: np.ndarray, row_angle_deg: float,
                               original_shape: tuple[int, int],
                               rotated_shape: tuple[int, int]) -> np.ndarray:
    """회전영상 픽셀 (y, x) → 원본영상 픽셀 (y, x).
       전제: rot = ndi_rotate(orig, angle=-row_angle_deg, reshape=True).
       scipy ndi.rotate는 내부적으로 출력→입력 affine을 다음 행렬로 적용:
         M = [[cos A, sin A], [-sin A, cos A]],  A = deg2rad(-row_angle_deg)
       따라서:
         y_orig - cy_orig = cos A·(y_rot - cy_rot) + sin A·(x_rot - cx_rot)
         x_orig - cx_orig = -sin A·(y_rot - cy_rot) + cos A·(x_rot - cx_rot)
    """
    A = np.deg2rad(-row_angle_deg)
    cos_A, sin_A = np.cos(A), np.sin(A)
    cy_rot = rotated_shape[0] / 2.0
    cx_rot = rotated_shape[1] / 2.0
    cy_orig = original_shape[0] / 2.0
    cx_orig = original_shape[1] / 2.0

    y_rel = rot_yx[:, 0] - cy_rot
    x_rel = rot_yx[:, 1] - cx_rot
    y_orig = cos_A * y_rel + sin_A * x_rel + cy_orig
    x_orig = -sin_A * y_rel + cos_A * x_rel + cx_orig
    return np.column_stack([y_orig, x_orig])


# -------- 픽셀 → 지리좌표 --------
def pixel_to_geo(yx: np.ndarray, transform) -> np.ndarray:
    """rasterio Affine transform 으로 (y, x) 픽셀 → (X, Y) 지리좌표."""
    ys = yx[:, 0] + 0.5
    xs = yx[:, 1] + 0.5
    Xs = transform.a * xs + transform.b * ys + transform.c
    Ys = transform.d * xs + transform.e * ys + transform.f
    return np.column_stack([Xs, Ys])
