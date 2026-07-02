"""
Step 1 - 두둑/고랑 판별.

핵심 신호:
  * 각도 감지: 필지 mask 내부 gray intensity 의 perp 방향 평균 프로파일
    → 밴드패스(반복주기 60~130cm)의 var. 필지 사각형 편향 제거.
  * 두둑 위치: 최적 각도에서 gray 프로파일 + SAM 잎 밀도 결합 후 find_peaks.

흐름:
  1. rgb_disp 필지 mask (alpha 아닌 곳) 안에서 gray = 0.299R+0.587G+0.114B 계산.
  2. 각도 0~180° 스캔 (weighted histogram):
       - 각 각도의 perp 축으로 픽셀 사영, 각 bin 안 gray 평균 (weight=필지 mask).
       - 프로파일에 bandpass 적용 (60~130cm)
       - var(bandpass) = angle score. 정규화.
     SAM 잎도 같은 방식으로 각도 스캔 후 정규화하여 결합 (weight = 낮게).
  3. 최적 각도에서:
       - gray 프로파일 (필지 mask 내부) 정규화
       - SAM 밀도 프로파일 정규화
       - inverse gray + SAM (두둑 = 밝음 = 큰 gray, SAM = 잎 = 큰 밀도. 두 신호 동방향)
       - detrend (bandpass 유사)
       - find_peaks (min_dist = 15cm, height threshold)
  4. 인접 피크 간격 KMeans(k=2, initial 35/55) → 작은/큰 분류.
     * 큰 간격이 없거나 empty이면 median-split fallback.
  5. 그룹핑: 작은 간격 = 같은 두둑 안 인접 행 / 큰 간격 = 새 두둑.
  6. 두둑 record: {id, 행 위치, 폭, 방향축 min/max, 유형}.

사용:
  python -u scripts/pipeline/step1_ridges.py GJSM-1-1_Smart

출력:
  result/pipeline/{field}/ridges.npz         - 두둑 데이터
  result/pipeline/{field}/step1_ridges.png   - 4패널 시각화
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.vq import kmeans2

# Windows cp949 콘솔 회피
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.common import (
    load_field_npz, leaf_centroids, get_field_dir,
    angle_perp_ridge, setup_korean_font,
)


# ---- 각도 스캔 ----
ANGLE_STEP_DEG = 1.0
HIST_BIN_CM_ANGLE = 2.0          # 각도 스캔용 (거친 bin)
BANDPASS_PERIOD_MIN_CM = 60
BANDPASS_PERIOD_MAX_CM = 130
SAM_ANGLE_WEIGHT = 0.4           # SAM 각도 스코어 결합 가중 (필지 편향 없는 신호이지만 sparse)

# ---- 최종 프로파일 ----
# Step 1 목표 = 두둑 중심선 1개/두둑. dual-row 조간 신호(30cm 스케일)는 제거해야 함.
# → 최종 프로파일에도 진짜 bandpass (25~80cm) 적용.
HIST_BIN_CM_FINAL = 0.5
BANDPASS_LO_CM_FINAL = 25        # 25cm 이하 (조간) smooth로 제거
BANDPASS_HI_CM_FINAL = 80        # 80cm 이상 baseline 제거
PEAK_MIN_SPACING_CM = 40         # 두둑 사이 최소 거리 (좁은 스펙 65cm 반복 감안)
PEAK_HEIGHT_FRAC = 0.20          # 결합 프로파일에서 상대 높이

# ---- 두둑 그룹핑 ----
KMEANS_INIT_SMALL_CM = 35
KMEANS_INIT_LARGE_CM = 65
CLASSIFY_NARROW_MAX_CM = 45
CLASSIFY_WIDE_MIN_CM = 55


# ---- 유틸 ----
def rgb_to_gray_and_mask(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    rgb (H, W, 3) uint8.
    return:
      gray  (H, W) float32
      mask  (H, W) bool  - alpha 대신 (R,G,B)==(0,0,0) 검정 배경 제외
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mask = (r + g + b) > 5          # 완전 검정만 배경으로
    return gray, mask


def bandpass_1d(profile: np.ndarray, bin_cm: float,
                lo_cm: float, hi_cm: float) -> np.ndarray:
    """반복주기 lo~hi cm 를 통과. sigma_lo(제거)와 sigma_hi(smooth) 사용."""
    sigma_low = lo_cm / 2 / bin_cm      # 짧은 주기 smooth 제거
    sigma_high = hi_cm / 2 / bin_cm     # 긴 주기 baseline 제거
    smooth = gaussian_filter1d(profile, sigma=sigma_low)
    baseline = gaussian_filter1d(profile, sigma=sigma_high)
    return smooth - baseline


def weighted_profile(coords_flat: np.ndarray, weights_flat: np.ndarray,
                      mask_flat: np.ndarray, bin_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    coords_flat, weights_flat, mask_flat : 1D 같은 길이.
    mask_flat True 인 픽셀만 사용.
    return: (mean_per_bin, edges)
    """
    idx = mask_flat
    if idx.sum() == 0:
        return np.array([]), np.array([])
    c = coords_flat[idx]
    w = weights_flat[idx]
    lo, hi = c.min(), c.max()
    edges = np.arange(lo, hi + bin_size, bin_size)
    sum_w, _ = np.histogram(c, bins=edges, weights=w)
    count, _ = np.histogram(c, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_w = np.where(count > 0, sum_w / np.maximum(count, 1), 0.0)
    return mean_w.astype(np.float32), edges


def sam_density_profile(sam_pts: np.ndarray, perp: np.ndarray,
                         origin: np.ndarray, bin_size_px: float,
                         edges_ref: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """SAM 잎 사영 히스토그램. edges_ref 있으면 그것 사용."""
    proj = (sam_pts - origin) @ perp
    if edges_ref is None:
        lo, hi = proj.min(), proj.max()
        edges = np.arange(lo, hi + bin_size_px, bin_size_px)
    else:
        edges = edges_ref
    hist, _ = np.histogram(proj, bins=edges)
    return hist.astype(np.float32), edges


# ---- 각도 스캔 ----
def build_gray_coord_grid(gray: np.ndarray, mask: np.ndarray, rgb_step: int):
    """
    gray disp 좌표 (y_disp, x_disp) 를 원본 픽셀 좌표로 변환한 1D 배열 반환.
    """
    H, W = gray.shape
    ys_d, xs_d = np.mgrid[0:H, 0:W]
    ys_orig = (ys_d * rgb_step).astype(np.float32)
    xs_orig = (xs_d * rgb_step).astype(np.float32)
    coords = np.stack([ys_orig.ravel(), xs_orig.ravel()], axis=1)  # (N, 2)
    return coords, gray.ravel(), mask.ravel()


def scan_angles(gray_coords: np.ndarray, gray_vals: np.ndarray, gray_mask: np.ndarray,
                sam_pts: np.ndarray, gsd_m: float) -> dict:
    px_to_cm = gsd_m * 100
    bin_size_px = HIST_BIN_CM_ANGLE / px_to_cm
    angles = np.arange(0, 180, ANGLE_STEP_DEG)
    gray_scores = np.zeros(len(angles))
    sam_scores = np.zeros(len(angles))

    # 중심화 (필지 mask 안 픽셀 mean)
    gray_center = gray_coords[gray_mask].mean(axis=0)
    gray_coords_c = gray_coords - gray_center
    sam_center_v = sam_pts.mean(axis=0)
    sam_c = sam_pts - sam_center_v

    for i, a_deg in enumerate(angles):
        perp, _ = angle_perp_ridge(a_deg)

        # gray weighted profile
        proj_g = gray_coords_c @ perp
        mean_prof, _ = weighted_profile(proj_g, gray_vals, gray_mask, bin_size_px)
        if len(mean_prof) > 10:
            bp = bandpass_1d(mean_prof, HIST_BIN_CM_ANGLE,
                             BANDPASS_PERIOD_MIN_CM, BANDPASS_PERIOD_MAX_CM)
            gray_scores[i] = float(np.var(bp))
        else:
            gray_scores[i] = 0.0

        # SAM 잎 밀도 프로파일 (기본 histogram)
        proj_s = sam_c @ perp
        edges = np.arange(proj_s.min(), proj_s.max() + bin_size_px, bin_size_px)
        hist, _ = np.histogram(proj_s, bins=edges)
        if len(hist) > 10:
            bp = bandpass_1d(hist.astype(float), HIST_BIN_CM_ANGLE,
                             BANDPASS_PERIOD_MIN_CM, BANDPASS_PERIOD_MAX_CM)
            sam_scores[i] = float(np.var(bp))

    gray_n = gray_scores / (gray_scores.max() + 1e-9)
    sam_n = sam_scores / (sam_scores.max() + 1e-9)
    combined = gray_n + SAM_ANGLE_WEIGHT * sam_n
    best_idx = int(np.argmax(combined))
    return dict(
        angles=angles, gray_scores=gray_scores, sam_scores=sam_scores,
        gray_norm=gray_n, sam_norm=sam_n, combined=combined,
        best_angle_deg=float(angles[best_idx]),
        gray_best_angle_deg=float(angles[int(np.argmax(gray_scores))]),
        sam_best_angle_deg=float(angles[int(np.argmax(sam_scores))]),
        gray_center=gray_center, sam_center=sam_center_v,
    )


# ---- 최종 프로파일 ----
def build_final_profile(gray_coords: np.ndarray, gray_vals: np.ndarray, gray_mask: np.ndarray,
                         sam_pts: np.ndarray, a_deg: float, gsd_m: float) -> dict:
    px_to_cm = gsd_m * 100
    bin_size_px = HIST_BIN_CM_FINAL / px_to_cm
    perp, ridge = angle_perp_ridge(a_deg)

    # 원점: 필지 안 픽셀 mean (안정적)
    origin = gray_coords[gray_mask].mean(axis=0)

    # gray weighted profile
    proj_g = (gray_coords - origin) @ perp
    gray_mean, edges = weighted_profile(proj_g, gray_vals, gray_mask, bin_size_px)
    if len(edges) == 0:
        return {}
    centers_px = (edges[:-1] + edges[1:]) / 2
    centers_cm = centers_px * px_to_cm

    # SAM profile (같은 edges) + bandpass
    proj_s = (sam_pts - origin) @ perp
    sam_hist, _ = np.histogram(proj_s, bins=edges)
    sam_hist = sam_hist.astype(np.float32)
    sam_bp = bandpass_1d(sam_hist, HIST_BIN_CM_FINAL,
                          BANDPASS_LO_CM_FINAL, BANDPASS_HI_CM_FINAL)
    sam_smooth = gaussian_filter1d(sam_hist,
                                    sigma=BANDPASS_LO_CM_FINAL / 2 / HIST_BIN_CM_FINAL)

    # gray bandpass
    gray_bp = bandpass_1d(gray_mean, HIST_BIN_CM_FINAL,
                           BANDPASS_LO_CM_FINAL, BANDPASS_HI_CM_FINAL)
    gray_smooth = gaussian_filter1d(gray_mean,
                                     sigma=BANDPASS_LO_CM_FINAL / 2 / HIST_BIN_CM_FINAL)

    # 부호: 밝은 stripe (두둑) 에서 gray_bp 큰 값. 그대로 사용.
    # 정규화 (0~1)
    def norm01(x):
        x = np.asarray(x, dtype=np.float32)
        lo, hi = np.percentile(x, [1, 99])
        if hi - lo < 1e-6:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0, 1)

    gray_norm = norm01(gray_bp)
    sam_norm = norm01(sam_bp)
    combined = (gray_norm + sam_norm) / 2

    return dict(
        origin=origin, perp=perp, ridge=ridge,
        edges=edges, centers_px=centers_px, centers_cm=centers_cm,
        gray_mean=gray_mean, gray_smooth=gray_smooth,
        gray_bp=gray_bp, gray_norm=gray_norm,
        sam_hist=sam_hist, sam_smooth=sam_smooth, sam_norm=sam_norm,
        combined=combined, px_to_cm=px_to_cm,
    )


def detect_row_peaks(profile: dict) -> np.ndarray:
    combined = profile["combined"]
    px_to_cm = profile["px_to_cm"]
    centers_px = profile["centers_px"]
    min_dist_px = PEAK_MIN_SPACING_CM / px_to_cm
    height = combined.max() * PEAK_HEIGHT_FRAC
    peaks, _ = find_peaks(combined, distance=min_dist_px, height=height)
    return centers_px[peaks]


# ---- 두둑 그룹핑 ----
def group_rows_into_ridges(row_pos_px: np.ndarray, px_to_cm: float) -> dict:
    if len(row_pos_px) < 2:
        return dict(
            row_pos_sorted_px=np.asarray(row_pos_px, dtype=np.float32),
            ridge_id_per_row=np.zeros(len(row_pos_px), dtype=int),
            row_id_within_ridge=np.zeros(len(row_pos_px), dtype=int),
            gaps_cm=np.array([]),
            is_small=np.array([], dtype=bool), is_large=np.array([], dtype=bool),
            small_center_cm=np.nan, large_center_cm=np.nan,
            small_gaps_cm=np.array([]), large_gaps_cm=np.array([]),
        )

    row_pos_sorted = np.sort(row_pos_px)
    gaps_px = np.diff(row_pos_sorted)
    gaps_cm = gaps_px * px_to_cm

    # kmeans2 (초기 = 스펙)
    init = np.array([[KMEANS_INIT_SMALL_CM], [KMEANS_INIT_LARGE_CM]])
    try:
        centers, labels = kmeans2(gaps_cm.reshape(-1, 1).astype(np.float64),
                                   init, minit="matrix", seed=0)
        centers = centers.flatten()
        # 재정렬
        order = np.argsort(centers)
        small_center = float(centers[order[0]])
        large_center = float(centers[order[1]])
        is_small = labels == order[0]
        is_large = ~is_small
        # empty cluster fallback: median split
        if is_small.sum() == 0 or is_large.sum() == 0:
            raise ValueError("empty cluster")
    except Exception:
        med = float(np.median(gaps_cm))
        is_small = gaps_cm <= med
        is_large = ~is_small
        small_center = float(np.median(gaps_cm[is_small])) if is_small.any() else np.nan
        large_center = float(np.median(gaps_cm[is_large])) if is_large.any() else np.nan

    ridge_id = np.zeros(len(row_pos_sorted), dtype=int)
    row_id_within = np.zeros(len(row_pos_sorted), dtype=int)
    rid = 0
    for i in range(1, len(row_pos_sorted)):
        if is_large[i - 1]:
            rid += 1
            row_id_within[i] = 0
        else:
            row_id_within[i] = row_id_within[i - 1] + 1
        ridge_id[i] = rid

    return dict(
        row_pos_sorted_px=row_pos_sorted,
        ridge_id_per_row=ridge_id,
        row_id_within_ridge=row_id_within,
        gaps_cm=gaps_cm,
        is_small=is_small, is_large=is_large,
        small_center_cm=small_center, large_center_cm=large_center,
        small_gaps_cm=gaps_cm[is_small],
        large_gaps_cm=gaps_cm[is_large],
    )


def build_ridge_records(row_pos_sorted_px: np.ndarray, ridge_id_per_row: np.ndarray,
                         profile: dict, sam_pts: np.ndarray,
                         gray_coords: np.ndarray, gray_mask: np.ndarray) -> list[dict]:
    px_to_cm = profile["px_to_cm"]
    origin = profile["origin"]
    perp = profile["perp"]
    ridge_dir = profile["ridge"]

    # 필지 픽셀 좌표 (perp/ridge)
    fp = gray_coords[gray_mask] - origin
    fp_perp = fp @ perp
    fp_ridge = fp @ ridge_dir

    sam_rel = sam_pts - origin
    sam_perp = sam_rel @ perp
    sam_ridge = sam_rel @ ridge_dir

    ridges = []
    unique_rids = np.unique(ridge_id_per_row)
    for rid in unique_rids:
        rows_mask = ridge_id_per_row == rid
        rows_px = row_pos_sorted_px[rows_mask]
        n_rows = len(rows_px)
        center_perp_px = float(rows_px.mean())
        width_px = float(rows_px.max() - rows_px.min()) if n_rows > 1 else 0.0
        width_cm = width_px * px_to_cm

        # band half = 조간의 절반 + margin (15cm)
        band_half_cm = max(width_cm / 2 + 15, 20)
        band_half_px = band_half_cm / px_to_cm

        # 두둑 유형
        if n_rows < 2:
            rtype = "single"
        elif width_cm <= CLASSIFY_NARROW_MAX_CM:
            rtype = "narrow"
        elif width_cm >= CLASSIFY_WIDE_MIN_CM:
            rtype = "wide"
        else:
            rtype = "mid"

        # 길이 = 필지 픽셀 안에서 밴드 내 ridge_axis min/max (percentile 안전)
        in_band = np.abs(fp_perp - center_perp_px) <= band_half_px
        if in_band.sum() > 10:
            ridge_min = float(np.percentile(fp_ridge[in_band], 1))
            ridge_max = float(np.percentile(fp_ridge[in_band], 99))
        else:
            ridge_min, ridge_max = 0.0, 0.0

        # 밴드 안 SAM 잎 개수
        in_band_sam = np.abs(sam_perp - center_perp_px) <= band_half_px
        n_leaves = int(in_band_sam.sum())

        ridges.append(dict(
            ridge_id=int(rid), n_rows=int(n_rows),
            row_pos_perp_px=rows_px.tolist(),
            center_perp_px=center_perp_px,
            width_px=width_px, width_cm=width_cm,
            ridge_type=rtype, band_half_cm=float(band_half_cm),
            ridge_min_px=ridge_min, ridge_max_px=ridge_max,
            length_cm=(ridge_max - ridge_min) * px_to_cm,
            n_leaves=n_leaves,
        ))
    return ridges


# ---- 시각화 ----
def draw_ridge_polygon_on_ax(ax, ridge: dict, profile: dict, rgb_step: int, color):
    origin = profile["origin"]; perp = profile["perp"]; ridge_dir = profile["ridge"]
    band_half_px = ridge["band_half_cm"] / profile["px_to_cm"]
    corners = []
    for r_off, p_off in [
        (ridge["ridge_min_px"], -band_half_px),
        (ridge["ridge_max_px"], -band_half_px),
        (ridge["ridge_max_px"], +band_half_px),
        (ridge["ridge_min_px"], +band_half_px),
    ]:
        pt = origin + r_off * ridge_dir + (ridge["center_perp_px"] + p_off) * perp
        corners.append(pt)
    corners = np.array(corners) / rgb_step
    poly = mpatches.Polygon(corners[:, ::-1], closed=True,
                             edgecolor=color, facecolor=color,
                             alpha=0.20, linewidth=0.8)
    ax.add_patch(poly)


def draw_row_line_on_ax(ax, ridge: dict, row_perp_px: float, profile: dict,
                         rgb_step: int, color="red"):
    origin = profile["origin"]; perp = profile["perp"]; ridge_dir = profile["ridge"]
    p1 = origin + ridge["ridge_min_px"] * ridge_dir + row_perp_px * perp
    p2 = origin + ridge["ridge_max_px"] * ridge_dir + row_perp_px * perp
    p1d = p1 / rgb_step; p2d = p2 / rgb_step
    ax.plot([p1d[1], p2d[1]], [p1d[0], p2d[0]], color=color, lw=0.6, alpha=0.85)


def visualize(res: dict, out_png: Path):
    setup_korean_font()
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 1.0, 1.0])

    rgb = res["rgb_disp"]; rgb_step = res["rgb_step"]
    profile = res["profile"]; ridges = res["ridges"]
    sam_pts = res["sam_pts"]; scan = res["scan"]
    row_pos = res["row_pos_sorted_px"]
    group = res["group"]

    # (1) RGB + 두둑 폴리곤 + 행 라인
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    if len(sam_pts):
        ax1.scatter(sam_pts[:, 1] / rgb_step, sam_pts[:, 0] / rgb_step,
                    s=1, c="lime", alpha=0.30, edgecolor="none")
    n_r = max(len(ridges), 1)
    cmap = plt.cm.tab20(np.arange(n_r) % 20)
    for r in ridges:
        c = cmap[r["ridge_id"] % 20]
        draw_ridge_polygon_on_ax(ax1, r, profile, rgb_step, color=c)
        for row_p in r["row_pos_perp_px"]:
            draw_row_line_on_ax(ax1, r, row_p, profile, rgb_step, color="red")
        # ID 라벨 (7개마다 표기 - 과밀 방지)
        if r["ridge_id"] % 5 == 0:
            cy = (r["ridge_min_px"] + r["ridge_max_px"]) / 2
            pt = profile["origin"] + cy * profile["ridge"] + r["center_perp_px"] * profile["perp"]
            pt_d = pt / rgb_step
            ax1.text(pt_d[1], pt_d[0], str(r["ridge_id"]),
                     color="white", fontsize=7, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.10", facecolor="black", alpha=0.55))
    n_n = sum(r["ridge_type"] == "narrow" for r in ridges)
    n_w = sum(r["ridge_type"] == "wide" for r in ridges)
    n_m = sum(r["ridge_type"] == "mid" for r in ridges)
    n_s = sum(r["ridge_type"] == "single" for r in ridges)
    ax1.set_title(
        f"두둑 검출 - 각도 {scan['best_angle_deg']:.1f}°, "
        f"두둑 {len(ridges)}개 (narrow/wide/mid/single = "
        f"{n_n}/{n_w}/{n_m}/{n_s})",
        fontsize=13, fontweight="bold")
    ax1.axis("off")

    # (2) 통계 요약
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    small_med = np.median(group['small_gaps_cm']) if len(group['small_gaps_cm']) else np.nan
    large_med = np.median(group['large_gaps_cm']) if len(group['large_gaps_cm']) else np.nan
    lines = [
        f"필지: {res['field']}",
        f"검출 잎 (SAM): {len(sam_pts):,}",
        f"두둑 방향 각도: {scan['best_angle_deg']:.1f}°",
        f"   Gray 밴드패스 최적: {scan['gray_best_angle_deg']:.1f}°",
        f"   SAM 밀도 최적: {scan['sam_best_angle_deg']:.1f}°",
        "",
        f"검출된 행 (row) 수: {len(row_pos)}",
        f"검출된 두둑 수: {len(ridges)}",
        f"   좁은 두둑 (<={CLASSIFY_NARROW_MAX_CM}cm): {n_n}",
        f"   넓은 두둑 (>={CLASSIFY_WIDE_MIN_CM}cm): {n_w}",
        f"   중간 두둑: {n_m}",
        f"   단일 행 두둑: {n_s}",
        "",
        f"KMeans 간격 클러스터:",
        f"   작은 (조간, 두둑 내) median: "
        + (f"{small_med:.1f}cm  (n={len(group['small_gaps_cm'])})"
            if not np.isnan(small_med) else "N/A"),
        f"   큰   (두둑 사이) median: "
        + (f"{large_med:.1f}cm  (n={len(group['large_gaps_cm'])})"
            if not np.isnan(large_med) else "N/A"),
    ]
    ax2.text(0.02, 0.98, "\n".join(lines), fontsize=12,
             va="top", ha="left", transform=ax2.transAxes)

    # (3) 각도 스캔
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(scan["angles"], scan["gray_norm"], color="C1",
             lw=1.4, label=f"Gray 밴드패스 var (정규화) - best {scan['gray_best_angle_deg']:.0f}°")
    ax3.plot(scan["angles"], scan["sam_norm"], color="C2",
             lw=1.0, alpha=0.7, label=f"SAM 밴드패스 var (정규화) - best {scan['sam_best_angle_deg']:.0f}°")
    ax3.plot(scan["angles"], scan["combined"] / scan["combined"].max(),
             color="C0", lw=1.6,
             label=f"결합 (SAM w={SAM_ANGLE_WEIGHT}) - best {scan['best_angle_deg']:.0f}°")
    ax3.axvline(scan["best_angle_deg"], color="red", lw=1.2, alpha=0.7)
    ax3.set_xlabel("두둑 후보 각도 (°)"); ax3.set_ylabel("정규화 var")
    ax3.set_title("각도 스캔 - Gray 밴드패스 var 최대 = 두둑 방향")
    ax3.legend(loc="upper right"); ax3.grid(alpha=0.3)

    # (4) 최종 프로파일 + 행 피크 + 두둑 밴드
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(profile["centers_cm"], profile["gray_norm"], color="C1",
             alpha=0.75, label="Gray 밴드패스 (norm)")
    ax4.plot(profile["centers_cm"], profile["sam_norm"], color="C2",
             alpha=0.75, label="SAM 밀도 (norm)")
    ax4.plot(profile["centers_cm"], profile["combined"], color="C0",
             lw=1.5, label="결합")
    for row_p_px in row_pos:
        ax4.axvline(row_p_px * profile["px_to_cm"], color="red", lw=0.4, alpha=0.5)
    for r in ridges:
        cx_cm = r["center_perp_px"] * profile["px_to_cm"]
        w_cm = r["band_half_cm"]
        c = cmap[r["ridge_id"] % 20]
        ax4.axvspan(cx_cm - w_cm, cx_cm + w_cm, color=c, alpha=0.12)
    ax4.set_xlabel("두둑 수직축 (cm)"); ax4.set_ylabel("정규화 신호")
    ax4.set_title(f"결합 1D 프로파일 - 빨간선 = 검출된 행 {len(row_pos)}개, "
                  f"색밴드 = 두둑 {len(ridges)}개")
    ax4.legend(loc="upper right"); ax4.grid(alpha=0.3)

    plt.suptitle(f"Step 1: 두둑/고랑 판별 - {res['field']}",
                 fontsize=15, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


# ---- 메인 ----
def main():
    if len(sys.argv) < 2:
        print("사용: python step1_ridges.py <field_name>")
        sys.exit(1)
    field = sys.argv[1]
    t0 = time.time()

    print(f"[{field}] npz 로드")
    d = load_field_npz(field)
    sam_pts = leaf_centroids(d["leaves"])
    print(f"   SAM 잎 {len(sam_pts):,}개, GSD {d['gsd_m']*1000:.2f}mm, "
          f"rgb_disp {d['rgb_disp'].shape}, step {d['rgb_step']}")

    print(f"[{field}] Gray + 필지 mask 준비")
    gray, mask = rgb_to_gray_and_mask(d["rgb_disp"])
    print(f"   필지 픽셀 {int(mask.sum()):,} / {mask.size:,} ({mask.mean()*100:.1f}%)")

    print(f"[{field}] 좌표 그리드 생성")
    gray_coords, gray_vals, gray_mask = build_gray_coord_grid(gray, mask, d["rgb_step"])

    print(f"[{field}] 각도 스캔 (0~180°, {ANGLE_STEP_DEG}° step, "
          f"bandpass {BANDPASS_PERIOD_MIN_CM}~{BANDPASS_PERIOD_MAX_CM}cm)")
    scan = scan_angles(gray_coords, gray_vals, gray_mask, sam_pts, d["gsd_m"])
    print(f"   Gray 최적: {scan['gray_best_angle_deg']:.1f}°  "
          f"SAM 최적: {scan['sam_best_angle_deg']:.1f}°  "
          f"결합 최적: {scan['best_angle_deg']:.1f}°")

    print(f"[{field}] 최종 프로파일 (bin {HIST_BIN_CM_FINAL}cm)")
    profile = build_final_profile(gray_coords, gray_vals, gray_mask,
                                    sam_pts, scan["best_angle_deg"], d["gsd_m"])
    row_pos_px = detect_row_peaks(profile)
    print(f"   검출 행 {len(row_pos_px)}개")

    print(f"[{field}] 행 그룹핑")
    group = group_rows_into_ridges(row_pos_px, profile["px_to_cm"])
    ridges = build_ridge_records(group["row_pos_sorted_px"],
                                  group["ridge_id_per_row"],
                                  profile, sam_pts, gray_coords, gray_mask)
    n_n = sum(r["ridge_type"] == "narrow" for r in ridges)
    n_w = sum(r["ridge_type"] == "wide" for r in ridges)
    n_m = sum(r["ridge_type"] == "mid" for r in ridges)
    n_s = sum(r["ridge_type"] == "single" for r in ridges)
    print(f"   두둑 {len(ridges)}개 - narrow {n_n}, wide {n_w}, mid {n_m}, single {n_s}")
    if len(group["small_gaps_cm"]):
        print(f"   조간 median {np.median(group['small_gaps_cm']):.1f}cm "
              f"(n={len(group['small_gaps_cm'])})")
    if len(group["large_gaps_cm"]):
        print(f"   두둑 간 median {np.median(group['large_gaps_cm']):.1f}cm "
              f"(n={len(group['large_gaps_cm'])})")

    field_dir = get_field_dir(field)
    out_npz = field_dir / "ridges.npz"
    out_png = field_dir / "step1_ridges.png"

    ridge_arr = np.array([[
        r["ridge_id"], r["n_rows"], r["center_perp_px"], r["width_px"],
        r["width_cm"], r["band_half_cm"],
        r["ridge_min_px"], r["ridge_max_px"], r["length_cm"], r["n_leaves"],
    ] for r in ridges], dtype=np.float32) if ridges \
        else np.zeros((0, 10), dtype=np.float32)
    ridge_types = np.array([r["ridge_type"] for r in ridges])
    row_pos_per_ridge = [np.asarray(r["row_pos_perp_px"], dtype=np.float32) for r in ridges]

    np.savez_compressed(
        out_npz,
        best_angle_deg=np.array([scan["best_angle_deg"]]),
        origin=profile["origin"], perp=profile["perp"], ridge_dir=profile["ridge"],
        px_to_cm=np.array([profile["px_to_cm"]]),
        row_pos_sorted_px=group["row_pos_sorted_px"],
        ridge_id_per_row=group["ridge_id_per_row"],
        ridge_arr=ridge_arr, ridge_types=ridge_types,
        row_pos_per_ridge=np.asarray(row_pos_per_ridge, dtype=object),
        small_center_cm=np.array([group["small_center_cm"]]),
        large_center_cm=np.array([group["large_center_cm"]]),
    )
    print(f"[{field}] 저장: {out_npz.name}")

    visualize({
        "field": field, "rgb_disp": d["rgb_disp"], "rgb_step": d["rgb_step"],
        "sam_pts": sam_pts, "scan": scan, "profile": profile,
        "row_pos_sorted_px": group["row_pos_sorted_px"],
        "ridges": ridges, "group": group,
    }, out_png)
    print(f"[{field}] 저장: {out_png.name}")
    print(f"[{field}] 완료 ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
