"""
두둑 기반 후처리 — v4.npz의 콩잎 중심점 → 두둑 라인 역추적 → 고랑 FP 제거

흐름:
  1. 모든 콩잎 중심점에 PCA → dominant 방향 = 두둑 방향
  2. 두둑 수직축으로 투영 → 1D 밀도 프로파일 (histogram + smoothing)
  3. find_peaks (min 간격 30cm) → 두둑 위치 검출
  4. 각 잎의 최근접 두둑 거리 ≤ 15cm → keep, 초과 → drop (고랑 FP)

사용:
  python -u scripts/sam_ridge_filter_v4.py [npz_path]

출력:
  result/sam_test/GJSM-1-1_ridge_filter_v4.png       (4-panel 시각화)
  result/sam_test/GJSM-1-1_ridge_filter_v4_stats.txt
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
OUT_DIR = ROOT / "result" / "sam_test"
ROI_M2 = 30.0 * 30.0
STD_PER_HA = 76923

# ─── A. 두둑 검출 파라미터 (강화) ───
RIDGE_SPACING_MIN_CM = 20
RIDGE_KEEP_CM = 20
HIST_BIN_CM = 1.0
HIST_SMOOTH_SIGMA = 0.5      # 0.8 → 0.5 (더 sharp — 인접 봉우리 살림)
PEAK_HEIGHT_FRAC = 0.03      # 0.05 → 0.03 (낮은 봉우리도 인정)

# ─── B. Dedup 파라미터 ───
DEDUP_ENABLED = True
DEDUP_RADIUS_CM = 5.0        # 5cm 이내 중심점 → 1개로 병합

# ─── C. 크기 + 두둑 중심 거리 필터 (잡초 대응) ───
# ⚠️ 생육 초기 (잡초 미발생) → 두둑 위 = 모두 콩잎 가정 → C+D 비활성
CD_ENABLED = False
SIZE_MIN_CM2 = 2.0
SIZE_MAX_CM2 = 100.0
CENTER_TIGHT_CM = 12.0
SIZE_STRICT_MIN_CM2 = 3.0
SIZE_STRICT_MAX_CM2 = 60.0

# ─── 두둑 방향 검출 ───
MANUAL_ANGLE_DEG = None      # None = 각도 자동 스캔, 숫자 = 수동 override
ANGLE_SCAN_STEP_DEG = 1.0


def find_best_angle(centered: np.ndarray, gsd_m: float,
                    angle_range=(0.0, 180.0), step_deg: float = 1.0):
    """
    각도 스캔 (Radon 유사) — 두둑 수직축 밀도 프로파일이 가장 뾰족(peaky)한 각도 반환.
    peakiness metric = var(hist_smooth) / mean(hist_smooth)² (변동계수 제곱).
    """
    px_to_cm = gsd_m * 100
    bin_size_px = 1.0 / px_to_cm  # 1cm bins
    angles = np.arange(angle_range[0], angle_range[1], step_deg)
    scores = np.zeros(len(angles), dtype=float)
    for i, a_deg in enumerate(angles):
        a = np.radians(a_deg)
        # a_deg = 0 → 두둑 수평 → perp = (row axis) = (1, 0)
        # a_deg = 90 → 두둑 수직 → perp = (col axis) = (0, 1)
        perp = np.array([np.cos(a), np.sin(a)])  # (dy, dx)
        proj_a = centered @ perp
        edges = np.arange(proj_a.min(), proj_a.max() + bin_size_px, bin_size_px)
        hist, _ = np.histogram(proj_a, bins=edges)
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1.5)
        mean_h = hist_smooth.mean() + 1e-9
        scores[i] = hist_smooth.var() / (mean_h ** 2)
    best = int(np.argmax(scores))
    return angles[best], angles, scores


def compute_ridges(centroids: np.ndarray, gsd_m: float,
                    manual_angle_deg: float | None = None):
    """centroids: (N, 2) [y, x] in original pixel coords."""
    px_to_cm = gsd_m * 100
    mean = centroids.mean(axis=0)
    centered = centroids - mean

    # ─── 두둑 방향 검출 ───
    if manual_angle_deg is not None:
        ridge_angle_deg = float(manual_angle_deg)
        scan_angles, scan_scores = np.array([]), np.array([])
        print(f"[각도] 수동 override: {ridge_angle_deg:.1f}°")
    else:
        ridge_angle_deg, scan_angles, scan_scores = find_best_angle(
            centered, gsd_m, step_deg=ANGLE_SCAN_STEP_DEG)
        # 참고: PCA 결과도 비교
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        pca_dir = eigvecs[:, -1]
        pca_angle = np.degrees(np.arctan2(pca_dir[0], pca_dir[1]))
        print(f"[각도] 자동 스캔: {ridge_angle_deg:.1f}° (PCA 참조: {pca_angle:.1f}°)")

    a = np.radians(ridge_angle_deg)
    # find_best_angle의 perp 규약과 통일: perp = (cos(a), sin(a))
    # 이 perp에 수직인 ridge_dir = (-sin(a), cos(a))
    # a_deg=0 → perp=(1,0)=+y축, ridge=(0,1)=수평(x축)
    # a_deg=90 → perp=(0,1)=+x축, ridge=(-1,0)=수직
    # a_deg=61 → perp=(0.485,0.875), ridge=(-0.875,0.485) → 우상향(/)
    perp_dir = np.array([np.cos(a), np.sin(a)])
    ridge_dir = np.array([-np.sin(a), np.cos(a)])

    # perpendicular 축 투영 → 1D 밀도
    proj = centered @ perp_dir      # (N,) 수직 거리 (px)

    bin_size_px = HIST_BIN_CM / px_to_cm
    bins = np.arange(proj.min(), proj.max() + bin_size_px, bin_size_px)
    hist, edges = np.histogram(proj, bins=bins)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=HIST_SMOOTH_SIGMA)

    min_spacing_px = RIDGE_SPACING_MIN_CM / px_to_cm
    height_thresh = hist_smooth.max() * PEAK_HEIGHT_FRAC
    peaks, _ = find_peaks(hist_smooth, distance=min_spacing_px, height=height_thresh)
    ridge_proj_px = edges[peaks] + bin_size_px / 2  # bin 중심

    return {
        "mean": mean,
        "ridge_dir": ridge_dir,
        "perp_dir": perp_dir,
        "ridge_angle_deg": ridge_angle_deg,
        "proj": proj,
        "hist": hist,
        "hist_smooth": hist_smooth,
        "hist_edges": edges,
        "ridge_proj_px": ridge_proj_px,
        "px_to_cm": px_to_cm,
        "scan_angles": scan_angles,
        "scan_scores": scan_scores,
    }


def dedup_centroids(centroids: np.ndarray, areas: np.ndarray,
                     radius_px: float):
    """
    centroids: (N, 2) [y, x]
    areas: (N,) 각 잎 면적 (cm²)
    radius_px: 병합 반경 (픽셀)
    반환: (keep_idx, cluster_labels, n_clusters)
      - keep_idx: 클러스터당 대표 인덱스 (가장 큰 면적)
      - cluster_labels: 각 centroid의 클러스터 라벨 (0-based)
    """
    n = len(centroids)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=radius_px, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n), np.arange(n), n
    row = np.concatenate([pairs[:, 0], pairs[:, 1]])
    col = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(row), dtype=np.int8)
    graph = csr_matrix((data, (row, col)), shape=(n, n))
    n_clusters, labels = connected_components(graph, directed=False)
    keep_idx = np.empty(n_clusters, dtype=int)
    for c in range(n_clusters):
        cluster_members = np.where(labels == c)[0]
        keep_idx[c] = cluster_members[np.argmax(areas[cluster_members])]
    return keep_idx, labels, n_clusters


def filter_by_ridge(ridge_info: dict, keep_cm: float = RIDGE_KEEP_CM):
    proj = ridge_info["proj"]
    ridge_pos = ridge_info["ridge_proj_px"]
    if len(ridge_pos) == 0:
        return np.ones(len(proj), dtype=bool), np.full(len(proj), np.inf)
    dists = np.abs(proj[:, None] - ridge_pos[None, :]).min(axis=1)
    keep_thresh_px = keep_cm / ridge_info["px_to_cm"]
    keep = dists <= keep_thresh_px
    return keep, dists


def draw_ridge_lines(ax, ridge_info: dict, step: int, img_shape: tuple):
    mean = ridge_info["mean"]
    ridge_dir = ridge_info["ridge_dir"]
    perp_dir = ridge_info["perp_dir"]
    ridge_pos = ridge_info["ridge_proj_px"]
    H, W = img_shape[:2]

    ts = np.linspace(-10000, 10000, 2)
    for rp in ridge_pos:
        line_y = (mean[0] + rp * perp_dir[0] + ts * ridge_dir[0]) / step
        line_x = (mean[1] + rp * perp_dir[1] + ts * ridge_dir[1]) / step
        ax.plot(line_x, line_y, color="yellow", lw=0.6, alpha=0.7)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # 인자 처리:
    #   (1) 절대 경로 npz → 그대로 사용
    #   (2) 필지명 → 자동 검색: 우선 FULL_v4, 없으면 30m ROI v4
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        p = Path(arg)
        if p.exists():
            npz_path = p
        else:
            candidates = [
                OUT_DIR / f"{arg}_sam_FULL_v4.npz",
                OUT_DIR / f"{arg}_sam_roi_test_ds1_v4.npz",
            ]
            npz_path = next((c for c in candidates if c.exists()), candidates[-1])
    else:
        npz_path = OUT_DIR / "GJSM-1-1_sam_roi_test_ds1_v4.npz"
    stem = npz_path.stem
    field_name = stem.replace("_sam_FULL_v4", "").replace("_sam_roi_test_ds1_v4", "")
    is_full = "_sam_FULL_v4" in stem
    print(f"로드: {npz_path}  (모드: {'전체필지' if is_full else '30m ROI'})")
    d = np.load(npz_path)
    rgb = d["rgb_disp"]
    step = int(d["rgb_step"][0])
    leaves = d["leaf_arr"]
    gsd = float(d["gsd_ds"][0])
    n_all = int(d["n_all_masks"][0])
    # 전체 필지 npz 에서만 valid_area_ha 존재
    if "valid_area_ha" in d.files:
        valid_area_ha_override = float(d["valid_area_ha"][0])
    else:
        valid_area_ha_override = None
    print(f"  RGB {rgb.shape}, 콩잎 {len(leaves)}개, GSD={gsd*1000:.2f}mm"
          + (f", 유효면적 {valid_area_ha_override:.2f}ha" if valid_area_ha_override else ""))

    centroids = leaves[:, :2]   # (y, x) 원본 좌표
    xs_orig = leaves[:, 1]
    ys_orig = leaves[:, 0]

    ridge_info = compute_ridges(centroids, gsd, manual_angle_deg=MANUAL_ANGLE_DEG)
    keep, dists = filter_by_ridge(ridge_info)
    n_kept = int(keep.sum())
    n_drop = int((~keep).sum())

    # ─── B. Dedup — 두둑 위 통과한 잎들 중 5cm 이내 중심점 병합 ───
    ridge_kept_idx = np.where(keep)[0]           # 원본 인덱스
    ridge_kept_centroids = centroids[ridge_kept_idx]
    ridge_kept_areas = leaves[ridge_kept_idx, 2]  # area_cm2

    if DEDUP_ENABLED and len(ridge_kept_idx) > 0:
        dedup_radius_px = DEDUP_RADIUS_CM / (gsd * 100)
        rep_local_idx, cluster_labels, n_clusters = dedup_centroids(
            ridge_kept_centroids, ridge_kept_areas, dedup_radius_px)
        # 원본 leaves 인덱스로 변환
        final_idx = ridge_kept_idx[rep_local_idx]
        n_merged = len(ridge_kept_idx) - n_clusters
    else:
        final_idx = ridge_kept_idx
        n_clusters = len(ridge_kept_idx)
        n_merged = 0

    dedup_mask = np.zeros(len(leaves), dtype=bool)
    dedup_mask[final_idx] = True
    n_dedup = int(dedup_mask.sum())

    # ─── C+D. 크기 + 두둑 중심 거리 조건 필터 ───
    if CD_ENABLED and n_dedup > 0:
        areas = leaves[:, 2]                        # area_cm2
        dist_to_ridge_cm = dists * ridge_info["px_to_cm"]  # 두둑 중심까지 거리(cm)

        area_ok = (areas >= SIZE_MIN_CM2) & (areas <= SIZE_MAX_CM2)
        center_tight = dist_to_ridge_cm <= CENTER_TIGHT_CM
        area_strict = (areas >= SIZE_STRICT_MIN_CM2) & (areas <= SIZE_STRICT_MAX_CM2)

        # 중심 ≤10cm → 완화 조건 / 10-20cm → 엄격 조건
        cd_pass = np.where(center_tight, area_ok, area_ok & area_strict)
        final_mask = dedup_mask & cd_pass
        n_cd_drop = int(dedup_mask.sum() - final_mask.sum())
    else:
        final_mask = dedup_mask
        n_cd_drop = 0

    n_final = int(final_mask.sum())

    # 유효 면적: 전체 필지 npz → 알파 기반 valid_area_ha, 30m ROI → 900m²
    area_ha_calc = valid_area_ha_override if valid_area_ha_override is not None \
                    else (ROI_M2 / 10000)
    density_kept = n_kept / area_ha_calc
    rate_kept = density_kept / STD_PER_HA * 100
    density_dedup = n_dedup / area_ha_calc
    rate_dedup = density_dedup / STD_PER_HA * 100
    density_final = n_final / area_ha_calc
    rate_final = density_final / STD_PER_HA * 100
    density_orig = len(leaves) / area_ha_calc
    rate_orig = density_orig / STD_PER_HA * 100

    # 두둑 간격 통계
    ridge_pos = ridge_info["ridge_proj_px"]
    if len(ridge_pos) > 1:
        spacings_cm = np.diff(ridge_pos) * ridge_info["px_to_cm"]
        spacing_str = (f"평균 {spacings_cm.mean():.1f}cm  "
                       f"(min {spacings_cm.min():.1f}, max {spacings_cm.max():.1f})")
    else:
        spacing_str = "N/A"

    lines = []
    lines.append("=" * 60)
    lines.append("두둑 기반 후처리 필터 — v4")
    lines.append("=" * 60)
    lines.append(f"두둑 방향 (각도): {ridge_info['ridge_angle_deg']:.2f}°")
    lines.append(f"검출된 두둑 수:   {len(ridge_pos)}")
    lines.append(f"두둑 간격:        {spacing_str}")
    lines.append(f"유지 임계:        ±{RIDGE_KEEP_CM}cm")
    lines.append(f"Dedup 반경:      {DEDUP_RADIUS_CM}cm (활성: {DEDUP_ENABLED})")
    lines.append("")
    lines.append(f"원본 (v4):       {len(leaves):5d}개 → 밀도 {density_orig:>7,.0f}/ha → 입모율 {rate_orig:>6.1f}%")
    lines.append(f"A. 두둑 위:      {n_kept:5d}개 → 밀도 {density_kept:>7,.0f}/ha → 입모율 {rate_kept:>6.1f}%  (고랑 제거 {n_drop})")
    lines.append(f"B. + dedup:      {n_dedup:5d}개 → 밀도 {density_dedup:>7,.0f}/ha → 입모율 {rate_dedup:>6.1f}%  (병합 {n_merged}, {100*n_merged/max(n_kept,1):.1f}%)")
    lines.append(f"C+D. + 크기·거리:{n_final:5d}개 → 밀도 {density_final:>7,.0f}/ha → 입모율 {rate_final:>6.1f}%  (제거 {n_cd_drop})")
    lines.append(f"\n참고: 표준 100% 입모 = {STD_PER_HA:,}/ha")
    text = "\n".join(lines)
    print(text)

    (OUT_DIR / f"{field_name}_ridge_filter_v4_stats.txt").write_text(text, encoding="utf-8")

    # ─── 시각화 (top 3-panel + 3 diagnostic rows) ───
    fig = plt.figure(figsize=(22, 17))
    gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 1, 1])

    # (1) 원본 RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    ax1.set_title("(1) RGB ROI")
    ax1.axis("off")

    # (2) v4 원본 + 검출된 두둑 라인
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb)
    ax2.scatter(xs_orig / step, ys_orig / step, s=6,
                c="red", alpha=0.4, edgecolor="none")
    draw_ridge_lines(ax2, ridge_info, step, rgb.shape)
    ax2.set_title(f"(2) v4 원본 {len(leaves)}개 + 검출 두둑 {len(ridge_pos)}줄 "
                  f"({ridge_info['ridge_angle_deg']:.1f}°)")
    ax2.axis("off")

    # (3) 최종 — 4단계 시각화
    # 초록: 최종(A+B+C+D 모두 통과) / 노랑: C+D에서 컷된 잡초 후보
    # 주황: dedup에서 병합됨 / 회색: 고랑
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rgb)
    ax3.scatter(xs_orig[~keep] / step, ys_orig[~keep] / step, s=5,
                c="gray", alpha=0.25, edgecolor="none")
    dedup_merged_mask = np.zeros(len(leaves), dtype=bool)
    dedup_merged_mask[ridge_kept_idx] = True
    dedup_merged_mask &= ~dedup_mask
    ax3.scatter(xs_orig[dedup_merged_mask] / step,
                ys_orig[dedup_merged_mask] / step, s=5,
                c="orange", alpha=0.45, edgecolor="none")
    cd_dropped_mask = dedup_mask & ~final_mask
    ax3.scatter(xs_orig[cd_dropped_mask] / step,
                ys_orig[cd_dropped_mask] / step, s=7,
                c="yellow", alpha=0.75, edgecolor="none")
    ax3.scatter(xs_orig[final_mask] / step, ys_orig[final_mask] / step,
                s=10, c="lime", alpha=0.90, edgecolor="none")
    ax3.set_title(f"(3) 최종 {n_final}개 → 입모율 {rate_final:.1f}%\n"
                  f"    A→{n_kept}, B→{n_dedup}, C+D→{n_final}")
    ax3.text(0.01, 0.99, f"final = {n_final}",
             transform=ax3.transAxes, fontsize=13, fontweight="bold",
             color="white", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))
    ax3.axis("off")

    # (4) 각도 스캔 스코어 (자동 스캔일 때만)
    ax4 = fig.add_subplot(gs[1, :])
    if len(ridge_info["scan_angles"]) > 0:
        ax4.plot(ridge_info["scan_angles"], ridge_info["scan_scores"],
                 color="C2", lw=1.5)
        ax4.axvline(ridge_info["ridge_angle_deg"], color="red", lw=1.2,
                    label=f"best = {ridge_info['ridge_angle_deg']:.1f}°")
        ax4.set_xlabel("두둑 각도 후보 (°, 0=수평 / 90=수직)")
        ax4.set_ylabel("peakiness (CV²)")
        ax4.set_title("각도 스캔 — 밀도 프로파일이 가장 뾰족한 각도 = 두둑 방향")
        ax4.legend(loc="best")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, f"수동 각도: {ridge_info['ridge_angle_deg']:.1f}°",
                 ha="center", va="center", fontsize=14, transform=ax4.transAxes)
        ax4.axis("off")

    # (5) 1D 밀도 프로파일 (두둑 수직축)
    ax5 = fig.add_subplot(gs[2, :])
    edges = ridge_info["hist_edges"]
    hist = ridge_info["hist"]
    hist_smooth = ridge_info["hist_smooth"]
    centers_cm = (edges[:-1] + edges[1:]) / 2 * ridge_info["px_to_cm"]
    ax5.bar(centers_cm, hist, width=HIST_BIN_CM * 0.9, color="lightgray",
            edgecolor="none", label="원본 밀도")
    ax5.plot(centers_cm, hist_smooth, color="C0", lw=1.5, label="smoothed")
    for rp in ridge_pos:
        rp_cm = rp * ridge_info["px_to_cm"]
        ax5.axvline(rp_cm, color="orange", lw=1.0, alpha=0.7)
    ax5.set_xlabel("두둑 수직축 거리 (cm)")
    ax5.set_ylabel("잎 개수")
    ax5.set_title(f"두둑 수직 방향 밀도 프로파일 — 노란선 = 검출된 두둑 {len(ridge_pos)}개")
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)

    # (6) 진단: dedup 통과 후 잎들의 area 분포 + C+D 임계선
    ax6 = fig.add_subplot(gs[3, :])
    if n_dedup > 0:
        areas_dedup = leaves[dedup_mask, 2]
        areas_final = leaves[final_mask, 2]
        bins_a = np.linspace(0, 60, 61)  # 0-60cm², 1cm² bin
        ax6.hist(areas_dedup, bins=bins_a, color="lightblue",
                 edgecolor="C0", label=f"B 통과 (n={n_dedup})")
        ax6.hist(areas_final, bins=bins_a, color="lime",
                 alpha=0.55, edgecolor="green", label=f"최종 (n={n_final})")
        ax6.axvline(SIZE_MIN_CM2, color="red", ls="--", lw=1,
                    label=f"완화 min={SIZE_MIN_CM2}")
        ax6.axvline(SIZE_MAX_CM2, color="red", ls="--", lw=1,
                    label=f"완화 max={SIZE_MAX_CM2}")
        ax6.axvline(SIZE_STRICT_MIN_CM2, color="darkred", ls=":", lw=1,
                    label=f"엄격 min={SIZE_STRICT_MIN_CM2} (거리 10-20cm)")
        ax6.axvline(SIZE_STRICT_MAX_CM2, color="darkred", ls=":", lw=1,
                    label=f"엄격 max={SIZE_STRICT_MAX_CM2}")
        ax6.set_xlabel("잎 면적 (cm²)")
        ax6.set_ylabel("빈도")
        ax6.set_title("잎 면적 분포 — 파랑=dedup 후, 초록=최종 (C+D 통과), 빨강 점선=크기 컷")
        ax6.legend(loc="upper right", fontsize=9)
        ax6.grid(True, alpha=0.3)

    plt.suptitle(f"두둑 필터(A) + Dedup(B) + 크기·거리(C+D) v4 — {field_name}  "
                 f"[±{RIDGE_KEEP_CM}cm, dedup {DEDUP_RADIUS_CM}cm]  "
                 f"→ 최종 입모율 {rate_final:.1f}%",
                 fontsize=13)
    plt.tight_layout()
    out_png = OUT_DIR / f"{field_name}_ridge_filter_v4.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n저장: {out_png}")


if __name__ == "__main__":
    main()
