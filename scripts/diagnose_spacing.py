"""
데이터 기반 dedup 임계 진단 스크립트.

목적: 두둑 위 잎들의 인접 간격 분포에서 bimodal 봉우리를 찾아
      자동 dedup 임계값 검출 + 실측 주간 간격 리포트.

흐름:
  1. 6필지 30m ROI npz 로드
  2. 각 필지에서 두둑 필터 (A) 적용 — 두둑 위 잎만 유지
  3. 두둑별로 잎을 두둑 방향으로 정렬 → 인접 간격 계산
  4. 필지 전체 인접 간격 히스토그램 → bimodal 봉우리 검출
  5. 두 봉우리 사이 valley = 자동 dedup 임계값
  6. 자동 dedup 적용 후:
     - 개체 수 (dedup 후 남은 잎)
     - 실측 주간 간격 median/mean
  7. 5cm 고정 dedup 결과와 비교

출력:
  result/sam_test/spacing_diag/{field}_hist.png    (필지별 히스토그램 + 봉우리·임계)
  result/sam_test/spacing_diag/comparison.png      (6필지 비교)
  result/sam_test/spacing_diag/summary.md          (마크다운 요약표)
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
OUT_DIR = ROOT / "result" / "sam_test"
DIAG_DIR = OUT_DIR / "spacing_diag"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["GJSM-1-1_Smart", "GJSM-1-1_normal",
          "GJSM-1-2", "GJSM-1-3", "GJSM-2-2", "GJSM-2-3"]

# 필터 파라미터 (기존과 동일)
RIDGE_SPACING_MIN_CM = 20
RIDGE_KEEP_CM = 20
HIST_BIN_CM = 1.0
HIST_SMOOTH_SIGMA = 0.5
PEAK_HEIGHT_FRAC = 0.03

# 인접 간격 히스토그램 설정
GAP_HIST_MAX_CM = 40.0        # 40cm까지 관찰
GAP_HIST_BIN_CM = 0.5         # 0.5cm bin
GAP_SMOOTH_SIGMA = 1.5

# 봉우리 검출 범위
SAME_LEAF_MAX_CM = 8.0        # 같은 잎 조각 봉우리는 이 안에서만 찾음
REAL_GAP_MIN_CM = 10.0        # 실제 간격 봉우리는 이 이후


def find_best_angle(centered, gsd_m, step_deg=1.0):
    px_to_cm = gsd_m * 100
    bin_size_px = 1.0 / px_to_cm
    angles = np.arange(0, 180, step_deg)
    scores = np.zeros(len(angles))
    for i, a_deg in enumerate(angles):
        a = np.radians(a_deg)
        perp = np.array([np.cos(a), np.sin(a)])
        proj_a = centered @ perp
        edges = np.arange(proj_a.min(), proj_a.max() + bin_size_px, bin_size_px)
        hist, _ = np.histogram(proj_a, bins=edges)
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1.5)
        m = hist_smooth.mean() + 1e-9
        scores[i] = hist_smooth.var() / (m ** 2)
    return angles[int(np.argmax(scores))]


def compute_ridges(centroids, gsd_m):
    px_to_cm = gsd_m * 100
    mean = centroids.mean(axis=0)
    centered = centroids - mean
    ridge_angle_deg = find_best_angle(centered, gsd_m)
    a = np.radians(ridge_angle_deg)
    perp_dir = np.array([np.cos(a), np.sin(a)])
    ridge_dir = np.array([-np.sin(a), np.cos(a)])
    proj_perp = centered @ perp_dir
    proj_par = centered @ ridge_dir
    bin_size_px = HIST_BIN_CM / px_to_cm
    edges = np.arange(proj_perp.min(), proj_perp.max() + bin_size_px, bin_size_px)
    hist, _ = np.histogram(proj_perp, bins=edges)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=HIST_SMOOTH_SIGMA)
    min_spacing_px = RIDGE_SPACING_MIN_CM / px_to_cm
    peaks, _ = find_peaks(hist_smooth,
                          distance=min_spacing_px,
                          height=hist_smooth.max() * PEAK_HEIGHT_FRAC)
    ridge_pos_px = edges[peaks] + bin_size_px / 2
    return dict(mean=mean, ridge_angle_deg=ridge_angle_deg,
                perp_dir=perp_dir, ridge_dir=ridge_dir,
                proj_perp=proj_perp, proj_par=proj_par,
                ridge_pos_px=ridge_pos_px, px_to_cm=px_to_cm,
                centered=centered)


def compute_adjacent_gaps_per_ridge(ridge_info, keep_thresh_px):
    """두둑별로 잎을 두둑 방향으로 정렬 → 인접 간격(cm) 반환."""
    rp = ridge_info["ridge_pos_px"]
    if len(rp) == 0:
        return np.array([])
    proj_perp = ridge_info["proj_perp"]
    proj_par = ridge_info["proj_par"]
    px_to_cm = ridge_info["px_to_cm"]
    all_gaps = []
    for r in rp:
        on_ridge = np.abs(proj_perp - r) <= keep_thresh_px
        if on_ridge.sum() < 2:
            continue
        along = np.sort(proj_par[on_ridge])
        gaps_cm = np.diff(along) * px_to_cm
        all_gaps.extend(gaps_cm.tolist())
    return np.array(all_gaps)


def detect_dedup_threshold(gaps_cm: np.ndarray):
    """
    인접 간격 히스토그램에서 bimodal 봉우리 → valley 위치 = dedup 임계.
    반환: (threshold_cm, peak1_cm, peak2_cm, valley_min, hist, edges, hist_smooth)
    """
    gaps = gaps_cm[(gaps_cm > 0) & (gaps_cm <= GAP_HIST_MAX_CM)]
    if len(gaps) < 10:
        return dict(threshold_cm=None, peak1_cm=None, peak2_cm=None,
                    hist=np.array([]), edges=np.array([]), hist_smooth=np.array([]),
                    n_gaps=int(len(gaps)))
    edges = np.arange(0, GAP_HIST_MAX_CM + GAP_HIST_BIN_CM, GAP_HIST_BIN_CM)
    hist, _ = np.histogram(gaps, bins=edges)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=GAP_SMOOTH_SIGMA)
    centers = (edges[:-1] + edges[1:]) / 2

    # 같은-잎 봉우리 (0-SAME_LEAF_MAX_CM)
    mask_same = centers <= SAME_LEAF_MAX_CM
    peaks_same, _ = find_peaks(hist_smooth[mask_same])
    peak1_idx = None
    if len(peaks_same) > 0:
        peak1_idx = int(peaks_same[np.argmax(hist_smooth[mask_same][peaks_same])])
    # 실제 간격 봉우리 (REAL_GAP_MIN_CM 이후)
    mask_real = centers >= REAL_GAP_MIN_CM
    real_idx_offset = int(mask_real.argmax())
    peaks_real, _ = find_peaks(hist_smooth[mask_real])
    peak2_idx = None
    if len(peaks_real) > 0:
        peak2_idx = int(peaks_real[np.argmax(hist_smooth[mask_real][peaks_real])]) \
                    + real_idx_offset

    peak1_cm = float(centers[peak1_idx]) if peak1_idx is not None else None
    peak2_cm = float(centers[peak2_idx]) if peak2_idx is not None else None

    # Valley (두 봉우리 사이 최소)
    if peak1_idx is not None and peak2_idx is not None and peak1_idx < peak2_idx:
        valley_slice = hist_smooth[peak1_idx:peak2_idx + 1]
        valley_offset = int(np.argmin(valley_slice))
        threshold_idx = peak1_idx + valley_offset
        threshold_cm = float(centers[threshold_idx])
    else:
        threshold_cm = None

    return dict(threshold_cm=threshold_cm, peak1_cm=peak1_cm, peak2_cm=peak2_cm,
                hist=hist, edges=edges, hist_smooth=hist_smooth,
                n_gaps=int(len(gaps)))


def dedup_by_radius(centroids, areas, radius_px):
    n = len(centroids)
    if n == 0 or radius_px <= 0:
        return np.arange(n)
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=radius_px, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n)
    row = np.concatenate([pairs[:, 0], pairs[:, 1]])
    col = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = csr_matrix((np.ones(len(row), np.int8), (row, col)), shape=(n, n))
    n_clusters, labels = connected_components(graph, directed=False)
    keep = np.empty(n_clusters, dtype=int)
    for c in range(n_clusters):
        m = np.where(labels == c)[0]
        keep[c] = m[np.argmax(areas[m])]
    return keep


def analyze_field(field: str):
    npz_path = OUT_DIR / f"{field}_sam_roi_test_ds1_v4.npz"
    d = np.load(npz_path)
    leaves = d["leaf_arr"]
    gsd = float(d["gsd_ds"][0])

    centroids = leaves[:, :2]
    areas = leaves[:, 2]

    # 두둑 필터
    ri = compute_ridges(centroids, gsd)
    keep_thresh_px = RIDGE_KEEP_CM / ri["px_to_cm"]
    if len(ri["ridge_pos_px"]) > 0:
        rp = ri["ridge_pos_px"]
        dists = np.abs(ri["proj_perp"][:, None] - rp[None, :]).min(axis=1)
        keep_A = dists <= keep_thresh_px
    else:
        keep_A = np.ones(len(leaves), dtype=bool)

    ridge_idx = np.where(keep_A)[0]
    ridge_centroids = centroids[ridge_idx]
    ridge_areas = areas[ridge_idx]

    # 두둑 위 잎들의 인접 간격 (필터링 전 raw)
    ri_sub = dict(ri)
    ri_sub["proj_perp"] = ri["proj_perp"][ridge_idx]
    ri_sub["proj_par"] = ri["proj_par"][ridge_idx]
    raw_gaps = compute_adjacent_gaps_per_ridge(ri_sub, keep_thresh_px)

    # 봉우리 + 임계 검출
    detect = detect_dedup_threshold(raw_gaps)
    thr_auto = detect["threshold_cm"]

    # dedup 3가지: 0cm (none), 5cm (기존), auto
    px_to_cm = ri["px_to_cm"]

    def apply_and_measure(radius_cm):
        if radius_cm is None or radius_cm <= 0:
            reps = np.arange(len(ridge_idx))
        else:
            reps = dedup_by_radius(ridge_centroids, ridge_areas,
                                    radius_cm / px_to_cm)
        keep_final = ridge_idx[reps]
        # dedup 후 재계산: 두둑 위 남은 것들의 인접 간격
        ri_final = dict(ri)
        ri_final["proj_perp"] = ri["proj_perp"][keep_final]
        ri_final["proj_par"] = ri["proj_par"][keep_final]
        gaps_final = compute_adjacent_gaps_per_ridge(ri_final, keep_thresh_px)
        # 극단 outlier 제거 (>60cm은 두둑 끝단 gap)
        gaps_final = gaps_final[gaps_final <= 60]
        return {
            "n_final": int(len(keep_final)),
            "median_gap": float(np.median(gaps_final)) if len(gaps_final) else np.nan,
            "mean_gap": float(np.mean(gaps_final)) if len(gaps_final) else np.nan,
            "gaps": gaps_final,
        }

    r_none = apply_and_measure(0)
    r_5cm = apply_and_measure(5.0)
    r_auto = apply_and_measure(thr_auto) if thr_auto is not None else r_5cm

    ridge_spacings_cm = np.diff(ri["ridge_pos_px"]) * px_to_cm \
                        if len(ri["ridge_pos_px"]) > 1 else np.array([])

    return {
        "field": field,
        "n_ridges": int(len(ri["ridge_pos_px"])),
        "ridge_spacing_cm": float(ridge_spacings_cm.mean())
                            if len(ridge_spacings_cm) else np.nan,
        "n_raw_on_ridge": int(len(ridge_idx)),
        "peak1_cm": detect["peak1_cm"],
        "peak2_cm": detect["peak2_cm"],
        "auto_threshold_cm": thr_auto,
        "detect_hist": detect["hist"],
        "detect_edges": detect["edges"],
        "detect_hist_smooth": detect["hist_smooth"],
        "detect_n_gaps": detect["n_gaps"],
        "none": r_none,
        "fixed5": r_5cm,
        "auto": r_auto,
    }


def draw_field_hist(res: dict, out_png: Path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    edges = res["detect_edges"]
    hist = res["detect_hist"]
    hs = res["detect_hist_smooth"]
    if len(edges) == 0:
        ax.text(0.5, 0.5, "간격 데이터 부족", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        plt.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close()
        return
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, hist, width=GAP_HIST_BIN_CM * 0.9,
           color="lightblue", edgecolor="none", label="원본 인접 간격")
    ax.plot(centers, hs, color="C0", lw=1.5, label="smoothed")
    if res["peak1_cm"] is not None:
        ax.axvline(res["peak1_cm"], color="red", lw=1.5,
                   label=f"봉우리 1 = {res['peak1_cm']:.1f}cm (같은 잎)")
    if res["peak2_cm"] is not None:
        ax.axvline(res["peak2_cm"], color="green", lw=1.5,
                   label=f"봉우리 2 = {res['peak2_cm']:.1f}cm (개체 간)")
    if res["auto_threshold_cm"] is not None:
        ax.axvline(res["auto_threshold_cm"], color="orange", lw=2, ls="--",
                   label=f"자동 임계 = {res['auto_threshold_cm']:.1f}cm")
    ax.axvline(5.0, color="gray", lw=1, ls=":", label="고정 임계 5.0cm")
    ax.set_xlim(0, GAP_HIST_MAX_CM)
    ax.set_xlabel("인접 잎 간 거리 (cm, 두둑 방향)")
    ax.set_ylabel("빈도")
    ax.set_title(f"{res['field']}  —  두둑 위 인접 간격 분포  "
                 f"(개체 수 {res['n_raw_on_ridge']:,}, 두둑 {res['n_ridges']}개)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 요약 텍스트
    summary = (f"고정 5cm dedup: {res['fixed5']['n_final']:,}개, "
               f"주간 median {res['fixed5']['median_gap']:.1f}cm\n"
               f"자동 dedup: {res['auto']['n_final']:,}개, "
               f"주간 median {res['auto']['median_gap']:.1f}cm")
    ax.text(0.98, 0.60, summary, transform=ax.transAxes,
            fontsize=10, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="gray", alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def draw_comparison(results: list, out_png: Path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    names = [r["field"] for r in results]
    x = np.arange(len(names))
    w = 0.28

    # (1) 개체 수 (dedup 방식별)
    n_none = [r["none"]["n_final"] for r in results]
    n_5cm = [r["fixed5"]["n_final"] for r in results]
    n_auto = [r["auto"]["n_final"] for r in results]
    ax1.bar(x - w, n_none, w, color="#aaaaaa", label="dedup 없음")
    ax1.bar(x, n_5cm, w, color="#4477bb", label="고정 5cm")
    ax1.bar(x + w, n_auto, w, color="#22aa22", label="자동 임계")
    ax1.set_ylabel("개체 수", fontsize=12)
    ax1.set_title("dedup 방식별 개체 수 비교", fontsize=13, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15)
    ax1.grid(True, axis="y", alpha=0.3); ax1.legend()

    # (2) 실측 주간 간격 median
    g_none = [r["none"]["median_gap"] for r in results]
    g_5cm = [r["fixed5"]["median_gap"] for r in results]
    g_auto = [r["auto"]["median_gap"] for r in results]
    ax2.bar(x - w, g_none, w, color="#aaaaaa", label="dedup 없음")
    ax2.bar(x, g_5cm, w, color="#4477bb", label="고정 5cm")
    ax2.bar(x + w, g_auto, w, color="#22aa22", label="자동 임계")
    ax2.axhline(20.0, color="red", ls="--", lw=1, alpha=0.7,
                label="파종 스펙 20cm")
    ax2.set_ylabel("실측 주간 간격 median (cm)", fontsize=12)
    ax2.set_title("dedup 방식별 실측 주간 간격", fontsize=13, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15)
    ax2.grid(True, axis="y", alpha=0.3); ax2.legend()

    # (3) 자동 임계값과 봉우리
    peak1 = [r["peak1_cm"] or 0 for r in results]
    peak2 = [r["peak2_cm"] or 0 for r in results]
    thr = [r["auto_threshold_cm"] or 0 for r in results]
    ax3.plot(x, peak1, "o-", color="red", label="같은 잎 봉우리 (peak 1)")
    ax3.plot(x, peak2, "s-", color="green", label="개체 간 봉우리 (peak 2)")
    ax3.plot(x, thr, "^-", color="orange", label="자동 임계 (valley)")
    ax3.axhline(5.0, color="gray", ls=":", lw=1, alpha=0.6, label="고정 5cm 참고")
    ax3.set_ylabel("cm", fontsize=12)
    ax3.set_title("필지별 봉우리·임계값", fontsize=13, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(names, rotation=15)
    ax3.grid(True, alpha=0.3); ax3.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def make_markdown(results: list) -> str:
    lines = []
    lines.append("# 데이터 기반 dedup 임계 진단\n")
    lines.append("**목적**: 두둑 위 잎들의 인접 간격 분포에서 bimodal 봉우리를 찾아 "
                 "자동 dedup 임계값 검출\n")
    lines.append("- 봉우리 1 (~2-5cm) = 같은 잎이 여러 마스크로 쪼개진 것")
    lines.append("- 봉우리 2 (~15-25cm) = 실제 이웃 개체 간 거리")
    lines.append("- **자동 임계 = 두 봉우리 사이 valley**\n")

    lines.append("## 필지별 진단 결과\n")
    lines.append("| 필지 | 두둑수 | 두둑 위 잎 | 봉우리1 | 봉우리2 | **자동임계** "
                 "| 5cm dedup 개체수 | 자동 dedup 개체수 | 실측 주간(자동) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        p1 = f"{r['peak1_cm']:.1f}" if r['peak1_cm'] else "-"
        p2 = f"{r['peak2_cm']:.1f}" if r['peak2_cm'] else "-"
        thr = f"**{r['auto_threshold_cm']:.1f}cm**" if r['auto_threshold_cm'] else "-"
        gap_auto = f"{r['auto']['median_gap']:.1f}" \
                   if not np.isnan(r['auto']['median_gap']) else "-"
        lines.append(
            f"| {r['field']} | {r['n_ridges']} | {r['n_raw_on_ridge']:,} | "
            f"{p1} | {p2} | {thr} | "
            f"{r['fixed5']['n_final']:,} | {r['auto']['n_final']:,} | "
            f"{gap_auto} |")

    lines.append("\n## 판단 가이드\n")
    lines.append("- **자동임계가 5cm 근처(3-7cm)** → 고정 5cm 방식이 적정")
    lines.append("- **자동임계가 5cm보다 훨씬 작음(< 3cm)** → 잎 조각이 조밀, "
                 "5cm이 살짝 과도한 병합 → 자동 임계로 완화 필요")
    lines.append("- **자동임계가 5cm보다 훨씬 큼(> 7cm)** → SAM sub-mask가 넓게 퍼짐, "
                 "5cm은 부족 → 자동 임계로 강화 필요")
    lines.append("- **봉우리 검출 실패** → 데이터 부족 또는 unimodal → 별도 검토")
    return "\n".join(lines)


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    print("=" * 60)
    print("데이터 기반 dedup 임계 진단")
    print("=" * 60)

    results = []
    for f in FIELDS:
        print(f"\n[{f}]")
        try:
            r = analyze_field(f)
        except FileNotFoundError:
            print(f"  ❌ npz 없음")
            continue
        results.append(r)
        thr = r["auto_threshold_cm"]
        thr_s = f"{thr:.1f}cm" if thr else "N/A"
        print(f"  두둑 {r['n_ridges']}개, 두둑 위 잎 {r['n_raw_on_ridge']:,}, "
              f"봉우리1 {r['peak1_cm'] or 0:.1f}cm, "
              f"봉우리2 {r['peak2_cm'] or 0:.1f}cm, 자동임계 {thr_s}")
        print(f"    dedup 없음: {r['none']['n_final']:,}개, "
              f"주간 median {r['none']['median_gap']:.1f}cm")
        print(f"    5cm 고정:  {r['fixed5']['n_final']:,}개, "
              f"주간 median {r['fixed5']['median_gap']:.1f}cm")
        print(f"    자동 임계: {r['auto']['n_final']:,}개, "
              f"주간 median {r['auto']['median_gap']:.1f}cm")
        draw_field_hist(r, DIAG_DIR / f"{f}_hist.png")

    if not results:
        return

    draw_comparison(results, DIAG_DIR / "comparison.png")
    md = make_markdown(results)
    (DIAG_DIR / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    # JSON (numpy 제거)
    def clean(r):
        return {k: v for k, v in r.items()
                if k not in ("detect_hist", "detect_edges", "detect_hist_smooth")
                and not (isinstance(v, dict) and "gaps" in v)}
    # 각 방식 gaps 배열 제거
    clean_list = []
    for r in results:
        c = {}
        for k, v in r.items():
            if k in ("detect_hist", "detect_edges", "detect_hist_smooth"):
                continue
            if k in ("none", "fixed5", "auto"):
                c[k] = {kk: vv for kk, vv in v.items() if kk != "gaps"}
            else:
                c[k] = v
        clean_list.append(c)
    (DIAG_DIR / "summary.json").write_text(
        json.dumps(clean_list, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 저장: {DIAG_DIR}")


if __name__ == "__main__":
    main()
