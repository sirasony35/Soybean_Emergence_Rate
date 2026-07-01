"""
6개 필지 종합 요약 리포트 생성.

입력: result/sam_test/{field}_sam_roi_test_ds1_v4.npz (6개)

출력:
  result/sam_test/summary/{field}_summary.png    (필지별 요약 카드)
  result/sam_test/summary/comparison.png         (6필지 비교 차트)
  result/sam_test/summary/summary_table.md       (마크다운 요약표)
  result/sam_test/summary/summary_stats.json     (구조화 통계)

계산:
  - 발아 콩잎 수 (필터 통과)
  - 재식밀도 (개체/ha)
  - 주간 간격 (두둑 방향 정렬 후 인접 잎 거리 median)
  - 두둑 간격 (수직축 봉우리 간 거리 avg)
  - 입모율 (76,923/ha 기준)
"""
from __future__ import annotations
import json
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
SUMMARY_DIR = OUT_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["GJSM-1-1_Smart", "GJSM-1-1_normal",
          "GJSM-1-2", "GJSM-1-3", "GJSM-2-2", "GJSM-2-3"]

ROI_M2 = 30.0 * 30.0
STD_PER_HA = 76923

# ─── 필터 파라미터 (sam_ridge_filter_v4.py와 동일) ───
RIDGE_SPACING_MIN_CM = 20
RIDGE_KEEP_CM = 20
HIST_BIN_CM = 1.0
HIST_SMOOTH_SIGMA = 0.5
PEAK_HEIGHT_FRAC = 0.03
DEDUP_RADIUS_CM = 5.0
# ⚠️ 생육 초기 (잡초 미발생) → C+D 비활성
CD_ENABLED = False
SIZE_MIN_CM2 = 2.0
SIZE_MAX_CM2 = 100.0
CENTER_TIGHT_CM = 12.0
SIZE_STRICT_MIN_CM2 = 3.0
SIZE_STRICT_MAX_CM2 = 60.0


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
    proj = centered @ perp_dir
    bin_size_px = HIST_BIN_CM / px_to_cm
    edges = np.arange(proj.min(), proj.max() + bin_size_px, bin_size_px)
    hist, _ = np.histogram(proj, bins=edges)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=HIST_SMOOTH_SIGMA)
    min_spacing_px = RIDGE_SPACING_MIN_CM / px_to_cm
    height_thresh = hist_smooth.max() * PEAK_HEIGHT_FRAC
    peaks, _ = find_peaks(hist_smooth, distance=min_spacing_px, height=height_thresh)
    ridge_proj_px = edges[peaks] + bin_size_px / 2
    return {
        "mean": mean, "ridge_dir": ridge_dir, "perp_dir": perp_dir,
        "ridge_angle_deg": ridge_angle_deg, "proj": proj,
        "ridge_proj_px": ridge_proj_px, "px_to_cm": px_to_cm,
        "centered": centered,
    }


def dedup(centroids, areas, radius_px):
    n = len(centroids)
    if n == 0:
        return np.array([], dtype=int)
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


def compute_intra_row_spacing_cm(final_centroids, ridge_info, ridge_pos, keep_thresh_px):
    """각 두둑 위 잎들을 두둑 방향으로 정렬 → 인접 간격 median (cm)."""
    if len(ridge_pos) == 0 or len(final_centroids) == 0:
        return np.nan, np.nan
    centered = final_centroids - ridge_info["mean"]
    proj_perp = centered @ ridge_info["perp_dir"]      # 두둑 수직축
    proj_par = centered @ ridge_info["ridge_dir"]      # 두둑 방향
    all_spacings = []
    for rp in ridge_pos:
        mask = np.abs(proj_perp - rp) <= keep_thresh_px
        if mask.sum() < 2:
            continue
        along = np.sort(proj_par[mask])
        gaps = np.diff(along) * ridge_info["px_to_cm"]
        # 극단 outlier 제거 (>60cm는 두둑 끝단 gap)
        gaps = gaps[gaps <= 60]
        if len(gaps) > 0:
            all_spacings.extend(gaps.tolist())
    if not all_spacings:
        return np.nan, np.nan
    return float(np.median(all_spacings)), float(np.mean(all_spacings))


def process_field(field_name: str) -> dict:
    npz_path = OUT_DIR / f"{field_name}_sam_roi_test_ds1_v4.npz"
    d = np.load(npz_path)
    rgb = d["rgb_disp"]
    step = int(d["rgb_step"][0])
    leaves = d["leaf_arr"]        # (N, 10) [y, x, area, h, s, v, y0, x0, y1, x1]
    gsd = float(d["gsd_ds"][0])
    n_all_masks = int(d["n_all_masks"][0])

    centroids = leaves[:, :2]
    areas_cm2 = leaves[:, 2]

    # ─── A. 두둑 필터 ───
    ridge_info = compute_ridges(centroids, gsd)
    ridge_pos = ridge_info["ridge_proj_px"]
    proj = ridge_info["proj"]
    keep_thresh_px = RIDGE_KEEP_CM / ridge_info["px_to_cm"]
    if len(ridge_pos) > 0:
        dists = np.abs(proj[:, None] - ridge_pos[None, :]).min(axis=1)
        keep_A = dists <= keep_thresh_px
    else:
        dists = np.full(len(leaves), np.inf)
        keep_A = np.ones(len(leaves), dtype=bool)

    # ─── B. Dedup ───
    ridge_idx = np.where(keep_A)[0]
    ridge_centroids = centroids[ridge_idx]
    ridge_areas = areas_cm2[ridge_idx]
    dedup_radius_px = DEDUP_RADIUS_CM / ridge_info["px_to_cm"]
    rep_local = dedup(ridge_centroids, ridge_areas, dedup_radius_px)
    dedup_idx = ridge_idx[rep_local]
    dedup_mask = np.zeros(len(leaves), dtype=bool)
    dedup_mask[dedup_idx] = True

    # ─── C+D. 크기·거리 (생육 초기 → 잡초 미발생 → 비활성) ───
    if CD_ENABLED:
        dist_cm = dists * ridge_info["px_to_cm"]
        area_ok = (areas_cm2 >= SIZE_MIN_CM2) & (areas_cm2 <= SIZE_MAX_CM2)
        center_tight = dist_cm <= CENTER_TIGHT_CM
        area_strict = (areas_cm2 >= SIZE_STRICT_MIN_CM2) & (areas_cm2 <= SIZE_STRICT_MAX_CM2)
        cd_pass = np.where(center_tight, area_ok, area_ok & area_strict)
        final_mask = dedup_mask & cd_pass
    else:
        final_mask = dedup_mask

    # ─── 통계 ───
    n_final = int(final_mask.sum())
    final_centroids = centroids[final_mask]
    density_per_ha = n_final / (ROI_M2 / 10000)
    rate = density_per_ha / STD_PER_HA * 100

    ridge_spacings_cm = np.diff(ridge_pos) * ridge_info["px_to_cm"] if len(ridge_pos) > 1 else np.array([])
    ridge_spacing_avg = float(ridge_spacings_cm.mean()) if len(ridge_spacings_cm) else np.nan

    intra_med, intra_avg = compute_intra_row_spacing_cm(
        final_centroids, ridge_info, ridge_pos, keep_thresh_px)

    return {
        "field": field_name,
        "n_all_masks": n_all_masks,
        "n_leaves_raw": int(len(leaves)),
        "n_final": n_final,
        "density_per_ha": density_per_ha,
        "emergence_rate_pct": rate,
        "ridge_angle_deg": float(ridge_info["ridge_angle_deg"]),
        "n_ridges": int(len(ridge_pos)),
        "ridge_spacing_cm": ridge_spacing_avg,
        "intra_row_spacing_median_cm": intra_med,
        "intra_row_spacing_mean_cm": intra_avg,
        "roi_size_m": 30.0,
        "_rgb": rgb, "_step": step,
        "_final_centroids": final_centroids,
    }


def draw_field_card(res: dict, out_png: Path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax_txt, ax_img) = plt.subplots(1, 2, figsize=(16, 8),
                                          gridspec_kw={"width_ratios": [1, 1.4]})
    ax_txt.axis("off")

    field = res["field"]
    ax_txt.text(0.02, 0.97, field, fontsize=22, fontweight="bold",
                transform=ax_txt.transAxes, verticalalignment="top")
    ax_txt.text(0.02, 0.90, f"({res['roi_size_m']:.0f}m × {res['roi_size_m']:.0f}m ROI)",
                fontsize=11, color="gray", transform=ax_txt.transAxes,
                verticalalignment="top")

    lines = [
        ("발아 콩잎",     f"{res['n_final']:,} 개"),
        ("재식밀도",      f"{res['density_per_ha']:,.0f} 개체/ha"),
        ("주간 간격",     f"{res['intra_row_spacing_median_cm']:.1f} cm  (median)"
                          f"  |  {res['intra_row_spacing_mean_cm']:.1f} cm (mean)"
                          if not np.isnan(res['intra_row_spacing_median_cm'])
                          else "N/A"),
        ("두둑 간격",     f"{res['ridge_spacing_cm']:.1f} cm  "
                          f"({res['n_ridges']}개 두둑)"
                          if not np.isnan(res['ridge_spacing_cm'])
                          else "N/A"),
        ("두둑 방향",     f"{res['ridge_angle_deg']:.0f}°  (0°=수평, 90°=수직)"),
    ]
    y = 0.78
    for k, v in lines:
        ax_txt.text(0.02, y, f"{k}", fontsize=13, color="#444",
                    transform=ax_txt.transAxes, verticalalignment="top")
        ax_txt.text(0.30, y, f"{v}", fontsize=14, fontweight="bold",
                    transform=ax_txt.transAxes, verticalalignment="top")
        y -= 0.09

    # 큰 입모율 표시
    ax_txt.text(0.02, 0.20, "입모율", fontsize=13, color="#444",
                transform=ax_txt.transAxes, verticalalignment="top")
    rate = res["emergence_rate_pct"]
    color = "#22aa22" if rate >= 70 else "#c66600" if rate >= 50 else "#cc3333"
    ax_txt.text(0.30, 0.24, f"{rate:.1f} %",
                fontsize=44, fontweight="bold", color=color,
                transform=ax_txt.transAxes, verticalalignment="top")
    ax_txt.text(0.30, 0.05, "(표준 76,923/ha 기준)",
                fontsize=9, color="gray", transform=ax_txt.transAxes,
                verticalalignment="top")

    # RGB + 초록점
    rgb = res["_rgb"]
    step = res["_step"]
    ax_img.imshow(rgb)
    fc = res["_final_centroids"]
    ax_img.scatter(fc[:, 1] / step, fc[:, 0] / step, s=6,
                   c="lime", alpha=0.85, edgecolor="none")
    ax_img.axis("off")
    ax_img.set_title(f"발아 콩잎 검출 위치 ({res['n_final']:,}개)",
                     fontsize=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def draw_comparison(results: list, out_png: Path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    names = [r["field"] for r in results]
    rates = [r["emergence_rate_pct"] for r in results]
    densities = [r["density_per_ha"] for r in results]
    n_leaves = [r["n_final"] for r in results]

    colors = ["#22aa22" if r >= 70 else "#c66600" if r >= 50 else "#cc3333"
              for r in rates]

    # 입모율 막대
    bars1 = ax1.bar(names, rates, color=colors)
    ax1.set_ylabel("입모율 (%)", fontsize=12)
    ax1.set_title("6필지 입모율 비교", fontsize=14, fontweight="bold")
    ax1.axhline(70, color="#22aa22", ls="--", lw=1, alpha=0.5, label="70% (양호)")
    ax1.axhline(50, color="#c66600", ls="--", lw=1, alpha=0.5, label="50% (주의)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right")
    for b, r in zip(bars1, rates):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                 f"{r:.1f}%", ha="center", fontsize=11, fontweight="bold")

    # 재식밀도 + 잎 개수
    ax2.bar(names, densities, color="#4477bb")
    ax2.set_ylabel("재식밀도 (개체/ha)", fontsize=12)
    ax2.axhline(STD_PER_HA, color="black", ls="--", lw=1, alpha=0.5,
                label=f"표준 {STD_PER_HA:,}/ha (100%)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right")
    for i, (n, d) in enumerate(zip(n_leaves, densities)):
        ax2.text(i, d + 1500, f"{d:,.0f}\n({n:,}잎)", ha="center", fontsize=10)
    ax2.set_title("6필지 재식밀도 비교", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def make_markdown_table(results: list) -> str:
    lines = []
    lines.append("# 새만금 논콩 입모율 분석 — 6필지 요약\n")
    lines.append("**ROI**: 각 필지 중심 30m × 30m  |  **GSD**: 5.25mm/px  "
                 "|  **모델**: SAM v2 (강화)  |  **표준**: 76,923개체/ha (65×20cm 파종)\n")
    lines.append("## 필지별 요약\n")
    lines.append("| 필지 | 발아 콩잎 | 재식밀도(/ha) | 주간(cm) | 두둑간격(cm) | "
                 "두둑방향 | **입모율** |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        rate = r["emergence_rate_pct"]
        rate_str = f"**{rate:.1f}%**"
        intra_str = f"{r['intra_row_spacing_median_cm']:.1f}" \
                    if not np.isnan(r['intra_row_spacing_median_cm']) else "-"
        ridge_str = f"{r['ridge_spacing_cm']:.1f}" \
                    if not np.isnan(r['ridge_spacing_cm']) else "-"
        lines.append(
            f"| {r['field']} | {r['n_final']:,} | "
            f"{r['density_per_ha']:,.0f} | {intra_str} | "
            f"{ridge_str} | {r['ridge_angle_deg']:.0f}° | {rate_str} |")

    # 파종기 비교 (Smart vs normal)
    smart = next((r for r in results if "_Smart" in r["field"]), None)
    normal = next((r for r in results if "_normal" in r["field"]), None)
    if smart and normal:
        lines.append("\n## 파종기 비교 (GJSM-1-1)\n")
        lines.append("| 항목 | Smart | Normal | 차이 |")
        lines.append("|---|---:|---:|---:|")
        diff_rate = normal["emergence_rate_pct"] - smart["emergence_rate_pct"]
        diff_den = normal["density_per_ha"] - smart["density_per_ha"]
        diff_leaf = normal["n_final"] - smart["n_final"]
        lines.append(f"| 발아 콩잎 | {smart['n_final']:,} | "
                     f"{normal['n_final']:,} | {diff_leaf:+,} |")
        lines.append(f"| 재식밀도(/ha) | {smart['density_per_ha']:,.0f} | "
                     f"{normal['density_per_ha']:,.0f} | {diff_den:+,.0f} |")
        lines.append(f"| **입모율** | **{smart['emergence_rate_pct']:.1f}%** | "
                     f"**{normal['emergence_rate_pct']:.1f}%** | "
                     f"**{diff_rate:+.1f}%p** |")

    lines.append("\n## 상세 결과 파일")
    lines.append("- 필지별 요약 카드: `result/sam_test/summary/{필지}_summary.png`")
    lines.append("- 비교 차트: `result/sam_test/summary/comparison.png`")
    lines.append("- 6패널 상세 분석: `result/sam_test/{필지}_ridge_filter_v4.png`")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("6필지 종합 요약 리포트 생성")
    print("=" * 60)

    results = []
    for f in FIELDS:
        print(f"\n[{f}] 처리 중...")
        try:
            r = process_field(f)
        except FileNotFoundError as e:
            print(f"  ❌ npz 없음: {e}")
            continue
        results.append(r)
        print(f"  발아 콩잎 {r['n_final']:,}, 재식밀도 {r['density_per_ha']:,.0f}/ha, "
              f"주간 {r['intra_row_spacing_median_cm']:.1f}cm, "
              f"입모율 {r['emergence_rate_pct']:.1f}%")

        # 필지별 카드
        card_path = SUMMARY_DIR / f"{f}_summary.png"
        draw_field_card(r, card_path)

    if not results:
        print("❌ 결과 없음")
        return

    # 비교 차트
    print("\n[요약] 비교 차트 생성...")
    draw_comparison(results, SUMMARY_DIR / "comparison.png")

    # 마크다운
    md = make_markdown_table(results)
    (SUMMARY_DIR / "summary_table.md").write_text(md, encoding="utf-8")
    print(md)

    # JSON (내부 데이터 제거)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in results]
    (SUMMARY_DIR / "summary_stats.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"\n✅ 저장 위치: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
