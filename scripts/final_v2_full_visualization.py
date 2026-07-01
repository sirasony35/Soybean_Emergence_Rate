"""
1-1 필지 파종기 최종 리포트 (v2 개선)

3장의 이미지 생성:
  ① full_field_leaves.png       — 전체 필지 RGB + 검출된 콩잎(초록) + 두둑 라인
  ② ridge_precision_analysis.png — 두둑별 3지표 측정 + 균일도
  ③ final_summary.png           — 최종 비교 리포트
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
from sklearn.cluster import KMeans


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
OUT_DIR = ROOT / "result" / "sam_test"
REPORT_DIR = OUT_DIR / "final_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = {
    "Smart":  dict(name="스마트 파종기", npz="GJSM-1-1_Smart_sam_FULL_v4.npz",
                    spec=dict(ridge_width=70, row_gap=30, plant_gap=20, furrow=35),
                    # Smart 스펙 내부 줄 30cm → min_dist 15 (스펙의 50%)
                    min_ridge_dist_cm=15, smooth_sigma=0.4,
                    color="#0066cc"),
    "Normal": dict(name="일반 파종기",   npz="GJSM-1-1_normal_sam_FULL_v4.npz",
                    spec=dict(ridge_width=140, row_gap=70, plant_gap=20, furrow=35),
                    # Normal 스펙 내부 줄 70cm → min_dist 55 (스펙의 80%)
                    min_ridge_dist_cm=55, smooth_sigma=1.2,
                    color="#cc6600"),
}

RIDGE_KEEP_CM = 20
DEDUP_RADIUS_CM = 5.0


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


def dedup_radius(centroids, areas, radius_px):
    n = len(centroids)
    if n == 0:
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


def analyze(key: str):
    field = FIELDS[key]
    d = np.load(OUT_DIR / field["npz"])
    leaves = d["leaf_arr"]
    gsd = float(d["gsd_ds"][0])
    valid_ha = float(d["valid_area_ha"][0])
    rgb = d["rgb_disp"]; step = int(d["rgb_step"][0])
    centroids = leaves[:, :2]; areas = leaves[:, 2]

    px_to_cm = gsd * 100
    mean = centroids.mean(axis=0); centered = centroids - mean
    ang = find_best_angle(centered, gsd)
    a = np.radians(ang)
    perp = np.array([np.cos(a), np.sin(a)])
    par = np.array([-np.sin(a), np.cos(a)])
    proj_perp = centered @ perp
    proj_par = centered @ par

    # 봉우리(줄) 검출 — 필지별 스펙에 맞춰 튜닝
    bin_size_px = 0.5 / px_to_cm
    edges = np.arange(proj_perp.min(), proj_perp.max() + bin_size_px, bin_size_px)
    hist, _ = np.histogram(proj_perp, bins=edges)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=field["smooth_sigma"])
    peaks, _ = find_peaks(hist_smooth,
                          distance=field["min_ridge_dist_cm"] / px_to_cm,
                          height=hist_smooth.max() * 0.02)
    peak_pos_px = edges[peaks] + bin_size_px / 2
    peak_pos_cm = peak_pos_px * px_to_cm

    # 두둑 필터 A + Dedup B (전체 잎 → 최종 개체)
    keep_thresh_px = RIDGE_KEEP_CM / px_to_cm
    dists = np.abs(proj_perp[:, None] - peak_pos_px[None, :]).min(axis=1)
    keep_A = dists <= keep_thresh_px
    ridge_idx = np.where(keep_A)[0]
    reps = dedup_radius(centroids[ridge_idx], areas[ridge_idx],
                        DEDUP_RADIUS_CM / px_to_cm)
    final_idx = ridge_idx[reps]
    final_centroids = centroids[final_idx]
    final_proj_perp = final_centroids @ perp - mean @ perp
    final_proj_par = final_centroids @ par - mean @ par

    # 봉우리 간격 히스토그램 → KMeans 분리
    gaps = np.diff(peak_pos_cm) if len(peak_pos_cm) > 1 else np.array([])
    if len(gaps) >= 2:
        spec = field["spec"]
        expect_period = spec["ridge_width"] + spec["furrow"]
        init = np.array([[spec["row_gap"]], [expect_period - spec["row_gap"]]])
        km = KMeans(n_clusters=2, n_init=1, init=init, random_state=0).fit(gaps.reshape(-1, 1))
        labels = km.labels_; centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        small_gaps = gaps[labels == order[0]]
        large_gaps = gaps[labels == order[1]]
        row_gap_measured = float(np.median(small_gaps))
        big_gap_measured = float(np.median(large_gaps))
        period_measured = row_gap_measured + big_gap_measured
        ridge_width_measured = period_measured - field["spec"]["furrow"]
        sep_ratio = (big_gap_measured - row_gap_measured) / max(row_gap_measured, 1)
    else:
        row_gap_measured = np.nan; big_gap_measured = np.nan
        period_measured = np.nan; ridge_width_measured = np.nan
        sep_ratio = 0
        small_gaps = np.array([]); large_gaps = np.array([])

    # 두둑별 주간 계산 + 균일도
    per_ridge = []
    for pk_px in peak_pos_px:
        on = np.abs(proj_perp - pk_px) <= keep_thresh_px
        if on.sum() < 2:
            continue
        # dedup 후 잎
        idx_on = np.where(on)[0]
        reps_local = dedup_radius(centroids[idx_on], areas[idx_on],
                                    DEDUP_RADIUS_CM / px_to_cm)
        idx_final = idx_on[reps_local]
        along = np.sort(proj_par[idx_final]) * px_to_cm
        g = np.diff(along)
        g = g[(g > 3) & (g <= 60)]
        if len(g) < 2:
            continue
        per_ridge.append(dict(
            peak_cm=pk_px * px_to_cm,
            n_leaves=int(len(idx_final)),
            median_gap=float(np.median(g)),
            mean_gap=float(np.mean(g)),
            std_gap=float(np.std(g)),
            cv=float(np.std(g) / np.mean(g)) if np.mean(g) > 0 else np.nan,
        ))
    all_plant_gaps = np.array([r["median_gap"] for r in per_ridge])
    plant_gap_median = float(np.median(all_plant_gaps)) if len(all_plant_gaps) else np.nan
    plant_gap_cv = float(np.std(all_plant_gaps) / np.mean(all_plant_gaps)) \
                    if len(all_plant_gaps) and np.mean(all_plant_gaps) > 0 else np.nan
    all_cvs = np.array([r["cv"] for r in per_ridge if not np.isnan(r["cv"])])
    plant_gap_ridge_cv_mean = float(all_cvs.mean()) if len(all_cvs) else np.nan

    return dict(
        key=key, field=field,
        rgb_disp=rgb, step=step,
        n_leaves_raw=int(len(leaves)), n_leaves_final=int(len(final_idx)),
        final_centroids=final_centroids,
        valid_ha=valid_ha,
        density_per_ha=len(final_idx) / valid_ha,
        ridge_angle_deg=float(ang),
        n_peaks=int(len(peak_pos_cm)),
        peak_pos_cm=peak_pos_cm,
        gaps_cm=gaps,
        row_gap_measured=row_gap_measured,
        big_gap_measured=big_gap_measured,
        period_measured=period_measured,
        ridge_width_measured=ridge_width_measured,
        sep_ratio=sep_ratio,
        plant_gap_median=plant_gap_median,
        plant_gap_cv=plant_gap_cv,
        plant_gap_ridge_cv_mean=plant_gap_ridge_cv_mean,
        per_ridge=per_ridge,
    )


# ═══════════════════════════════════════════════════
# 이미지 1: 전체 필지 + 검출 콩잎
# ═══════════════════════════════════════════════════
def draw_full_field(smart, normal, out_png):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    for ax, res in zip(axes, [smart, normal]):
        rgb = res["rgb_disp"]; step = res["step"]; f = res["field"]
        ax.imshow(rgb)
        fc = res["final_centroids"]
        ax.scatter(fc[:, 1] / step, fc[:, 0] / step, s=3,
                    c="lime", alpha=0.7, edgecolor="none")
        ax.axis("off")
        title = (f"{f['name']}   ·   유효 {res['valid_ha']:.2f}ha   ·   "
                 f"발아 콩잎 {res['n_leaves_final']:,}개   ·   "
                 f"두둑 방향 {res['ridge_angle_deg']:.0f}°")
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=f["color"], pad=12)
    fig.suptitle("GJSM-1-1 필지 — SAM 검출 콩잎 (초록점)",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════
# 이미지 2: 두둑 정밀도 + 균일도
# ═══════════════════════════════════════════════════
def draw_precision(smart, normal, out_png):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # (a) 두둑 간격 히스토그램
    ax = axes[0, 0]
    for res, c, lbl in [(smart, "#0066cc", "Smart"), (normal, "#cc6600", "Normal")]:
        g = res["gaps_cm"]
        if len(g):
            ax.hist(g, bins=np.arange(0, min(g.max() + 5, 150), 2),
                     alpha=0.5, color=c, label=f"{lbl} (n={len(g)})")
    ax.set_xlabel("인접 봉우리(줄) 간격 (cm)")
    ax.set_ylabel("빈도")
    ax.set_title("두둑 구조 검출 — 봉우리 간격 분포",
                  fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    # 스펙 마커
    ax.axvline(30, color="#0066cc", ls="--", alpha=0.5, label="_")
    ax.axvline(70, color="#cc6600", ls="--", alpha=0.5, label="_")
    ax.text(30, ax.get_ylim()[1] * 0.9, "Smart\n스펙 30cm",
             color="#0066cc", ha="center", fontsize=9)
    ax.text(70, ax.get_ylim()[1] * 0.9, "Normal\n스펙 70cm",
             color="#cc6600", ha="center", fontsize=9)

    # (b) 두둑별 주간 median 분포
    ax = axes[0, 1]
    for res, c, lbl in [(smart, "#0066cc", "Smart"), (normal, "#cc6600", "Normal")]:
        gaps = np.array([r["median_gap"] for r in res["per_ridge"]])
        if len(gaps):
            ax.hist(gaps, bins=np.arange(10, min(gaps.max() + 5, 60), 1),
                     alpha=0.5, color=c,
                     label=f"{lbl} (median {np.median(gaps):.1f}cm)")
    ax.axvline(20, color="red", ls="--", label="파종 스펙 20cm")
    ax.set_xlabel("두둑별 파종 간격 median (cm)")
    ax.set_ylabel("두둑 개수")
    ax.set_title("주간(파종 간격) 분포 — 두둑별 median",
                  fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # (c) 두둑별 주간 CV (균일도)
    ax = axes[1, 0]
    for res, c, lbl in [(smart, "#0066cc", "Smart"), (normal, "#cc6600", "Normal")]:
        cvs = np.array([r["cv"] for r in res["per_ridge"] if not np.isnan(r["cv"])])
        if len(cvs):
            ax.hist(cvs, bins=np.arange(0, min(cvs.max() + 0.05, 1.2), 0.025),
                     alpha=0.5, color=c,
                     label=f"{lbl} (mean {cvs.mean():.3f})")
    ax.set_xlabel("두둑 내 주간 CV (낮을수록 균일)")
    ax.set_ylabel("두둑 개수")
    ax.set_title("파종 균일도 (CV = std/mean, 0에 가까울수록 균일)",
                  fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # (d) 두둑당 잎 수 분포
    ax = axes[1, 1]
    for res, c, lbl in [(smart, "#0066cc", "Smart"), (normal, "#cc6600", "Normal")]:
        n_leaves = np.array([r["n_leaves"] for r in res["per_ridge"]])
        if len(n_leaves):
            ax.hist(n_leaves,
                     bins=np.arange(0, n_leaves.max() + 20, 10),
                     alpha=0.5, color=c,
                     label=f"{lbl} (mean {n_leaves.mean():.0f})")
    ax.set_xlabel("두둑당 검출 잎 수")
    ax.set_ylabel("두둑 개수")
    ax.set_title("두둑별 검출 잎 수 (파종 조밀도)",
                  fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("두둑별 파종 정밀도 · 균일도 분석",
                 fontsize=17, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════
# 이미지 3: 최종 비교 리포트
# ═══════════════════════════════════════════════════
def draw_final_summary(smart, normal, out_png):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, height_ratios=[0.4, 2.2, 1.4, 1.2],
                          hspace=0.5, wspace=0.15)

    # 타이틀
    ax_t = fig.add_subplot(gs[0, :]); ax_t.axis("off")
    ax_t.text(0.5, 0.7, "GJSM-1-1 필지 파종기 최종 비교 리포트",
              ha="center", fontsize=24, fontweight="bold",
              transform=ax_t.transAxes)
    ax_t.text(0.5, 0.2, "이미지 실측 → 두둑 너비 · 파종 줄간격 · 파종 간격 · 균일도",
              ha="center", fontsize=13, color="gray",
              transform=ax_t.transAxes)

    # 두 필지 카드
    for i, res in enumerate([smart, normal]):
        col_slice = slice(0, 2) if i == 0 else slice(2, 4)
        ax = fig.add_subplot(gs[1, col_slice]); ax.axis("off")
        f = res["field"]; s = f["spec"]

        ax.text(0.02, 0.97, f["name"],
                 fontsize=22, fontweight="bold", color=f["color"],
                 transform=ax.transAxes, verticalalignment="top")
        ax.text(0.02, 0.92,
                 f"검출 잎 {res['n_leaves_final']:,}개  ·  "
                 f"유효 {res['valid_ha']:.2f}ha  ·  "
                 f"재식밀도 {res['density_per_ha']:,.0f}/ha  ·  "
                 f"두둑 방향 {res['ridge_angle_deg']:.0f}°",
                 fontsize=10, color="gray",
                 transform=ax.transAxes, verticalalignment="top")

        rows = [
            ("두둑 너비",       s["ridge_width"], res["ridge_width_measured"]),
            ("파종 줄 간격",    s["row_gap"],     res["row_gap_measured"]),
            ("파종 간격 (주간)", s["plant_gap"],   res["plant_gap_median"]),
        ]
        y = 0.80
        for label, sp, ms in rows:
            if np.isnan(ms):
                icon = "[측정불가]"; ms_s = "N/A"; err_s = ""; col = "#888"
            else:
                err = abs(ms - sp) / sp
                ok = err <= 0.15
                icon = "[일치]" if ok else "[불일치]"
                ms_s = f"{ms:.1f}"; err_s = f"오차 ±{abs(ms - sp):.1f}cm"
                col = f["color"] if ok else "#cc3333"
            ax.text(0.03, y, label,
                     fontsize=13, fontweight="bold",
                     transform=ax.transAxes, verticalalignment="top")
            ax.text(0.03, y - 0.05,
                     f"  스펙 : {sp}cm     실측 : {ms_s}cm     {icon}     {err_s}",
                     fontsize=11, color=col,
                     transform=ax.transAxes, verticalalignment="top")
            y -= 0.10

        # 균일도
        y -= 0.02
        ax.text(0.03, y, "파종 균일도 (CV, 낮을수록 균일)",
                 fontsize=12, fontweight="bold",
                 transform=ax.transAxes, verticalalignment="top")
        ax.text(0.03, y - 0.05,
                 f"  두둑별 주간 CV 평균 : {res['plant_gap_ridge_cv_mean']:.3f}",
                 fontsize=11, transform=ax.transAxes, verticalalignment="top")

        # bimodal 판정
        y -= 0.14
        is_bi = res["sep_ratio"] > 1.0
        pattern = "dual-row 확인" if is_bi else "dual-row 없음 (균일 파종)"
        pcol = "#22aa22" if is_bi else "#cc6600"
        ax.text(0.03, y, "파종 패턴 검출",
                 fontsize=11, color="#555",
                 transform=ax.transAxes, verticalalignment="top")
        ax.text(0.03, y - 0.05, pattern,
                 fontsize=13, fontweight="bold", color=pcol,
                 transform=ax.transAxes, verticalalignment="top")

    # 비교표
    ax_tbl = fig.add_subplot(gs[2, :]); ax_tbl.axis("off")
    ax_tbl.text(0.5, 0.95, "요약 비교표",
                 ha="center", fontsize=15, fontweight="bold",
                 transform=ax_tbl.transAxes)

    cols = ["지표", "스펙 (Smart)", "실측 (Smart)",
            "스펙 (Normal)", "실측 (Normal)", "판정"]
    def fmt(v): return f"{v:.1f}cm" if not np.isnan(v) else "N/A"

    rows_data = [
        ("두둑 너비",       70, smart["ridge_width_measured"],
                              140, normal["ridge_width_measured"]),
        ("파종 줄 간격",    30, smart["row_gap_measured"],
                              70,  normal["row_gap_measured"]),
        ("파종 간격 (주간)", 20, smart["plant_gap_median"],
                              20, normal["plant_gap_median"]),
    ]
    tbl_rows = []
    for label, sp_s, m_s, sp_n, m_n in rows_data:
        # 판정 코멘트
        s_ok = (not np.isnan(m_s)) and (abs(m_s - sp_s) / sp_s <= 0.15)
        n_ok = (not np.isnan(m_n)) and (abs(m_n - sp_n) / sp_n <= 0.15)
        if s_ok and not n_ok: v = "Smart 우수"
        elif n_ok and not s_ok: v = "Normal 우수"
        elif s_ok and n_ok: v = "둘 다 양호"
        else: v = "둘 다 불일치"
        tbl_rows.append([label, f"{sp_s}cm", fmt(m_s),
                          f"{sp_n}cm", fmt(m_n), v])

    tbl = ax_tbl.table(cellText=tbl_rows, colLabels=cols, cellLoc="center",
                        loc="center", bbox=[0.05, 0.05, 0.9, 0.85])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#eeeeee")
        tbl[(0, j)].set_text_props(fontweight="bold")

    # 배경색
    for i, (label, sp_s, m_s, sp_n, m_n) in enumerate(rows_data, 1):
        s_ok = (not np.isnan(m_s)) and (abs(m_s - sp_s) / sp_s <= 0.15)
        n_ok = (not np.isnan(m_n)) and (abs(m_n - sp_n) / sp_n <= 0.15)
        tbl[(i, 2)].set_facecolor("#d9f2d9" if s_ok else "#f9d9d9")
        tbl[(i, 4)].set_facecolor("#d9f2d9" if n_ok else "#f9d9d9")

    # 최종 결론
    ax_conc = fig.add_subplot(gs[3, :]); ax_conc.axis("off")
    ax_conc.text(0.5, 0.85, "최종 판정",
                  ha="center", fontsize=15, fontweight="bold",
                  transform=ax_conc.transAxes)

    smart_oks = sum([
        (not np.isnan(smart["ridge_width_measured"])) and abs(smart["ridge_width_measured"] - 70) / 70 <= 0.15,
        (not np.isnan(smart["row_gap_measured"])) and abs(smart["row_gap_measured"] - 30) / 30 <= 0.15,
        (not np.isnan(smart["plant_gap_median"])) and abs(smart["plant_gap_median"] - 20) / 20 <= 0.15,
    ])
    normal_oks = sum([
        (not np.isnan(normal["ridge_width_measured"])) and abs(normal["ridge_width_measured"] - 140) / 140 <= 0.15,
        (not np.isnan(normal["row_gap_measured"])) and abs(normal["row_gap_measured"] - 70) / 70 <= 0.15,
        (not np.isnan(normal["plant_gap_median"])) and abs(normal["plant_gap_median"] - 20) / 20 <= 0.15,
    ])

    lines = [
        (f"스마트 파종기: {smart_oks}/3 스펙 일치  ·  "
         f"CV {smart['plant_gap_ridge_cv_mean']:.3f}",
         "#0066cc"),
        (f"일반 파종기: {normal_oks}/3 스펙 일치  ·  "
         f"CV {normal['plant_gap_ridge_cv_mean']:.3f}",
         "#cc6600"),
    ]
    y = 0.65
    for txt, c in lines:
        ax_conc.text(0.5, y, txt, ha="center", fontsize=14, fontweight="bold",
                      color=c, transform=ax_conc.transAxes)
        y -= 0.18

    winner_txt = ("스마트 파종기가 스펙 준수·정밀도 관점에서 명백히 우수"
                  if smart_oks > normal_oks else
                  "일반 파종기가 스펙 준수 관점에서 우수"
                  if normal_oks > smart_oks else
                  "두 파종기 정밀도 대등")
    ax_conc.text(0.5, 0.20, winner_txt, ha="center",
                  fontsize=16, fontweight="bold", color="#004488",
                  transform=ax_conc.transAxes,
                  bbox=dict(boxstyle="round,pad=0.5",
                            facecolor="#f0f8ff", edgecolor="#004488"))

    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60); print("1-1 최종 리포트 v2"); print("=" * 60)
    smart = analyze("Smart")
    normal = analyze("Normal")

    for k, r in [("Smart", smart), ("Normal", normal)]:
        print(f"\n[{k}]")
        print(f"  발아 잎: {r['n_leaves_final']:,}, 두둑: {r['n_peaks']}, "
              f"두둑간격 측정 {r['row_gap_measured']:.1f}cm, "
              f"주간 median {r['plant_gap_median']:.1f}cm")
        print(f"  두둑 너비 실측: {r['ridge_width_measured']:.1f}cm")
        print(f"  CV: {r['plant_gap_ridge_cv_mean']:.3f}")

    print("\n[이미지 1: 전체 필지 + 검출 콩잎]")
    draw_full_field(smart, normal, REPORT_DIR / "1_full_field_leaves.png")
    print(f"  저장: {REPORT_DIR / '1_full_field_leaves.png'}")

    print("\n[이미지 2: 두둑 정밀도 + 균일도]")
    draw_precision(smart, normal, REPORT_DIR / "2_ridge_precision_analysis.png")
    print(f"  저장: {REPORT_DIR / '2_ridge_precision_analysis.png'}")

    print("\n[이미지 3: 최종 비교 리포트]")
    draw_final_summary(smart, normal, REPORT_DIR / "3_final_summary.png")
    print(f"  저장: {REPORT_DIR / '3_final_summary.png'}")

    # JSON 저장
    def clean(r):
        return {k: v for k, v in r.items()
                if k not in ("rgb_disp", "final_centroids",
                              "peak_pos_cm", "gaps_cm", "per_ridge")}
    (REPORT_DIR / "final_stats.json").write_text(
        json.dumps({"smart": clean(smart), "normal": clean(normal)},
                    ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"\n✅ 완료: {REPORT_DIR}")


if __name__ == "__main__":
    main()
