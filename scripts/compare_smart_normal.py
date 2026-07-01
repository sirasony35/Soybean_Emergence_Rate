"""
Smart vs Normal 파종기 종합 비교 리포트

전체 필지 데이터 (FULL_v4 npz) 로부터 두둑별 상세 지표 계산:
  - 두둑별 잎 수
  - 두둑별 재식밀도 (잎/미터)
  - 두둑별 주간 간격 (median/mean/std)
  - 두둑별 균일도 (CV = std/mean)
  - 두둑 정렬 길이

출력:
  result/sam_test/comparison_final/
    ├─ smart_card.png            (Smart 요약 카드, 겹침 없음)
    ├─ normal_card.png           (Normal 요약 카드)
    ├─ side_by_side.png          (두 필지 나란히 대시보드)
    ├─ per_ridge_analysis.png    (두둑별 상세 지표 4개 subplot)
    ├─ summary.md                (마크다운 종합 요약)
    └─ summary.json              (구조화 통계)
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
CMP_DIR = OUT_DIR / "comparison_final"
CMP_DIR.mkdir(parents=True, exist_ok=True)

# 필터 파라미터
RIDGE_SPACING_MIN_CM = 20
RIDGE_KEEP_CM = 20
HIST_BIN_CM = 1.0
HIST_SMOOTH_SIGMA = 0.5
PEAK_HEIGHT_FRAC = 0.03
DEDUP_RADIUS_CM = 5.0

INTRA_ROW_SPACING_CM = 20.0
ROWS_PER_RIDGE = 2


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
                ridge_pos_px=ridge_pos_px, px_to_cm=px_to_cm)


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


def per_ridge_stats(ri, keep_thresh_px):
    """각 두둑에 대해 잎 수·길이·주간 간격 통계."""
    px_to_cm = ri["px_to_cm"]
    stats = []
    for idx, rp in enumerate(ri["ridge_pos_px"]):
        on = np.abs(ri["proj_perp"] - rp) <= keep_thresh_px
        n = int(on.sum())
        if n < 2:
            continue
        along = np.sort(ri["proj_par"][on])
        length_cm = float((along[-1] - along[0]) * px_to_cm)
        gaps_cm = np.diff(along) * px_to_cm
        gaps_valid = gaps_cm[(gaps_cm > 3) & (gaps_cm <= 60)]  # 3-60cm valid
        median_gap = float(np.median(gaps_valid)) if len(gaps_valid) else np.nan
        mean_gap = float(np.mean(gaps_valid)) if len(gaps_valid) else np.nan
        std_gap = float(np.std(gaps_valid)) if len(gaps_valid) else np.nan
        cv = std_gap / mean_gap if mean_gap and mean_gap > 0 else np.nan
        leaves_per_m = n / (length_cm / 100) if length_cm > 0 else 0
        stats.append(dict(
            ridge_idx=idx, n_leaves=n, length_m=length_cm / 100,
            leaves_per_m=leaves_per_m,
            median_gap_cm=median_gap, mean_gap_cm=mean_gap,
            std_gap_cm=std_gap, cv=cv,
        ))
    return stats


def analyze_full(field_name: str):
    npz_path = OUT_DIR / f"{field_name}_sam_FULL_v4.npz"
    d = np.load(npz_path)
    leaves = d["leaf_arr"]
    gsd = float(d["gsd_ds"][0])
    valid_area_ha = float(d["valid_area_ha"][0])
    rgb_disp = d["rgb_disp"]
    step = int(d["rgb_step"][0])

    centroids = leaves[:, :2]
    areas = leaves[:, 2]

    ri = compute_ridges(centroids, gsd)
    px_to_cm = ri["px_to_cm"]
    keep_thresh_px = RIDGE_KEEP_CM / px_to_cm

    # A. 두둑 필터
    rp = ri["ridge_pos_px"]
    dists = np.abs(ri["proj_perp"][:, None] - rp[None, :]).min(axis=1)
    keep_A = dists <= keep_thresh_px
    ridge_idx = np.where(keep_A)[0]
    # B. Dedup
    ridge_c = centroids[ridge_idx]
    ridge_a = areas[ridge_idx]
    reps = dedup_radius(ridge_c, ridge_a, DEDUP_RADIUS_CM / px_to_cm)
    final_idx = ridge_idx[reps]
    final_c = centroids[final_idx]

    # ─── 두둑별 통계는 dedup 후 잎으로 다시 계산 ───
    final_centered = final_c - ri["mean"]
    ri_final = dict(ri)
    ri_final["proj_perp"] = final_centered @ ri["perp_dir"]
    ri_final["proj_par"] = final_centered @ ri["ridge_dir"]
    per_ridge = per_ridge_stats(ri_final, keep_thresh_px)

    # 두둑 간격
    ridge_spacings_cm = np.diff(rp) * px_to_cm if len(rp) > 1 else np.array([])

    density = len(final_c) / valid_area_ha

    return dict(
        field=field_name,
        rgb_disp=rgb_disp, step=step,
        n_leaves_final=len(final_c),
        final_centroids=final_c,
        valid_area_ha=valid_area_ha,
        density_per_ha=density,
        n_ridges=len(rp),
        ridge_spacing_mean_cm=float(ridge_spacings_cm.mean()) if len(ridge_spacings_cm) else np.nan,
        ridge_spacing_median_cm=float(np.median(ridge_spacings_cm)) if len(ridge_spacings_cm) else np.nan,
        ridge_angle_deg=float(ri["ridge_angle_deg"]),
        per_ridge=per_ridge,
    )


def field_card(res: dict, out_png: Path, color_theme: str):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.15)

    ax_txt = fig.add_subplot(gs[0, 0]); ax_txt.axis("off")
    ax_img = fig.add_subplot(gs[0, 1])

    # 필지명 헤더
    ax_txt.text(0.02, 0.96, res["field"], fontsize=24, fontweight="bold",
                color=color_theme, transform=ax_txt.transAxes,
                verticalalignment="top")
    ax_txt.text(0.02, 0.90, f"전체 필지 · {res['valid_area_ha']:.2f} ha 유효",
                fontsize=11, color="gray", transform=ax_txt.transAxes,
                verticalalignment="top")

    # per-ridge 집계
    pr = res["per_ridge"]
    n_per_ridge = np.array([r["n_leaves"] for r in pr])
    lpm = np.array([r["leaves_per_m"] for r in pr])
    med_gaps = np.array([r["median_gap_cm"] for r in pr if not np.isnan(r["median_gap_cm"])])
    cvs = np.array([r["cv"] for r in pr if not np.isnan(r["cv"])])
    lengths = np.array([r["length_m"] for r in pr])

    metrics = [
        ("총 발아 콩잎",   f"{res['n_leaves_final']:,} 개"),
        ("전체 재식밀도",  f"{res['density_per_ha']:,.0f} /ha"),
        ("검출 두둑 수",   f"{res['n_ridges']}개"),
        ("두둑 간격 (평균)", f"{res['ridge_spacing_mean_cm']:.1f} cm"),
        ("두둑당 평균 잎",  f"{n_per_ridge.mean():.0f} 개  (median {np.median(n_per_ridge):.0f})"),
        ("두둑당 잎/m",   f"{lpm.mean():.1f} 개/m  (median {np.median(lpm):.1f})"),
        ("두둑 평균 길이", f"{lengths.mean():.1f} m"),
        ("주간 간격 (median)", f"{np.median(med_gaps):.1f} cm"),
        ("주간 균일도 CV",   f"{cvs.mean():.2f}  (낮을수록 균일)"),
    ]
    y = 0.80
    for k, v in metrics:
        ax_txt.text(0.02, y, k, fontsize=12, color="#444",
                    transform=ax_txt.transAxes, verticalalignment="top")
        ax_txt.text(0.48, y, v, fontsize=13, fontweight="bold",
                    transform=ax_txt.transAxes, verticalalignment="top")
        y -= 0.065

    # RGB + green dots
    ax_img.imshow(res["rgb_disp"])
    fc = res["final_centroids"]
    step = res["step"]
    ax_img.scatter(fc[:, 1] / step, fc[:, 0] / step, s=2,
                   c="lime", alpha=0.7, edgecolor="none")
    ax_img.axis("off")
    ax_img.set_title(f"발아 콩잎 위치 ({res['n_leaves_final']:,}개)  ·  "
                     f"두둑 {res['n_ridges']}줄  각도 {res['ridge_angle_deg']:.0f}°",
                     fontsize=13)

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def side_by_side(smart: dict, normal: dict, out_png: Path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.15)

    # 상단: 두 필지 RGB + dots
    for i, (res, color, title_c) in enumerate([
        (smart, "lime", "#0066cc"),
        (normal, "lime", "#cc6600"),
    ]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(res["rgb_disp"])
        fc = res["final_centroids"]
        step = res["step"]
        ax.scatter(fc[:, 1] / step, fc[:, 0] / step, s=2,
                   c=color, alpha=0.7, edgecolor="none")
        ax.axis("off")
        ax.set_title(f"{res['field']}   ·   "
                     f"발아잎 {res['n_leaves_final']:,}개   ·   "
                     f"재식밀도 {res['density_per_ha']:,.0f}/ha   ·   "
                     f"두둑 {res['n_ridges']}줄",
                     fontsize=13, color=title_c, fontweight="bold")

    # 하단: 지표별 막대 비교
    ax_b = fig.add_subplot(gs[1, :])
    # 지표 정의
    def agg(res, key, op="mean"):
        arr = np.array([r[key] for r in res["per_ridge"]
                        if not np.isnan(r.get(key, np.nan))])
        return arr.mean() if op == "mean" else np.median(arr)

    labels = [
        "총 발아잎(천)",
        "재식밀도(만/ha)",
        "두둑당 평균잎",
        "두둑당 잎/m",
        "주간 median(cm)",
        "균일도 CV(x10)",   # x10 스케일 맞춤
    ]
    s_vals = [
        smart["n_leaves_final"] / 1000,
        smart["density_per_ha"] / 10000,
        agg(smart, "n_leaves"),
        agg(smart, "leaves_per_m"),
        agg(smart, "median_gap_cm", "median"),
        agg(smart, "cv") * 10,
    ]
    n_vals = [
        normal["n_leaves_final"] / 1000,
        normal["density_per_ha"] / 10000,
        agg(normal, "n_leaves"),
        agg(normal, "leaves_per_m"),
        agg(normal, "median_gap_cm", "median"),
        agg(normal, "cv") * 10,
    ]
    x = np.arange(len(labels)); w = 0.35
    b1 = ax_b.bar(x - w/2, s_vals, w, color="#4477bb", label="Smart")
    b2 = ax_b.bar(x + w/2, n_vals, w, color="#dd7733", label="Normal")
    for bars, vals in [(b1, s_vals), (b2, n_vals)]:
        for b, v in zip(bars, vals):
            ax_b.text(b.get_x() + b.get_width() / 2, b.get_height(),
                      f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax_b.set_xticks(x); ax_b.set_xticklabels(labels)
    ax_b.set_title("주요 지표 비교 (Smart vs Normal)  ·  각 지표별 값이 클수록 "
                   "일반적으로 우수 (단, CV는 낮을수록 균일)",
                   fontsize=12, fontweight="bold")
    ax_b.grid(True, axis="y", alpha=0.3); ax_b.legend(fontsize=11)

    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def per_ridge_analysis(smart: dict, normal: dict, out_png: Path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def arr(res, key):
        return np.array([r[key] for r in res["per_ridge"]
                         if not np.isnan(r.get(key, np.nan))])

    # (1) 두둑당 잎 수 분포
    ax = axes[0, 0]
    s = arr(smart, "n_leaves"); n = arr(normal, "n_leaves")
    bins = np.linspace(0, max(s.max(), n.max()) + 20, 30)
    ax.hist(s, bins=bins, alpha=0.6, color="#4477bb",
            label=f"Smart (mean {s.mean():.0f})")
    ax.hist(n, bins=bins, alpha=0.6, color="#dd7733",
            label=f"Normal (mean {n.mean():.0f})")
    ax.set_xlabel("두둑당 잎 수"); ax.set_ylabel("두둑 개수")
    ax.set_title("두둑당 잎 수 분포", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # (2) 잎/m 분포
    ax = axes[0, 1]
    s = arr(smart, "leaves_per_m"); n = arr(normal, "leaves_per_m")
    bins = np.linspace(0, max(s.max(), n.max()) + 1, 30)
    ax.hist(s, bins=bins, alpha=0.6, color="#4477bb",
            label=f"Smart (mean {s.mean():.1f}/m)")
    ax.hist(n, bins=bins, alpha=0.6, color="#dd7733",
            label=f"Normal (mean {n.mean():.1f}/m)")
    ax.set_xlabel("두둑당 잎 밀도 (개/m)"); ax.set_ylabel("두둑 개수")
    ax.set_title("두둑 내 잎 밀도 분포 (선형 밀도)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # (3) 주간 median 분포
    ax = axes[1, 0]
    s = arr(smart, "median_gap_cm"); n = arr(normal, "median_gap_cm")
    bins = np.linspace(0, max(s.max(), n.max()) + 5, 30)
    ax.hist(s, bins=bins, alpha=0.6, color="#4477bb",
            label=f"Smart (median {np.median(s):.1f}cm)")
    ax.hist(n, bins=bins, alpha=0.6, color="#dd7733",
            label=f"Normal (median {np.median(n):.1f}cm)")
    ax.axvline(INTRA_ROW_SPACING_CM, color="red", ls="--",
               label=f"스펙 {INTRA_ROW_SPACING_CM:.0f}cm")
    ax.set_xlabel("두둑 내 주간 median (cm)"); ax.set_ylabel("두둑 개수")
    ax.set_title("두둑별 주간 간격 분포", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # (4) CV 분포 (균일도)
    ax = axes[1, 1]
    s = arr(smart, "cv"); n = arr(normal, "cv")
    bins = np.linspace(0, max(s.max(), n.max()) + 0.1, 30)
    ax.hist(s, bins=bins, alpha=0.6, color="#4477bb",
            label=f"Smart (mean {s.mean():.2f})")
    ax.hist(n, bins=bins, alpha=0.6, color="#dd7733",
            label=f"Normal (mean {n.mean():.2f})")
    ax.set_xlabel("두둑 내 주간 CV (낮을수록 균일)"); ax.set_ylabel("두둑 개수")
    ax.set_title("두둑별 주간 균일도 (CV = std/mean)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("파종기 두둑별 상세 지표 분포",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def make_summary(smart, normal) -> str:
    def agg(res, key, op="mean"):
        arr = np.array([r[key] for r in res["per_ridge"]
                        if not np.isnan(r.get(key, np.nan))])
        return arr.mean() if op == "mean" else np.median(arr)

    L = []
    L.append("# Smart vs Normal 파종기 종합 비교\n")
    L.append("**데이터**: 전체 필지 처리 결과 (FULL_v4)  ·  "
             "**필터**: 두둑(A) + Dedup 5cm(B)  ·  **잡초 필터**: 비활성\n")
    L.append("## 필지 개요\n")
    L.append("| 항목 | Smart | Normal |")
    L.append("|---|---:|---:|")
    L.append(f"| 유효 면적 | {smart['valid_area_ha']:.2f} ha | {normal['valid_area_ha']:.2f} ha |")
    L.append(f"| 발아 콩잎 (최종) | {smart['n_leaves_final']:,} | {normal['n_leaves_final']:,} |")
    L.append(f"| 재식밀도 | {smart['density_per_ha']:,.0f} /ha | {normal['density_per_ha']:,.0f} /ha |")
    L.append(f"| 검출 두둑 수 | {smart['n_ridges']} | {normal['n_ridges']} |")
    L.append(f"| 두둑 간격 (평균) | {smart['ridge_spacing_mean_cm']:.1f} cm | {normal['ridge_spacing_mean_cm']:.1f} cm |")
    L.append(f"| 두둑 각도 | {smart['ridge_angle_deg']:.0f}° | {normal['ridge_angle_deg']:.0f}° |")

    L.append("\n## 두둑별 세부 지표 (전체 두둑 집계)\n")
    L.append("| 지표 | Smart | Normal | 승자 |")
    L.append("|---|---:|---:|:---:|")
    s_n = agg(smart, "n_leaves"); n_n = agg(normal, "n_leaves")
    L.append(f"| 두둑당 평균 잎 수 | {s_n:.1f} | {n_n:.1f} | "
             f"{'**Smart**' if s_n > n_n else '**Normal**'} |")
    s_lpm = agg(smart, "leaves_per_m"); n_lpm = agg(normal, "leaves_per_m")
    L.append(f"| 두둑당 잎 밀도 (개/m) | {s_lpm:.1f} | {n_lpm:.1f} | "
             f"{'**Smart**' if s_lpm > n_lpm else '**Normal**'} |")
    s_g = agg(smart, "median_gap_cm", "median"); n_g = agg(normal, "median_gap_cm", "median")
    L.append(f"| 두둑 내 주간 median (cm) | {s_g:.1f} | {n_g:.1f} | "
             f"{'**Smart**' if abs(s_g - 20) < abs(n_g - 20) else '**Normal**'} "
             f"(20cm 근접) |")
    s_cv = agg(smart, "cv"); n_cv = agg(normal, "cv")
    L.append(f"| 주간 균일도 CV (낮을수록 균일) | {s_cv:.3f} | {n_cv:.3f} | "
             f"{'**Smart**' if s_cv < n_cv else '**Normal**'} |")
    s_len = agg(smart, "length_m"); n_len = agg(normal, "length_m")
    L.append(f"| 두둑 평균 길이 (m) | {s_len:.1f} | {n_len:.1f} | - |")

    L.append("\n## 종합 판정\n")
    scores = {"Smart": 0, "Normal": 0}
    if s_n > n_n: scores["Smart"] += 1
    else: scores["Normal"] += 1
    if s_lpm > n_lpm: scores["Smart"] += 1
    else: scores["Normal"] += 1
    if abs(s_g - 20) < abs(n_g - 20): scores["Smart"] += 1
    else: scores["Normal"] += 1
    if s_cv < n_cv: scores["Smart"] += 1
    else: scores["Normal"] += 1
    if smart["density_per_ha"] > normal["density_per_ha"]: scores["Smart"] += 1
    else: scores["Normal"] += 1
    L.append(f"- 5개 지표 승수: **Smart {scores['Smart']} · Normal {scores['Normal']}**")
    winner = max(scores, key=scores.get)
    L.append(f"- 종합 우세: **{winner}**\n")
    L.append("*지표별 정의:*")
    L.append("- **재식밀도**: 전체 면적당 총 개체수 (많을수록 우수)")
    L.append("- **두둑당 잎 수/잎 밀도(개/m)**: 개별 두둑 성능 (많을수록 우수)")
    L.append("- **주간 median 20cm 근접**: 파종 스펙(20cm) 준수 정확도")
    L.append("- **CV**: 두둑 내 주간 간격의 변동계수. 낮을수록 균일한 파종")
    return "\n".join(L)


def main():
    print("=" * 60); print("Smart vs Normal 종합 비교 (전체 필지)"); print("=" * 60)

    smart = analyze_full("GJSM-1-1_Smart")
    normal = analyze_full("GJSM-1-1_normal")

    print(f"\nSmart: 잎 {smart['n_leaves_final']:,}, "
          f"밀도 {smart['density_per_ha']:,.0f}/ha, 두둑 {smart['n_ridges']}")
    print(f"Normal: 잎 {normal['n_leaves_final']:,}, "
          f"밀도 {normal['density_per_ha']:,.0f}/ha, 두둑 {normal['n_ridges']}")

    field_card(smart, CMP_DIR / "smart_card.png", "#0066cc")
    field_card(normal, CMP_DIR / "normal_card.png", "#cc6600")
    print(f"✅ 필지별 카드")

    side_by_side(smart, normal, CMP_DIR / "side_by_side.png")
    print(f"✅ 나란히 대시보드")

    per_ridge_analysis(smart, normal, CMP_DIR / "per_ridge_analysis.png")
    print(f"✅ 두둑별 상세 분포")

    md = make_summary(smart, normal)
    (CMP_DIR / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    # JSON (numpy 정리)
    def clean(r):
        return {k: v for k, v in r.items()
                if k not in ("rgb_disp", "final_centroids")}
    (CMP_DIR / "summary.json").write_text(
        json.dumps({"smart": clean(smart), "normal": clean(normal)},
                   ensure_ascii=False, indent=2, default=lambda x: float(x)
                   if isinstance(x, (np.floating,)) else str(x)),
        encoding="utf-8")

    print(f"\n📁 저장 위치: {CMP_DIR}")


if __name__ == "__main__":
    main()
