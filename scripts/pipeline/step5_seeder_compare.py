"""
Step 5 - 파종기 효율성 비교 리포트.

비교 지표:
  - 파종 간격 (median, mean, std, **CV** = std/mean × 100%)
    → CV가 낮을수록 균일 파종
  - 결주율 (miss_gap 개수 / 전체 인접 gap 개수)
  - 입모율 (검출 잎 / 기대 잎)
  - 조간 (dual-row 두 파종 줄 간격)

산출:
  result/report/seeder_comparison.png   - 4패널 비교 시각화
  result/report/seeder_stats.csv        - 파종기별·필지별 통계 표
  result/report/seeder_report.md        - 마크다운 요약 리포트

사용:
  python -u scripts/pipeline/step5_seeder_compare.py <field_A> <field_B> \\
      [--labels "라벨A,라벨B"]

예:
  python -u scripts/pipeline/step5_seeder_compare.py \\
      GJSM-1-1_Smart GJSM-1-1_normal --labels "스마트 파종기,일반 파종기"
"""
from __future__ import annotations
import sys
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.common import get_field_dir, setup_korean_font, REPORT_DIR

SPACING_SPEC_CM = 20            # 파종 간격 스펙
SPACING_TOL_CM = 5              # 적합 ±5cm

COLOR = {"A": "#F58C39", "B": "#8E44AD"}   # A=주황, B=보라


def load_field(field: str) -> dict:
    """필지 pipeline 산출(ridges + rows + spacing) 로드 후 지표 계산."""
    fdir = get_field_dir(field)
    R = np.load(fdir / "ridges.npz", allow_pickle=True)
    S = np.load(fdir / "rows.npz", allow_pickle=True)
    P = np.load(fdir / "spacing.npz", allow_pickle=True)

    ridge_arr = R["ridge_arr"]
    row_arr = S["row_arr"]
    row_types = S["type_arr"]
    stats_arr = P["stats_arr"]        # [rid, ridx, n, median, mean, std, miss, exp, emg, len, perp]
    stats_types = P["stats_types"]
    positions_obj = P["positions_obj"]

    # per-row 통계 리스트
    rows = []
    for i, r in enumerate(stats_arr):
        rows.append(dict(
            ridge_id=int(r[0]), row_idx=int(r[1]),
            n_leaves=int(r[2]),
            median_cm=float(r[3]), mean_cm=float(r[4]), std_cm=float(r[5]),
            miss_gaps=int(r[6]), expected=int(r[7]),
            emergence_pct=float(r[8]), length_cm=float(r[9]),
            type=str(stats_types[i]),
            positions_cm=positions_obj[i],
        ))
    # 조간 정보 (row_arr: [rid, n_peaks, p1, p2, gap_cm, n_leaves])
    inter_row_gaps = [float(r[4]) for r in row_arr if not np.isnan(r[4])]

    # 파종 간격 gap 리스트 (모든 두둑·줄 통합)
    # all_intra_gaps: 원본 전체 (결주 포함)
    # normal_intra_gaps: 결주 제외 (스펙 근처 10~35cm) — 균일성 CV 산출용
    # per_row_cv: row별 std/mean × 100 (결주 제외 gap 기준) — 진짜 균일성 지표
    all_intra_gaps = []
    normal_intra_gaps = []
    all_miss_gaps = []
    per_row_cv = []              # row별 CV (스펙 근처 median row만)
    GAP_NORMAL_MIN = 10          # cm
    GAP_NORMAL_MAX = 35          # cm
    # 결주 판정: 스펙 절대값 기준 (스펙 20cm × 1.5 = 30cm 초과)
    # 이렇게 하면 필드별 median이 달라도 공정하게 비교됨
    MISS_ABS_CM = SPACING_SPEC_CM * 1.5   # = 30cm
    for r in rows:
        if r["n_leaves"] < 2 or len(r["positions_cm"]) < 2:
            continue
        gaps = np.diff(r["positions_cm"])
        all_intra_gaps.extend(gaps.tolist())
        med = r["median_cm"]
        # 결주 gap: 스펙 절대 기준 30cm 초과
        all_miss_gaps.extend(gaps[gaps > MISS_ABS_CM].tolist())
        # 정상 gap 필터 (global 10~35cm)
        normal_mask = (gaps >= GAP_NORMAL_MIN) & (gaps <= GAP_NORMAL_MAX)
        normal_gaps_row = gaps[normal_mask]
        normal_intra_gaps.extend(normal_gaps_row.tolist())

        # row별 CV — median이 스펙 근처(10~35cm)이고 정상 gap 5개 이상일 때만
        if (GAP_NORMAL_MIN <= med <= GAP_NORMAL_MAX
                and len(normal_gaps_row) >= 5
                and normal_gaps_row.mean() > 0):
            per_row_cv.append(
                float(normal_gaps_row.std() / normal_gaps_row.mean() * 100))

    # 요약 통계
    def stats(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return dict(n=0, median=np.nan, mean=np.nan, std=np.nan, cv=np.nan)
        return dict(
            n=len(arr), median=float(np.median(arr)),
            mean=float(arr.mean()), std=float(arr.std()),
            cv=float(arr.std() / arr.mean() * 100) if arr.mean() > 0 else np.nan,
        )

    total_gaps = len(all_intra_gaps)
    n_miss = len(all_miss_gaps)
    miss_rate_pct = 100.0 * n_miss / total_gaps if total_gaps else np.nan

    # 유형별 필터
    narrow_rows = [r for r in rows if r["type"] == "narrow"]
    wide_rows = [r for r in rows if r["type"] == "wide"]
    valid_rows = [r for r in rows if r["type"] in ("narrow", "wide")
                   and r["n_leaves"] >= 5]

    # median 파종 간격 (줄별 median의 median)
    med_all = [r["median_cm"] for r in valid_rows]
    med_narrow = [r["median_cm"] for r in narrow_rows if r["n_leaves"] >= 5]
    med_wide = [r["median_cm"] for r in wide_rows if r["n_leaves"] >= 5]

    # 입모율
    emg_valid = [r["emergence_pct"] for r in valid_rows]

    # 적합 줄 카운트
    adequate = [r for r in valid_rows if SPACING_SPEC_CM - SPACING_TOL_CM
                <= r["median_cm"] <= SPACING_SPEC_CM + SPACING_TOL_CM]

    return dict(
        field=field,
        n_ridges=len(ridge_arr), n_rows_total=len(rows),
        n_narrow=len(narrow_rows), n_wide=len(wide_rows),
        n_valid=len(valid_rows),
        inter_row_gap_stats=stats(inter_row_gaps),
        intra_gap_stats=stats(all_intra_gaps),           # 원본 (결주 포함)
        normal_intra_gap_stats=stats(normal_intra_gaps), # 결주 제외 (균일성)
        per_row_cv_median=float(np.median(per_row_cv)) if per_row_cv else np.nan,
        per_row_cv_n=len(per_row_cv),
        med_row_median_stats=stats(med_all),
        med_narrow_stats=stats(med_narrow),
        med_wide_stats=stats(med_wide),
        emergence_stats=stats(emg_valid),
        miss_gaps=n_miss, total_gaps=total_gaps,
        miss_rate_pct=miss_rate_pct,
        n_adequate=len(adequate),
        adequate_rate_pct=100.0 * len(adequate) / len(valid_rows) if valid_rows else np.nan,
        rows=rows,
        _intra_gaps=all_intra_gaps,
        _normal_intra_gaps=normal_intra_gaps,
        _med_all=med_all,
        _emg_valid=emg_valid,
    )


def visualize(A: dict, B: dict, labels: tuple[str, str], out_png: Path):
    setup_korean_font()
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0])

    def hist_compare(ax, A_vals, B_vals, bins, xlabel, title,
                      spec_span=None, spec_line=None):
        A_vals = np.asarray(A_vals); B_vals = np.asarray(B_vals)
        if spec_span:
            ax.axvspan(*spec_span, color="green", alpha=0.12,
                       label=f"적합 {spec_span[0]:.0f}~{spec_span[1]:.0f}")
        if spec_line is not None:
            ax.axvline(spec_line, color="green", ls="--", lw=1.5)
        if len(A_vals):
            ax.hist(A_vals, bins=bins, color=COLOR["A"], alpha=0.75,
                     edgecolor="darkorange",
                     label=f"{labels[0]} (n={len(A_vals)})")
        if len(B_vals):
            ax.hist(B_vals, bins=bins, color=COLOR["B"], alpha=0.55,
                     edgecolor="indigo",
                     label=f"{labels[1]} (n={len(B_vals)})")
        # median 라인
        if len(A_vals):
            ax.axvline(np.median(A_vals), color=COLOR["A"], lw=2,
                       ls="-", alpha=0.9)
        if len(B_vals):
            ax.axvline(np.median(B_vals), color=COLOR["B"], lw=2,
                       ls="-", alpha=0.9)
        ax.set_xlabel(xlabel); ax.set_ylabel("빈도")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (1) 줄별 median 파종 간격 분포
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.arange(0, 60, 2)
    hist_compare(ax1, A["_med_all"], B["_med_all"], bins,
                  "줄별 median 파종 간격 (cm)",
                  "① 파종 간격 median 분포",
                  spec_span=(SPACING_SPEC_CM - SPACING_TOL_CM,
                              SPACING_SPEC_CM + SPACING_TOL_CM),
                  spec_line=SPACING_SPEC_CM)

    # (2) 모든 인접 간격 분포 (raw)
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.arange(0, 80, 2)
    hist_compare(ax2, A["_intra_gaps"], B["_intra_gaps"], bins,
                  "인접 잎 간격 (cm) — 로그 y", "② 원본 잎 간격 분포",
                  spec_line=SPACING_SPEC_CM)
    ax2.set_yscale("log")

    # (3) 입모율 분포
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.arange(0, 200, 10)
    hist_compare(ax3, A["_emg_valid"], B["_emg_valid"], bins,
                  "줄별 입모율 (%)", "③ 입모율 분포",
                  spec_line=100)

    # (4) 지표 bar chart
    ax4 = fig.add_subplot(gs[1, :])
    metrics = [
        ("파종 median\n(cm)",
         A["med_row_median_stats"]["median"], B["med_row_median_stats"]["median"],
         f"스펙 {SPACING_SPEC_CM}"),
        ("파종 CV\n(%, 줄별 CV median, 낮을수록 균일)",
         A["per_row_cv_median"],
         B["per_row_cv_median"], ""),
        ("결주율\n(%)",
         A["miss_rate_pct"], B["miss_rate_pct"], ""),
        ("입모율 median\n(%)",
         A["emergence_stats"]["median"], B["emergence_stats"]["median"], "스펙 100"),
        ("적합 줄 비율\n(%, 15~25cm)",
         A["adequate_rate_pct"], B["adequate_rate_pct"], ""),
        ("조간 median\n(cm)",
         A["inter_row_gap_stats"]["median"], B["inter_row_gap_stats"]["median"], ""),
    ]
    x = np.arange(len(metrics))
    width = 0.35
    Av = [m[1] for m in metrics]; Bv = [m[2] for m in metrics]
    b1 = ax4.bar(x - width/2, Av, width, color=COLOR["A"],
                  edgecolor="darkorange", label=labels[0])
    b2 = ax4.bar(x + width/2, Bv, width, color=COLOR["B"],
                  edgecolor="indigo", label=labels[1])
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h):
            ax4.text(bar.get_x() + bar.get_width()/2, h * 1.02,
                      f"{h:.1f}", ha="center", va="bottom", fontsize=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels([m[0] for m in metrics], fontsize=10)
    for i, m in enumerate(metrics):
        if m[3]:
            ax4.text(i, ax4.get_ylim()[1] * 0.95, m[3], fontsize=9,
                     ha="center", color="green", alpha=0.8)
    ax4.set_ylabel("값")
    ax4.set_title("파종기별 지표 비교 — 파종 CV 낮음 = 균일 파종, 결주율 낮음 = 결주 적음",
                   fontsize=12, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=10); ax4.grid(alpha=0.3, axis="y")

    # (5) 요약 표
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    def row_line(label, key_A, key_B, unit="", fmt="{:.1f}"):
        vA = key_A if isinstance(key_A, (int, float)) else np.nan
        vB = key_B if isinstance(key_B, (int, float)) else np.nan
        return [label,
                fmt.format(vA) + unit if not np.isnan(vA) else "-",
                fmt.format(vB) + unit if not np.isnan(vB) else "-"]

    table_data = [
        ["지표", labels[0], labels[1]],
        ["두둑 개수", str(A["n_ridges"]), str(B["n_ridges"])],
        ["파종 줄 개수 (유효)",
         f"{A['n_valid']} / {A['n_rows_total']}",
         f"{B['n_valid']} / {B['n_rows_total']}"],
        ["  narrow / wide",
         f"{A['n_narrow']} / {A['n_wide']}",
         f"{B['n_narrow']} / {B['n_wide']}"],
        ["조간 median (cm)",
         f"{A['inter_row_gap_stats']['median']:.1f}",
         f"{B['inter_row_gap_stats']['median']:.1f}"],
        ["파종 간격 median (cm)",
         f"{A['med_row_median_stats']['median']:.1f}",
         f"{B['med_row_median_stats']['median']:.1f}"],
        ["정상 gap mean (cm) — 결주 제외",
         f"{A['normal_intra_gap_stats']['mean']:.1f}",
         f"{B['normal_intra_gap_stats']['mean']:.1f}"],
        ["정상 gap std (cm) — 결주 제외",
         f"{A['normal_intra_gap_stats']['std']:.1f}",
         f"{B['normal_intra_gap_stats']['std']:.1f}"],
        ["파종 CV (%) ★ 줄별 CV median",
         f"{A['per_row_cv_median']:.1f} (n={A['per_row_cv_n']})",
         f"{B['per_row_cv_median']:.1f} (n={B['per_row_cv_n']})"],
        ["결주 gap 수",
         str(A['miss_gaps']),
         str(B['miss_gaps'])],
        ["결주율 (%) ★",
         f"{A['miss_rate_pct']:.1f}",
         f"{B['miss_rate_pct']:.1f}"],
        ["입모율 median (%)",
         f"{A['emergence_stats']['median']:.1f}",
         f"{B['emergence_stats']['median']:.1f}"],
        ["적합 줄 비율 (%) ★ 15~25cm",
         f"{A['adequate_rate_pct']:.1f}",
         f"{B['adequate_rate_pct']:.1f}"],
    ]
    table = ax5.table(cellText=table_data, loc="center", cellLoc="center",
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False); table.set_fontsize(11)
    table.scale(1, 1.6)
    # 헤더 색
    for j in range(3):
        cell = table[(0, j)]
        cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white",
                                                            fontweight="bold")
    # A 컬럼 색
    for i in range(1, len(table_data)):
        table[(i, 1)].set_facecolor("#FCE8D6")
        table[(i, 2)].set_facecolor("#EBDEF0")

    plt.suptitle(
        f"Step 5: 파종기 효율성 비교 — {labels[0]} vs {labels[1]}\n"
        f"(파종 CV 낮음 = 균일 파종, 결주율 낮음 = 결주 적음, 적합 줄 비율 높음 = 정확도)",
        fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def write_csv(A, B, labels, out_csv: Path):
    rows = [
        ["metric", labels[0], labels[1]],
        ["field", A["field"], B["field"]],
        ["n_ridges", A["n_ridges"], B["n_ridges"]],
        ["n_rows_total", A["n_rows_total"], B["n_rows_total"]],
        ["n_valid_rows", A["n_valid"], B["n_valid"]],
        ["n_narrow", A["n_narrow"], B["n_narrow"]],
        ["n_wide", A["n_wide"], B["n_wide"]],
        ["inter_row_gap_median_cm",
         A["inter_row_gap_stats"]["median"], B["inter_row_gap_stats"]["median"]],
        ["intra_spacing_median_cm",
         A["med_row_median_stats"]["median"], B["med_row_median_stats"]["median"]],
        ["intra_spacing_mean_cm_normal",
         A["normal_intra_gap_stats"]["mean"], B["normal_intra_gap_stats"]["mean"]],
        ["intra_spacing_std_cm_normal",
         A["normal_intra_gap_stats"]["std"], B["normal_intra_gap_stats"]["std"]],
        ["intra_spacing_CV_pct_normal_pooled",
         A["normal_intra_gap_stats"]["cv"], B["normal_intra_gap_stats"]["cv"]],
        ["per_row_CV_median_pct",
         A["per_row_cv_median"], B["per_row_cv_median"]],
        ["per_row_CV_n_rows",
         A["per_row_cv_n"], B["per_row_cv_n"]],
        ["miss_gap_count", A["miss_gaps"], B["miss_gaps"]],
        ["miss_rate_pct", A["miss_rate_pct"], B["miss_rate_pct"]],
        ["emergence_median_pct",
         A["emergence_stats"]["median"], B["emergence_stats"]["median"]],
        ["adequate_row_ratio_pct",
         A["adequate_rate_pct"], B["adequate_rate_pct"]],
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def write_md(A, B, labels, out_md: Path):
    def fmt(v, u=""):
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) \
                else f"{v:.1f}{u}"

    def winner_label(a, b, lower_is_better=True):
        if np.isnan(a) or np.isnan(b):
            return "-"
        if a == b:
            return "동률"
        if lower_is_better:
            return labels[0] if a < b else labels[1]
        return labels[0] if a > b else labels[1]

    lines = [
        f"# 파종기 효율성 비교 리포트",
        "",
        f"- 대상: **{labels[0]}** ({A['field']}) vs **{labels[1]}** ({B['field']})",
        f"- 스펙: 파종 간격 {SPACING_SPEC_CM}cm (적합 ±{SPACING_TOL_CM}cm)",
        f"- 판정 기준:",
        f"  - 파종 CV(%): 줄별 CV (정상 gap 10~35cm 기준) 의 median — **낮을수록 균일**",
        f"  - 결주율(%): 스펙 30cm(=SPACING_SPEC×1.5) 초과 gap 개수 / 전체 gap 개수 × 100 — **낮을수록 좋음**",
        f"  - 적합 줄 비율(%): 줄별 median이 {SPACING_SPEC_CM-SPACING_TOL_CM}~{SPACING_SPEC_CM+SPACING_TOL_CM}cm에 들어간 줄 비율 — **높을수록 좋음**",
        "",
        "## 1. 두둑·줄 검출",
        "",
        f"| 지표 | {labels[0]} | {labels[1]} |",
        f"|---|---:|---:|",
        f"| 두둑 개수 | {A['n_ridges']} | {B['n_ridges']} |",
        f"| 파종 줄 (유효) | {A['n_valid']}/{A['n_rows_total']} | {B['n_valid']}/{B['n_rows_total']} |",
        f"| narrow / wide 두둑 | {A['n_narrow']} / {A['n_wide']} | {B['n_narrow']} / {B['n_wide']} |",
        f"| 조간 median | {fmt(A['inter_row_gap_stats']['median'], 'cm')} | {fmt(B['inter_row_gap_stats']['median'], 'cm')} |",
        "",
        "## 2. 파종 간격 (핵심 비교)",
        "",
        f"| 지표 | {labels[0]} | {labels[1]} | 우수 |",
        f"|---|---:|---:|:---:|",
        f"| 파종 median | {fmt(A['med_row_median_stats']['median'], 'cm')} | {fmt(B['med_row_median_stats']['median'], 'cm')} | 스펙 {SPACING_SPEC_CM}cm에 가까운 쪽 |",
        f"| 정상 gap mean (결주 제외) | {fmt(A['normal_intra_gap_stats']['mean'], 'cm')} | {fmt(B['normal_intra_gap_stats']['mean'], 'cm')} | - |",
        f"| 정상 gap std (결주 제외) | {fmt(A['normal_intra_gap_stats']['std'], 'cm')} | {fmt(B['normal_intra_gap_stats']['std'], 'cm')} | - |",
        f"| **파종 CV** ★ 줄별 median | **{fmt(A['per_row_cv_median'], '%')}** (n={A['per_row_cv_n']}) | **{fmt(B['per_row_cv_median'], '%')}** (n={B['per_row_cv_n']}) | **{winner_label(A['per_row_cv_median'], B['per_row_cv_median'], True)}** |",
        f"| **적합 줄 비율** ★ | **{fmt(A['adequate_rate_pct'], '%')}** | **{fmt(B['adequate_rate_pct'], '%')}** | **{winner_label(A['adequate_rate_pct'], B['adequate_rate_pct'], False)}** |",
        "",
        "## 3. 결주 · 입모",
        "",
        f"| 지표 | {labels[0]} | {labels[1]} | 우수 |",
        f"|---|---:|---:|:---:|",
        f"| 전체 gap 개수 | {A['total_gaps']} | {B['total_gaps']} | - |",
        f"| 결주 gap 개수 | {A['miss_gaps']} | {B['miss_gaps']} | - |",
        f"| **결주율** ★ | **{fmt(A['miss_rate_pct'], '%')}** | **{fmt(B['miss_rate_pct'], '%')}** | **{winner_label(A['miss_rate_pct'], B['miss_rate_pct'], True)}** |",
        f"| 입모율 median | {fmt(A['emergence_stats']['median'], '%')} | {fmt(B['emergence_stats']['median'], '%')} | 100%에 가까운 쪽 |",
        "",
        "## 4. 종합 판정",
        "",
    ]

    # 종합 점수 (낮은 CV, 낮은 결주율, 높은 적합률)
    def score(x, cv, miss, adeq):
        s = 0
        if not np.isnan(cv):
            s += (100 - min(cv, 100)) * 0.4
        if not np.isnan(miss):
            s += (100 - min(miss, 100)) * 0.3
        if not np.isnan(adeq):
            s += min(adeq, 100) * 0.3
        return s
    sA = score(A, A["per_row_cv_median"], A["miss_rate_pct"],
                A["adequate_rate_pct"])
    sB = score(B, B["per_row_cv_median"], B["miss_rate_pct"],
                B["adequate_rate_pct"])
    lines += [
        f"| 파종기 | 종합점수 (0-100) |",
        f"|---|---:|",
        f"| {labels[0]} | **{sA:.1f}** |",
        f"| {labels[1]} | **{sB:.1f}** |",
        "",
        f"→ **종합 우수: {labels[0] if sA >= sB else labels[1]}**",
        "",
        "종합점수 = (100−CV)×0.4 + (100−결주율)×0.3 + 적합률×0.3",
        "",
        "## 5. 데이터 산출 흐름 (알고리즘)",
        "",
        "1. **필지 crop** — 정사영상에서 shapefile 기반 필지 TIF 생성",
        "2. **SAM 콩잎 검출** — Segment Anything Model + HSV/면적 필터로 발아 잎 point cloud 산출",
        "3. **Step 1 두둑 검출** — 필지 mask 안 gray weighted profile + 반복주기 60~130cm bandpass → 각도 스캔 → 두둑 중심 1D peak",
        "4. **Step 2 두둑별 2줄 검출** — 각 두둑 밴드 안에서 gray 반전(파종 자국 어두운 홈) + SAM 밀도 → sub-profile → dual-row 검출 + 조간 산출",
        "5. **Step 3 줄 내 파종 간격** — 각 파종 줄 밴드 안 SAM 잎을 ridge 방향 사영 → 인접 간격 median = 파종 간격, gap > median×1.5 = 결주",
        "6. **Step 4 시각화** — 필지 overview + 두둑별 그리드",
        "7. **Step 5 (본 리포트)** — 두 필지(파종기 zone) 지표 비교",
        "",
        f"결과 이미지: `result/report/seeder_comparison.png`",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("field_A")
    parser.add_argument("field_B")
    parser.add_argument("--labels", default=None,
                         help="쉼표로 구분된 라벨 두 개 (예: '스마트,일반')")
    args = parser.parse_args()

    if args.labels:
        labels = tuple(args.labels.split(",")[:2])
    else:
        labels = (args.field_A, args.field_B)

    t0 = time.time()
    print(f"[{labels[0]}] 로드")
    A = load_field(args.field_A)
    print(f"   두둑 {A['n_ridges']}, 유효 줄 {A['n_valid']}, "
          f"CV 대상 줄 {A['per_row_cv_n']}")
    print(f"   파종 median {A['med_row_median_stats']['median']:.1f}cm  "
          f"CV(줄별 median) {A['per_row_cv_median']:.1f}%  "
          f"결주율 {A['miss_rate_pct']:.1f}%  "
          f"입모율 {A['emergence_stats']['median']:.1f}%")

    print(f"[{labels[1]}] 로드")
    B = load_field(args.field_B)
    print(f"   두둑 {B['n_ridges']}, 유효 줄 {B['n_valid']}, "
          f"CV 대상 줄 {B['per_row_cv_n']}")
    print(f"   파종 median {B['med_row_median_stats']['median']:.1f}cm  "
          f"CV(줄별 median) {B['per_row_cv_median']:.1f}%  "
          f"결주율 {B['miss_rate_pct']:.1f}%  "
          f"입모율 {B['emergence_stats']['median']:.1f}%")

    out_png = REPORT_DIR / "seeder_comparison.png"
    out_csv = REPORT_DIR / "seeder_stats.csv"
    out_md = REPORT_DIR / "seeder_report.md"

    print(f"\n[리포트] 시각화")
    visualize(A, B, labels, out_png)
    print(f"   저장: {out_png}")

    print(f"[리포트] CSV")
    write_csv(A, B, labels, out_csv)
    print(f"   저장: {out_csv}")

    print(f"[리포트] Markdown")
    write_md(A, B, labels, out_md)
    print(f"   저장: {out_md}")

    print(f"\n완료 ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
