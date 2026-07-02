"""
Step 4 - 두둑별 결과 이미지.

산출:
  1. field_overview.png     - 필지 전체 (두둑 라벨 + 파종 줄 + 통계)
  2. ridge_grid.png         - 두둑별 crop 그리드 (6 col × N row)
  3. ridges/ridge_{id:03d}.png - 두둑 개별 상세 (원하는 경우 --detail 인자)

각 두둑 crop 오버레이:
  - 원본 RGB
  - 파종 줄 2개 (빨간 라인)
  - SAM 잎 (초록 점)
  - 결주 gap 강조 (붉은 span)
  - 캡션: id, 유형, 조간 xcm, 파종 median ycm, 입모율 z%, 결주 n

사용:
  python -u scripts/pipeline/step4_visualize.py GJSM-1-1_Smart [--detail]
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.common import (
    load_field_npz, leaf_centroids, get_field_dir, setup_korean_font,
)


# ---- 파라미터 ----
CROP_MARGIN_CM = 30            # 두둑 밴드 + 여백 (perp/ridge 양쪽)
CROP_LENGTH_MAX_CM = 1500      # crop 세로 최대 (긴 두둑 잘라내기)
GRID_COLS = 6                  # 그리드 열 수
GRID_MAX_PAGE = 60             # 페이지당 두둑 최대

TYPE_COLOR = {
    "narrow": "#F58C39",       # 주황
    "wide":   "#8E44AD",       # 보라
    "single": "#7F8C8D",       # 회색
    "other":  "#16A085",       # 청록
}


def load_all(field: str):
    d = load_field_npz(field)
    sam_pts = leaf_centroids(d["leaves"])
    field_dir = get_field_dir(field)

    R = np.load(field_dir / "ridges.npz", allow_pickle=True)
    S = np.load(field_dir / "rows.npz", allow_pickle=True)
    P = np.load(field_dir / "spacing.npz", allow_pickle=True)

    ridge_arr = R["ridge_arr"]; ridge_types = R["ridge_types"]
    row_arr = S["row_arr"]; row_types = S["type_arr"]
    stats_arr = P["stats_arr"]; stats_types = P["stats_types"]
    positions_obj = P["positions_obj"]

    ridges = []
    for i, row in enumerate(ridge_arr):
        ridges.append(dict(
            ridge_id=int(row[0]), n_rows=int(row[1]),
            center_perp_px=float(row[2]),
            width_cm=float(row[4]),
            band_half_cm=float(row[5]),
            ridge_min_px=float(row[6]), ridge_max_px=float(row[7]),
            length_cm=float(row[8]), n_leaves=int(row[9]),
            ridge_type=str(ridge_types[i]),
        ))

    # ridge_id → row info
    row_lut = {}
    for i, row in enumerate(row_arr):
        rid = int(row[0])
        n_peaks = int(row[1])
        peaks = [x for x in [row[2], row[3]] if not np.isnan(x)][:n_peaks]
        row_lut[rid] = dict(
            peaks_perp_px=peaks,
            gap_cm=float(row[4]) if not np.isnan(row[4]) else np.nan,
            type_final=str(row_types[i]),
        )

    # ridge_id → per-row spacing stats + positions
    spacing_lut = {}  # ridge_id → [row_idx → stats]
    for j, row in enumerate(stats_arr):
        rid = int(row[0]); ridx = int(row[1])
        spacing_lut.setdefault(rid, {})[ridx] = dict(
            n=int(row[2]), median_cm=float(row[3]), mean_cm=float(row[4]),
            std_cm=float(row[5]), miss_gaps=int(row[6]),
            expected_plants=int(row[7]), emergence_pct=float(row[8]),
            length_cm=float(row[9]), row_perp_px=float(row[10]),
            positions_cm=positions_obj[j],
        )

    return dict(
        d=d, sam_pts=sam_pts, field_dir=field_dir,
        origin=R["origin"], perp=R["perp"], ridge_dir=R["ridge_dir"],
        px_to_cm=float(R["px_to_cm"][0]),
        ridges=ridges, row_lut=row_lut, spacing_lut=spacing_lut,
    )


# ---- crop 계산 ----
def compute_ridge_crop_bbox(ridge, origin, perp, ridge_dir, rgb_step, px_to_cm,
                              rgb_shape):
    """
    두둑의 4 corner (perp ± band_half + margin, ridge min/max ± margin) 를
    disp 좌표로 변환 → axis-aligned bbox 반환 (y0, x0, y1, x1).
    """
    band_half_px = ridge["band_half_cm"] / px_to_cm
    margin_px = CROP_MARGIN_CM / px_to_cm

    r_min = ridge["ridge_min_px"] - margin_px
    r_max = ridge["ridge_max_px"] + margin_px
    p_lo = -band_half_px - margin_px
    p_hi = +band_half_px + margin_px

    # 두둑 4 corner (원본 disp 좌표)
    corners = []
    for r_off, p_off in [(r_min, p_lo), (r_max, p_lo),
                           (r_max, p_hi), (r_min, p_hi)]:
        pt = origin + r_off * ridge_dir + (ridge["center_perp_px"] + p_off) * perp
        corners.append(pt / rgb_step)
    corners = np.array(corners)
    y0 = int(max(0, corners[:, 0].min()))
    y1 = int(min(rgb_shape[0], corners[:, 0].max() + 1))
    x0 = int(max(0, corners[:, 1].min()))
    x1 = int(min(rgb_shape[1], corners[:, 1].max() + 1))
    return y0, x0, y1, x1


def draw_ridge_on_ax(ax, ridge, row_info, spacing_by_row,
                      origin, perp, ridge_dir, rgb_step, px_to_cm,
                      sam_pts, y0, x0, y1, x1):
    """crop 영역에 오버레이 그리기."""
    band_half_px = ridge["band_half_cm"] / px_to_cm
    rtype = row_info["type_final"]
    color = TYPE_COLOR.get(rtype, "gray")

    # 두둑 밴드 경계 (perp ± band_half, ridge min/max)
    corners = []
    for r_off, p_off in [
        (ridge["ridge_min_px"], -band_half_px),
        (ridge["ridge_max_px"], -band_half_px),
        (ridge["ridge_max_px"], +band_half_px),
        (ridge["ridge_min_px"], +band_half_px),
    ]:
        pt = origin + r_off * ridge_dir + (ridge["center_perp_px"] + p_off) * perp
        corners.append(pt / rgb_step - np.array([y0, x0]))
    corners = np.array(corners)
    poly = mpatches.Polygon(corners[:, ::-1], closed=True,
                             edgecolor=color, facecolor="none", linewidth=1.4)
    ax.add_patch(poly)

    # 파종 줄 라인 2개
    for j, p_perp_px in enumerate(row_info["peaks_perp_px"]):
        p1 = origin + ridge["ridge_min_px"] * ridge_dir + p_perp_px * perp
        p2 = origin + ridge["ridge_max_px"] * ridge_dir + p_perp_px * perp
        p1d = p1 / rgb_step - np.array([y0, x0])
        p2d = p2 / rgb_step - np.array([y0, x0])
        ax.plot([p1d[1], p2d[1]], [p1d[0], p2d[0]],
                color="red", lw=1.0, alpha=0.85)

        # 결주 gap 표시 (해당 줄이 spacing 통계 있을 때)
        s = spacing_by_row.get(j)
        if s and s["n"] >= 2 and s["miss_gaps"] > 0:
            # 결주 gap 위치를 line 위에 span으로
            positions_cm = s["positions_cm"]
            miss_thresh = s["median_cm"] * 1.5
            for k in range(len(positions_cm) - 1):
                g = positions_cm[k + 1] - positions_cm[k]
                if g > miss_thresh:
                    # 각 위치 → ridge_dir 방향의 원본 px
                    pos1_px = positions_cm[k] / px_to_cm
                    pos2_px = positions_cm[k + 1] / px_to_cm
                    q1 = origin + pos1_px * ridge_dir + p_perp_px * perp
                    q2 = origin + pos2_px * ridge_dir + p_perp_px * perp
                    q1d = q1 / rgb_step - np.array([y0, x0])
                    q2d = q2 / rgb_step - np.array([y0, x0])
                    ax.plot([q1d[1], q2d[1]], [q1d[0], q2d[0]],
                             color="yellow", lw=2.2, alpha=0.65)

    # SAM 잎 (crop 영역 내부만)
    sam_disp = sam_pts[:, :2] / rgb_step - np.array([y0, x0])
    in_crop = (sam_disp[:, 0] >= 0) & (sam_disp[:, 0] < y1 - y0) \
              & (sam_disp[:, 1] >= 0) & (sam_disp[:, 1] < x1 - x0)
    ax.scatter(sam_disp[in_crop, 1], sam_disp[in_crop, 0],
                s=6, c="lime", alpha=0.7, edgecolor="darkgreen", linewidths=0.15)


def ridge_stats_line(ridge, row_info, spacing_by_row):
    rtype = row_info["type_final"]
    gap = row_info["gap_cm"]
    parts = [f"#{ridge['ridge_id']} [{rtype}]"]
    if not np.isnan(gap):
        parts.append(f"조간 {gap:.0f}cm")
    if 0 in spacing_by_row and spacing_by_row[0]["n"] >= 2:
        s0 = spacing_by_row[0]
        parts.append(f"R0 {s0['median_cm']:.0f}cm/{s0['emergence_pct']:.0f}%")
    if 1 in spacing_by_row and spacing_by_row[1]["n"] >= 2:
        s1 = spacing_by_row[1]
        parts.append(f"R1 {s1['median_cm']:.0f}cm/{s1['emergence_pct']:.0f}%")
    return " ".join(parts)


# ---- 필지 overview ----
def draw_field_overview(res: dict, out_png: Path):
    setup_korean_font()
    d = res["d"]; sam_pts = res["sam_pts"]
    origin = res["origin"]; perp = res["perp"]; ridge_dir = res["ridge_dir"]
    rgb_step = d["rgb_step"]

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(d["rgb_disp"])
    ax.scatter(sam_pts[:, 1] / rgb_step, sam_pts[:, 0] / rgb_step,
                s=0.5, c="lime", alpha=0.25, edgecolor="none")
    for r in res["ridges"]:
        row_info = res["row_lut"].get(r["ridge_id"], dict(peaks_perp_px=[],
                                                           type_final="single"))
        color = TYPE_COLOR.get(row_info["type_final"], "gray")
        for p_perp_px in row_info["peaks_perp_px"]:
            p1 = origin + r["ridge_min_px"] * ridge_dir + p_perp_px * perp
            p2 = origin + r["ridge_max_px"] * ridge_dir + p_perp_px * perp
            p1d = p1 / rgb_step; p2d = p2 / rgb_step
            ax.plot([p1d[1], p2d[1]], [p1d[0], p2d[0]],
                    color=color, lw=0.4, alpha=0.85)
        # 5개마다 라벨
        if r["ridge_id"] % 5 == 0:
            cy = (r["ridge_min_px"] + r["ridge_max_px"]) / 2
            pt = origin + cy * ridge_dir + r["center_perp_px"] * perp
            pt_d = pt / rgb_step
            ax.text(pt_d[1], pt_d[0], str(r["ridge_id"]),
                     color="white", fontsize=7, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.12",
                                facecolor="black", alpha=0.55))

    # 통계 요약 텍스트 (오른쪽 위)
    types_dict = {}
    gaps_by_type = {}
    spacing_med_by_type = {}
    emg_by_type = {}
    for rid, ri in res["row_lut"].items():
        t = ri["type_final"]
        types_dict[t] = types_dict.get(t, 0) + 1
        if not np.isnan(ri["gap_cm"]):
            gaps_by_type.setdefault(t, []).append(ri["gap_cm"])
        # spacing per row → 두둑 종합
        sl = res["spacing_lut"].get(rid, {})
        for ridx, s in sl.items():
            if s["n"] >= 5:
                spacing_med_by_type.setdefault(t, []).append(s["median_cm"])
                emg_by_type.setdefault(t, []).append(s["emergence_pct"])

    lines = [f"필지: {res['field']}",
             f"두둑 총 {len(res['ridges'])}"]
    for t in ["narrow", "wide", "single", "other"]:
        cnt = types_dict.get(t, 0)
        if cnt == 0:
            continue
        row = f"  {t}: {cnt}"
        if t in gaps_by_type:
            row += f"  조간 median {np.median(gaps_by_type[t]):.1f}cm"
        if t in spacing_med_by_type:
            row += (f"  파종 median {np.median(spacing_med_by_type[t]):.1f}cm"
                    f"  입모율 {np.median(emg_by_type[t]):.0f}%")
        lines.append(row)

    bbox = dict(boxstyle="round,pad=0.4",
                 facecolor="white", edgecolor="gray", alpha=0.9)
    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes,
             fontsize=11, va="top", ha="left", bbox=bbox)
    ax.set_title(f"필지 Overview - {res['field']}  "
                  f"(빨강=파종 줄, 초록=SAM 잎, 색=조간 유형)",
                  fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


# ---- 두둑 그리드 ----
def draw_ridge_grid(res: dict, out_png: Path,
                     page: int, n_pages: int, ridge_slice):
    setup_korean_font()
    d = res["d"]; sam_pts = res["sam_pts"]
    origin = res["origin"]; perp = res["perp"]; ridge_dir = res["ridge_dir"]
    rgb_step = d["rgb_step"]; px_to_cm = res["px_to_cm"]
    rgb = d["rgb_disp"]

    ridges = ridge_slice
    n = len(ridges)
    rows = int(np.ceil(n / GRID_COLS))
    fig, axes = plt.subplots(rows, GRID_COLS,
                              figsize=(GRID_COLS * 3.5, rows * 3.5))
    axes = np.atleast_1d(axes).ravel()
    for k, r in enumerate(ridges):
        ax = axes[k]
        row_info = res["row_lut"].get(r["ridge_id"], dict(peaks_perp_px=[],
                                                           type_final="single",
                                                           gap_cm=np.nan))
        spacing_by_row = res["spacing_lut"].get(r["ridge_id"], {})
        y0, x0, y1, x1 = compute_ridge_crop_bbox(
            r, origin, perp, ridge_dir, rgb_step, px_to_cm, rgb.shape)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            ax.text(0.5, 0.5, f"#{r['ridge_id']} 빈 crop",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        ax.imshow(crop)
        draw_ridge_on_ax(ax, r, row_info, spacing_by_row,
                          origin, perp, ridge_dir, rgb_step, px_to_cm,
                          sam_pts, y0, x0, y1, x1)
        title = ridge_stats_line(r, row_info, spacing_by_row)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.suptitle(f"두둑 그리드 - {res['field']}  "
                  f"페이지 {page}/{n_pages}  "
                  f"(#{ridges[0]['ridge_id']} ~ #{ridges[-1]['ridge_id']})",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


# ---- 두둑 상세 개별 ----
def draw_ridge_detail(res: dict, ridge, out_png: Path):
    setup_korean_font()
    d = res["d"]; sam_pts = res["sam_pts"]
    origin = res["origin"]; perp = res["perp"]; ridge_dir = res["ridge_dir"]
    rgb_step = d["rgb_step"]; px_to_cm = res["px_to_cm"]
    rgb = d["rgb_disp"]

    row_info = res["row_lut"].get(ridge["ridge_id"], dict(peaks_perp_px=[],
                                                           type_final="single",
                                                           gap_cm=np.nan))
    spacing_by_row = res["spacing_lut"].get(ridge["ridge_id"], {})
    y0, x0, y1, x1 = compute_ridge_crop_bbox(
        ridge, origin, perp, ridge_dir, rgb_step, px_to_cm, rgb.shape)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                               gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    crop = rgb[y0:y1, x0:x1]
    ax.imshow(crop)
    draw_ridge_on_ax(ax, ridge, row_info, spacing_by_row,
                      origin, perp, ridge_dir, rgb_step, px_to_cm,
                      sam_pts, y0, x0, y1, x1)
    title = ridge_stats_line(ridge, row_info, spacing_by_row)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    # 아래: 각 줄의 잎 도트 위치 (길이 방향)
    ax2 = axes[1]
    for j, s in spacing_by_row.items():
        if s["n"] < 2:
            continue
        pos_cm = s["positions_cm"] - s["positions_cm"].min()
        yy = 0.6 if j == 0 else 0.3
        ax2.scatter(pos_cm, [yy] * len(pos_cm), s=18, c="lime",
                    edgecolor="darkgreen", zorder=3, label=f"row {j}")
        # 결주 span
        med = s["median_cm"]
        if len(s["positions_cm"]) >= 2:
            gaps = np.diff(s["positions_cm"])
            for k, g in enumerate(gaps):
                if g > med * 1.5:
                    x1_ = pos_cm[k]; x2_ = pos_cm[k + 1]
                    ax2.axvspan(x1_, x2_, ymin=yy - 0.08, ymax=yy + 0.08,
                                 color="red", alpha=0.3)
        ax2.text(-30, yy, f"R{j}\n{s['n']}잎\n{s['median_cm']:.0f}cm",
                  fontsize=9, ha="right", va="center")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("줄 방향 위치 (cm)")
    ax2.set_title(f"잎 위치 (초록) + 결주 gap (빨강 span)", fontsize=11)
    ax2.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()


# ---- 메인 ----
def main():
    if len(sys.argv) < 2:
        print("사용: python step4_visualize.py <field_name> [--detail]")
        sys.exit(1)
    field = sys.argv[1]
    do_detail = "--detail" in sys.argv
    t0 = time.time()

    print(f"[{field}] 로드")
    res = load_all(field)
    res["field"] = field
    print(f"   두둑 {len(res['ridges'])}, SAM 잎 {len(res['sam_pts']):,}")

    print(f"[{field}] 필지 overview")
    draw_field_overview(res, res["field_dir"] / "field_overview.png")
    print(f"   저장: field_overview.png")

    print(f"[{field}] 두둑 그리드")
    ridges = res["ridges"]
    n = len(ridges)
    n_pages = int(np.ceil(n / GRID_MAX_PAGE))
    for p in range(n_pages):
        s = p * GRID_MAX_PAGE
        e = min(s + GRID_MAX_PAGE, n)
        out = res["field_dir"] / (
            f"ridge_grid.png" if n_pages == 1 else f"ridge_grid_p{p+1}.png")
        draw_ridge_grid(res, out, p + 1, n_pages, ridges[s:e])
        print(f"   저장: {out.name}  (두둑 {s}~{e-1})")

    if do_detail:
        detail_dir = res["field_dir"] / "ridges"
        detail_dir.mkdir(exist_ok=True)
        print(f"[{field}] 두둑 상세 개별 저장 → {detail_dir.name}/")
        for r in ridges:
            out = detail_dir / f"ridge_{r['ridge_id']:03d}.png"
            draw_ridge_detail(res, r, out)
        print(f"   {len(ridges)}개 저장")

    print(f"[{field}] 완료 ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
