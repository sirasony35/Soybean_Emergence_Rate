"""
입모율 재계산 스크립트 — 실측 기반 표준으로 갱신

================================================================
계산식 (문서화)
================================================================

■ 표준 재식밀도 공식 (100% 입모 = 개체/ha)

    표준밀도 = 10,000 m²/ha ÷ (두둑간격 × 주간간격) × 두둑당 줄 수

    - 두둑간격  : 각 필지에서 실측된 두둑 중심 간 거리 (m)
    - 주간간격  : 콩 개체 간 거리 (m). 파종기 표준 20cm = 0.20m
    - 두둑당 줄 수 : dual-row(2줄 파종)이므로 2

    예: 두둑간격 68cm, 주간 20cm, 2줄
        → 10,000 / (0.68 × 0.20) × 2 = 147,059 개체/ha

■ 입모율

    입모율(%) = (검출 잎 수 ÷ ROI 면적_ha) ÷ 표준밀도 × 100
              = 재식밀도(개체/ha) ÷ 표준밀도 × 100

================================================================
비교: 기존 방식과 새 방식
================================================================

■ 기존 (레거시): 단일-row 표준 76,923/ha
    - 65cm × 20cm 파종 가정, 1줄
    - 10,000 / (0.65 × 0.20) × 1 = 76,923
    - dual-row 반영 안 됨 → 100% 기준이 과소평가

■ 새 방식 (실측): 필지별 두둑간격 × dual-row 반영
    - 필지마다 검출된 실측 두둑간격 사용
    - dual-row(2줄) 반영
    - 100% 기준 상향 → 입모율은 낮아짐 (더 현실적)

================================================================
가정 / 한계
================================================================

  1. "1 검출 잎 = 1 개체" 로 근사 (dedup 5cm 반경으로 병합했지만
     완벽하지 않음. 실제 개체는 이보다 적을 수 있음).
  2. Dual-row 2줄 고정 가정. Smart/Normal 파종기의 실제 줄 수가
     다르면 별도 조정 필요.
  3. 주간 표준 20cm 고정 가정. 실측 주간과 다를 수 있음.
  4. Smart 필지의 검출 두둑간격 99.3cm은 다른 필지(65-70cm) 대비
     크게 이질적. 실제 파종 설계가 다르거나 두둑 검출이 일부
     실패했을 가능성 있음 → 재검토 필요.

사용:
  python -u scripts/recalc_emergence.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
SUMMARY_DIR = ROOT / "result" / "sam_test" / "summary"

# ─── 파종 조건 ───
INTRA_ROW_SPACING_CM = 20.0     # 주간 간격 (콩 개체 간)
ROWS_PER_RIDGE = 2              # dual-row (두둑당 2줄)

# ─── 레거시 표준 (비교용) ───
LEGACY_STD_PER_HA = 76923       # 65cm × 20cm 단일-row


def compute_field_standard(ridge_spacing_cm: float) -> float:
    """실측 두둑간격 기반 필지별 100% 입모 표준밀도 (개체/ha)"""
    if not np.isfinite(ridge_spacing_cm) or ridge_spacing_cm <= 0:
        return np.nan
    ridge_m = ridge_spacing_cm / 100.0
    intra_m = INTRA_ROW_SPACING_CM / 100.0
    return 10000.0 / (ridge_m * intra_m) * ROWS_PER_RIDGE


def recalc(results: list) -> list:
    """summary_stats.json 각 필지에 새 표준·새 입모율 추가"""
    for r in results:
        rs = r.get("ridge_spacing_cm")
        std_new = compute_field_standard(rs)
        density = r["density_per_ha"]
        rate_new = (density / std_new * 100) if np.isfinite(std_new) else np.nan
        r["standard_new_per_ha"] = float(std_new) if np.isfinite(std_new) else None
        r["emergence_rate_new_pct"] = float(rate_new) if np.isfinite(rate_new) else None
        r["emergence_rate_legacy_pct"] = r["emergence_rate_pct"]  # 기존값 보존
    return results


def draw_field_card_v2(r: dict, out_png: Path, rgb: np.ndarray,
                        step: int, final_centroids: np.ndarray):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax_txt, ax_img) = plt.subplots(1, 2, figsize=(16, 8),
                                          gridspec_kw={"width_ratios": [1, 1.4]})
    ax_txt.axis("off")

    field = r["field"]
    ax_txt.text(0.02, 0.97, field, fontsize=22, fontweight="bold",
                transform=ax_txt.transAxes, verticalalignment="top")
    ax_txt.text(0.02, 0.90, "(30m × 30m ROI, 실측 표준)",
                fontsize=10, color="gray", transform=ax_txt.transAxes,
                verticalalignment="top")

    std_new = r["standard_new_per_ha"]
    lines = [
        ("발아 콩잎",    f"{r['n_final']:,} 개"),
        ("재식밀도",     f"{r['density_per_ha']:,.0f} 개체/ha"),
        ("주간 간격",    f"{r['intra_row_spacing_median_cm']:.1f} cm  (median)"
                         if r['intra_row_spacing_median_cm'] is not None
                         and np.isfinite(r['intra_row_spacing_median_cm']) else "N/A"),
        ("두둑 간격",    f"{r['ridge_spacing_cm']:.1f} cm  ({r['n_ridges']}개)"
                         if np.isfinite(r.get('ridge_spacing_cm', np.nan)) else "N/A"),
        ("표준(실측)",   f"{std_new:,.0f} 개체/ha"
                         if std_new else "N/A"),
        ("표준(레거시)", f"{LEGACY_STD_PER_HA:,} 개체/ha"),
    ]
    y = 0.80
    for k, v in lines:
        ax_txt.text(0.02, y, f"{k}", fontsize=12, color="#444",
                    transform=ax_txt.transAxes, verticalalignment="top")
        ax_txt.text(0.32, y, f"{v}", fontsize=13, fontweight="bold",
                    transform=ax_txt.transAxes, verticalalignment="top")
        y -= 0.07

    # 두 입모율 나란히
    rate_new = r["emergence_rate_new_pct"]
    rate_leg = r["emergence_rate_legacy_pct"]
    color_new = "#22aa22" if rate_new and rate_new >= 40 else \
                "#c66600" if rate_new and rate_new >= 25 else "#cc3333"
    ax_txt.text(0.02, 0.30, "입모율 (실측 표준)", fontsize=11, color="#444",
                transform=ax_txt.transAxes, verticalalignment="top")
    ax_txt.text(0.02, 0.24,
                f"{rate_new:.1f} %" if rate_new is not None else "N/A",
                fontsize=36, fontweight="bold", color=color_new,
                transform=ax_txt.transAxes, verticalalignment="top")
    ax_txt.text(0.55, 0.28, "레거시 (참고)", fontsize=10, color="gray",
                transform=ax_txt.transAxes, verticalalignment="top")
    ax_txt.text(0.55, 0.22, f"{rate_leg:.1f} %",
                fontsize=20, color="gray",
                transform=ax_txt.transAxes, verticalalignment="top")

    ax_img.imshow(rgb)
    ax_img.scatter(final_centroids[:, 1] / step, final_centroids[:, 0] / step,
                   s=6, c="lime", alpha=0.85, edgecolor="none")
    ax_img.axis("off")
    ax_img.set_title(f"발아 콩잎 검출 위치 ({r['n_final']:,}개)", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def draw_comparison_v2(results: list, out_png: Path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))

    names = [r["field"] for r in results]
    rate_new = [r["emergence_rate_new_pct"] or 0 for r in results]
    rate_leg = [r["emergence_rate_legacy_pct"] for r in results]
    densities = [r["density_per_ha"] for r in results]
    stds_new = [r["standard_new_per_ha"] or 0 for r in results]

    # (1) 두 입모율 비교
    x = np.arange(len(names))
    w = 0.4
    ax1.bar(x - w/2, rate_new, w, color="#22aa22", label="실측 표준 기반")
    ax1.bar(x + w/2, rate_leg, w, color="#aaaaaa", label="레거시(76,923) 참고")
    ax1.set_ylabel("입모율 (%)", fontsize=12)
    ax1.set_title("입모율: 실측 표준 vs 레거시 (참고)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right")
    for xi, rn, rl in zip(x, rate_new, rate_leg):
        ax1.text(xi - w/2, rn + 1.5, f"{rn:.1f}", ha="center",
                 fontsize=10, fontweight="bold")
        ax1.text(xi + w/2, rl + 1.5, f"{rl:.1f}", ha="center",
                 fontsize=9, color="gray")

    # (2) 재식밀도 vs 필지별 표준
    ax2.bar(x - w/2, densities, w, color="#4477bb", label="검출 재식밀도")
    ax2.bar(x + w/2, stds_new, w, color="#e0c060", label="실측 표준밀도 (100%)")
    ax2.set_ylabel("개체/ha", fontsize=12)
    ax2.set_title("재식밀도 vs 필지별 표준밀도", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.axhline(LEGACY_STD_PER_HA, color="black", ls="--", lw=1, alpha=0.5,
                label=f"레거시 표준 {LEGACY_STD_PER_HA:,}")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    # (3) 검출 잎 개수
    ax3.bar(names, [r["n_final"] for r in results], color="#4477bb")
    ax3.set_ylabel("발아 콩잎 개수", fontsize=12)
    ax3.set_title("발아 콩잎 검출 개수 (30m × 30m ROI)",
                  fontsize=14, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3)
    for i, r in enumerate(results):
        ax3.text(i, r["n_final"] + 80, f"{r['n_final']:,}",
                 ha="center", fontsize=11, fontweight="bold")
    plt.setp(ax3.get_xticklabels(), rotation=15)

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def make_markdown_v2(results: list) -> str:
    lines = []
    lines.append("# 새만금 논콩 입모율 분석 — 6필지 요약 (실측 표준)\n")
    lines.append("**ROI**: 각 필지 중심 30m × 30m  |  **GSD**: 5.25mm/px  "
                 "|  **모델**: SAM v2  |  **파종 조건**: dual-row 2줄, 주간 20cm\n")
    lines.append("## 표준 재식밀도 계산식\n")
    lines.append("```")
    lines.append("표준 = 10,000 (m²/ha) ÷ (두둑간격_m × 주간간격_m) × 두둑당 줄 수")
    lines.append("     = 10,000 ÷ (실측_두둑간격_m × 0.20) × 2")
    lines.append("입모율(%) = 검출 재식밀도 ÷ 표준 × 100")
    lines.append("```\n")
    lines.append("## 필지별 결과\n")
    lines.append("| 필지 | 발아잎 | 재식밀도(/ha) | 두둑간격(cm) | 주간(cm) | "
                 "**표준(실측)** | **입모율(실측)** | 입모율(레거시) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        rs = r.get("ridge_spacing_cm", np.nan)
        intra = r.get("intra_row_spacing_median_cm", np.nan)
        std_new = r["standard_new_per_ha"]
        rate_new = r["emergence_rate_new_pct"]
        rate_leg = r["emergence_rate_legacy_pct"]
        rs_s = f"{rs:.1f}" if np.isfinite(rs) else "-"
        intra_s = f"{intra:.1f}" if intra is not None and np.isfinite(intra) else "-"
        std_s = f"{std_new:,.0f}" if std_new else "-"
        rn_s = f"**{rate_new:.1f}%**" if rate_new is not None else "-"
        rl_s = f"{rate_leg:.1f}%"
        lines.append(
            f"| {r['field']} | {r['n_final']:,} | "
            f"{r['density_per_ha']:,.0f} | {rs_s} | {intra_s} | "
            f"{std_s} | {rn_s} | {rl_s} |")

    smart = next((r for r in results if "_Smart" in r["field"]), None)
    normal = next((r for r in results if "_normal" in r["field"]), None)
    if smart and normal:
        lines.append("\n## 파종기 비교 (GJSM-1-1)\n")
        lines.append("| 항목 | Smart | Normal | 차이 |")
        lines.append("|---|---:|---:|---:|")
        lines.append(f"| 발아 콩잎 | {smart['n_final']:,} | "
                     f"{normal['n_final']:,} | "
                     f"{normal['n_final'] - smart['n_final']:+,} |")
        lines.append(f"| 재식밀도 (/ha) | {smart['density_per_ha']:,.0f} | "
                     f"{normal['density_per_ha']:,.0f} | "
                     f"{normal['density_per_ha'] - smart['density_per_ha']:+,.0f} |")
        lines.append(f"| 두둑 간격 (cm) | {smart['ridge_spacing_cm']:.1f} | "
                     f"{normal['ridge_spacing_cm']:.1f} | "
                     f"{normal['ridge_spacing_cm'] - smart['ridge_spacing_cm']:+.1f} |")
        lines.append(f"| 실측 표준 (/ha) | {smart['standard_new_per_ha']:,.0f} | "
                     f"{normal['standard_new_per_ha']:,.0f} | "
                     f"{normal['standard_new_per_ha'] - smart['standard_new_per_ha']:+,.0f} |")
        rn_s = smart["emergence_rate_new_pct"]
        rn_n = normal["emergence_rate_new_pct"]
        lines.append(f"| **입모율(실측)** | **{rn_s:.1f}%** | "
                     f"**{rn_n:.1f}%** | **{rn_n - rn_s:+.1f}%p** |")

    lines.append("\n## 계산 근거 및 가정\n")
    lines.append(f"- 주간 간격: **{INTRA_ROW_SPACING_CM:.0f}cm** (파종기 스펙, 고정)")
    lines.append(f"- 두둑당 줄 수: **{ROWS_PER_RIDGE}줄** (dual-row 파종)")
    lines.append(f"- 두둑 간격: **필지별 실측값** (SAM 검출된 콩잎 중심점의 "
                 f"perpendicular density peaks)")
    lines.append(f"- 레거시 표준: **{LEGACY_STD_PER_HA:,}/ha** "
                 f"(65cm × 20cm 단일-row 가정, 참고용)\n")
    lines.append("## 한계\n")
    lines.append("1. **1 잎 = 1 개체 근사**: SAM은 잎 마스크를 검출하는 것이지 "
                 "\"개체\"를 직접 세지 않음. 새싹 하나가 여러 잎(자엽 2 + 초생엽)을 "
                 "만들어 여러 마스크로 나올 수 있음. dedup(5cm 반경)으로 상당수 병합했지만 "
                 "완벽하지 않음.")
    lines.append("2. **Dual-row 고정**: 실제 파종기가 dual-row인지 실측 확인 필요.")
    lines.append("3. **주간 20cm 고정**: 필지별 실측 주간이 있음에도 표준값 사용. "
                 "표준값을 실측으로 대체하면 결과가 또 달라질 수 있음.")
    lines.append("4. **Smart 두둑간격 99.3cm 이질적**: 다른 필지(65-70cm) 대비 큰 차이. "
                 "실제 파종 설계가 다른지, 두둑 검출이 일부 실패했는지 재검토 권장.")
    lines.append("5. **ROI 대표성**: 각 필지 중심 30m × 30m만 처리. 필지 전체 평균 아님.")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("입모율 재계산 — 실측 표준 기반")
    print("=" * 60)

    json_path = SUMMARY_DIR / "summary_stats.json"
    if not json_path.exists():
        print(f"❌ {json_path} 없음. 먼저 generate_summary.py 실행.")
        return
    results = json.loads(json_path.read_text(encoding="utf-8"))

    results = recalc(results)

    print(f"\n주간 (고정): {INTRA_ROW_SPACING_CM}cm")
    print(f"두둑당 줄 수 (고정): {ROWS_PER_RIDGE}")
    print(f"레거시 표준 (참고): {LEGACY_STD_PER_HA:,}/ha\n")
    print(f"{'필지':<20} {'두둑간격':>10} {'실측표준':>12} "
          f"{'입모율(실측)':>12} {'입모율(레거시)':>14}")
    print("-" * 72)
    for r in results:
        rs = r.get("ridge_spacing_cm", np.nan)
        std_new = r["standard_new_per_ha"]
        rate_new = r["emergence_rate_new_pct"]
        rate_leg = r["emergence_rate_legacy_pct"]
        rs_s = f"{rs:.1f}cm" if np.isfinite(rs) else "-"
        std_s = f"{std_new:,.0f}/ha" if std_new else "-"
        rn_s = f"{rate_new:.1f}%" if rate_new is not None else "-"
        rl_s = f"{rate_leg:.1f}%"
        print(f"{r['field']:<20} {rs_s:>10} {std_s:>12} {rn_s:>12} {rl_s:>14}")

    # 갱신된 JSON 저장
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\n✅ JSON 갱신: {json_path}")

    # 갱신된 마크다운
    md_path = SUMMARY_DIR / "summary_table.md"
    md_path.write_text(make_markdown_v2(results), encoding="utf-8")
    print(f"✅ 마크다운 갱신: {md_path}")

    # 갱신된 비교 차트
    draw_comparison_v2(results, SUMMARY_DIR / "comparison.png")
    print(f"✅ 비교 차트 갱신: {SUMMARY_DIR / 'comparison.png'}")

    # 개별 카드 재생성 — npz에서 RGB·centroid 다시 로드
    print("\n필지별 카드 재생성...")
    from generate_summary import process_field  # 재사용
    for r in results:
        try:
            src = process_field(r["field"])
        except Exception as e:
            print(f"  ❌ {r['field']}: {e}")
            continue
        card_path = SUMMARY_DIR / f"{r['field']}_summary.png"
        draw_field_card_v2(r, card_path, src["_rgb"], src["_step"],
                           src["_final_centroids"])
        print(f"  ✅ {card_path.name}")

    print(f"\n✅ 완료 — 결과 폴더: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
