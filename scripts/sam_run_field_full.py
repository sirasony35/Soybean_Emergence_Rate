"""
SAM v2 전체 필지 처리 — 필지 TIF 전체(유효 픽셀) 대상.
성능 최적화: alpha=0 타일 스킵 + 타일 레벨 centroid/bbox 계산.

사용:
  python -u scripts/sam_run_field_full.py <field_name>
  python -u scripts/sam_run_field_full.py GJSM-1-1_Smart

출력:
  result/sam_test/{field}_sam_FULL_v4.npz
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELDS_DIR = ROOT / "result" / "fields"
OUT_DIR = ROOT / "result" / "sam_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"

# 콩잎 기본 필터 (v1 원본)
MIN_LEAF_AREA_CM2 = 1.0
MAX_LEAF_AREA_CM2 = 200.0
HUE_MIN, HUE_MAX = 20, 100
SAT_MIN = 20
VAL_MIN = 30

# SAM v2 강화 파라미터
PTS_PER_SIDE = 64
PRED_IOU_THRESH = 0.70
STABILITY_SCORE_THRESH = 0.75
MIN_MASK_REGION_AREA = 2

# 타일링
TILE_PX = 1024
STRIDE = TILE_PX - 128
ALPHA_MIN_FRAC = 0.10   # 타일 유효 픽셀이 10% 미만이면 스킵

VERSION_TAG = "FULL_v4"
SAM_CKPT = MODELS_DIR / "sam_vit_l_0b3195.pth"
SAM_TYPE = "vit_l"


def make_mask_gen(sam):
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=PTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        crop_n_layers=0,
        min_mask_region_area=MIN_MASK_REGION_AREA,
    )


def process_tile(tile_rgb, tile_hsv, mask_gen, ty, tx, gsd_m):
    px_area_cm2 = (gsd_m * 100) ** 2
    tile_masks = mask_gen.generate(tile_rgb)
    n_total = len(tile_masks)
    leaves = []
    for m in tile_masks:
        seg = m["segmentation"]
        area_px = int(m.get("area", int(seg.sum())))
        area_cm2 = area_px * px_area_cm2
        if area_cm2 < MIN_LEAF_AREA_CM2 or area_cm2 > MAX_LEAF_AREA_CM2:
            continue
        h_mean = float(tile_hsv[seg, 0].mean())
        s_mean = float(tile_hsv[seg, 1].mean())
        v_mean = float(tile_hsv[seg, 2].mean())
        if not (HUE_MIN <= h_mean <= HUE_MAX and s_mean >= SAT_MIN and v_mean >= VAL_MIN):
            continue
        ys, xs = np.where(seg)
        leaves.append([
            float(ys.mean()) + ty, float(xs.mean()) + tx,
            area_cm2,
            h_mean, s_mean, v_mean,
            int(ys.min()) + ty, int(xs.min()) + tx,
            int(ys.max()) + ty, int(xs.max()) + tx,
        ])
    return leaves, n_total


def main():
    if len(sys.argv) < 2:
        print("사용: python sam_run_field_full.py <field_name>")
        sys.exit(1)
    field_name = sys.argv[1]
    tif_path = FIELDS_DIR / f"{field_name}.tif"
    if not tif_path.exists():
        print(f"❌ TIF 없음: {tif_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"SAM v2 [전체 필지] — {field_name}  → {VERSION_TAG}")
    print(f"  pts={PTS_PER_SIDE}, iou={PRED_IOU_THRESH}, "
          f"stab={STABILITY_SCORE_THRESH}, min_area={MIN_MASK_REGION_AREA}px")
    print("=" * 60)

    # 전체 TIF 정보 + alpha 마스크
    t0 = time.time()
    with rasterio.open(tif_path) as src:
        gsd = abs(src.transform.a)
        H, W = src.height, src.width
        H_m, W_m = H * gsd, W * gsd
        area_total_ha = (H_m * W_m) / 10000
        # alpha 읽어서 유효 픽셀 계산
        alpha = src.read(4) if src.count >= 4 else np.ones((H, W), dtype=np.uint8) * 255
    valid_frac = (alpha != 0).sum() / alpha.size
    valid_area_ha = area_total_ha * valid_frac
    print(f"[TIF] {time.time()-t0:.1f}s  {H}×{W}px  {H_m:.1f}×{W_m:.1f}m  "
          f"전체 {area_total_ha:.2f}ha  유효 {valid_frac*100:.1f}%  "
          f"유효면적 {valid_area_ha:.2f}ha")

    # 유효 타일 인덱스 미리 계산 (alpha 마스크에서 각 타일 유효 픽셀 비율 체크)
    tile_indices = []
    for ty in range(0, H, STRIDE):
        for tx in range(0, W, STRIDE):
            y1 = min(ty + TILE_PX, H); x1 = min(tx + TILE_PX, W)
            if y1 - ty < 100 or x1 - tx < 100:
                continue
            tile_alpha = alpha[ty:y1, tx:x1]
            valid_pct = (tile_alpha != 0).sum() / tile_alpha.size
            if valid_pct >= ALPHA_MIN_FRAC:
                tile_indices.append((ty, tx, y1, x1, valid_pct))
    n_tiles = len(tile_indices)
    n_all_tiles = ((H - 1) // STRIDE + 1) * ((W - 1) // STRIDE + 1)
    print(f"[tiles] 전체 {n_all_tiles} → 유효 {n_tiles} "
          f"(스킵 {n_all_tiles - n_tiles})")

    del alpha  # 메모리 절약

    # SAM 로드
    t0 = time.time()
    sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT))
    sam.to("cuda")
    mask_gen = make_mask_gen(sam)
    print(f"[SAM] 모델 로드 {time.time()-t0:.1f}s")

    # 타일 순회 (필요한 것만 개별 window 읽기)
    all_leaves = []
    total_masks = 0
    t_start = time.time()
    for i, (ty, tx, y1, x1, vpct) in enumerate(tile_indices):
        with rasterio.open(tif_path) as src:
            win = Window(col_off=tx, row_off=ty, width=x1-tx, height=y1-ty)
            rgb = src.read(indexes=[1, 2, 3], window=win)
            alpha_tile = src.read(4, window=win) if src.count >= 4 else None
        tile_rgb = np.transpose(rgb, (1, 2, 0))
        if alpha_tile is not None:
            tile_rgb[alpha_tile == 0] = 0

        tile_hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
        try:
            leaves, n_total = process_tile(tile_rgb, tile_hsv, mask_gen, ty, tx, gsd)
        except RuntimeError as e:
            msg = str(e)
            if "device" in msg or "cpu" in msg:
                print(f"  ❌ NMS bug tile {i+1} (y={ty},x={tx}): {msg[:80]}")
                return
            raise
        all_leaves.extend(leaves)
        total_masks += n_total

        elapsed = time.time() - t_start
        eta_sec = elapsed / (i + 1) * (n_tiles - i - 1)
        eta_min = eta_sec / 60
        if (i + 1) % 5 == 0 or i == 0 or i == n_tiles - 1:
            print(f"  tile {i+1:>4}/{n_tiles} (y={ty},x={tx}, valid={vpct*100:.0f}%) "
                  f"→ 마스크 {n_total}, 잎 {len(leaves)}, 누적 {len(all_leaves)}  "
                  f"[경과 {elapsed/60:.1f}분, ETA {eta_min:.0f}분]")

    print(f"[run] SAM 완료 {(time.time()-t_start)/60:.1f}분  "
          f"총 마스크 {total_masks}, 총 잎 {len(all_leaves)}")

    # 결과 저장
    t0 = time.time()
    leaf_arr = np.array(all_leaves, dtype=np.float32) if all_leaves \
                else np.zeros((0, 10), dtype=np.float32)
    # 표시용 다운샘플 RGB (전체 필지 축소)
    with rasterio.open(tif_path) as src:
        # 표시용은 (약 2000px 이내)
        step = max(1, max(H, W) // 2000)
        rgb_disp = src.read(indexes=[1, 2, 3])[:, ::step, ::step]
        rgb_disp = np.transpose(rgb_disp, (1, 2, 0))
        if src.count >= 4:
            alpha_disp = src.read(4)[::step, ::step]
            rgb_disp[alpha_disp == 0] = 0

    npz_path = OUT_DIR / f"{field_name}_sam_{VERSION_TAG}.npz"
    np.savez_compressed(
        npz_path,
        rgb_disp=rgb_disp,
        rgb_step=np.array([step]),
        leaf_arr=leaf_arr,
        gsd_ds=np.array([gsd]),
        n_all_masks=np.array([total_masks]),
        valid_area_ha=np.array([valid_area_ha]),
        field_area_ha=np.array([area_total_ha]),
    )
    print(f"[npz] {time.time()-t0:.1f}s  {npz_path.name} "
          f"({npz_path.stat().st_size/1e6:.1f}MB)")

    n = len(all_leaves)
    density = n / valid_area_ha if valid_area_ha > 0 else 0
    print(f"\n=== {field_name} 결과 ===")
    print(f"  유효 면적: {valid_area_ha:.2f} ha")
    print(f"  검출 잎: {n:,} 개")
    print(f"  재식밀도(잎기준): {density:,.0f} /ha")
    print(f"  ※ 두둑 필터 A+B 적용 전 baseline")


if __name__ == "__main__":
    main()
