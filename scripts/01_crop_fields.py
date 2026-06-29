"""
[01] 13.8GB 정사영상을 5개 SHP 폴리곤으로 마스크 클립 → 필지별 TIF 저장

각 SHP는 EPSG:4326 → 정사영상 CRS(EPSG:32652)로 재투영 후 rasterio.mask로 클립.
폴리곤 외부 픽셀은 nodata(0) 처리, alpha 채널 유지.

산출: result/fields/{필지명}.tif
"""

from __future__ import annotations
from pathlib import Path
import time
import rasterio
from rasterio.mask import mask as rio_mask

from emergence_lib import load_polygon_zip

ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
ORTHO = ROOT / "data" / "간척지 초고해상도 정사영상-orthomosaic.tiff"
SHP_DIR = ROOT / "shapefile"
OUT_DIR = ROOT / "result" / "fields"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_one_field(src: rasterio.io.DatasetReader, shp_zip: Path, out_path: Path):
    field_name = shp_zip.stem.replace("_Boundary", "")
    print(f"\n== {field_name} ==")
    t0 = time.time()

    gdf = load_polygon_zip(shp_zip, src.crs)
    poly = gdf.geometry.iloc[0]
    print(f"  폴리곤 면적: {poly.area/10000:.3f} ha")
    print(f"  bounds (m): {tuple(round(v,1) for v in poly.bounds)}")

    out_img, out_transform = rio_mask(
        src,
        [poly.__geo_interface__],
        crop=True,
        all_touched=False,
        filled=True,
        nodata=0,
    )
    print(f"  크롭 후 shape: {out_img.shape}, 메모리: {out_img.nbytes/1e9:.2f} GB")

    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "nodata": 0,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "YES",
    })

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_img)
        # 오버뷰 생성 (탐색용)
        try:
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")
        except Exception as e:
            print(f"  (overview 생성 실패: {e})")

    dt = time.time() - t0
    print(f"  → {out_path.name} 저장 ({out_path.stat().st_size/1e6:.1f} MB, {dt:.1f}s)")


def main():
    print(f"정사영상: {ORTHO}")
    print(f"출력 폴더: {OUT_DIR}\n")

    shp_zips = sorted(SHP_DIR.glob("*.zip"))
    print(f"필지 SHP {len(shp_zips)}개 발견")

    with rasterio.open(ORTHO) as src:
        print(f"정사영상 CRS: {src.crs},  GSD: {abs(src.transform.a)*1000:.3f} mm/px")
        print(f"전체 크기: {src.width} × {src.height}")
        t_total = time.time()
        for shp_zip in shp_zips:
            field_name = shp_zip.stem.replace("_Boundary", "")
            out_path = OUT_DIR / f"{field_name}.tif"
            if out_path.exists():
                print(f"\n== {field_name} ==  (이미 존재, 스킵: {out_path.name})")
                continue
            crop_one_field(src, shp_zip, out_path)
        print(f"\n전체 완료: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
