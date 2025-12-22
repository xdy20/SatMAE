"""
步骤1: 从Sentinel2影像提取224×224 tiles

从大幅Sentinel2 GeoTIFF影像中，根据格网边界裁剪固定尺寸的tiles
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from skimage.transform import resize
from tqdm import tqdm
import yaml
import argparse


def load_config(config_path='config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_beijing_tiles_metadata(sentinel2_dir):
    """
    预加载北京4个分块影像的bounds元数据

    返回:
        tiles_metadata: 列表，每个元素包含{file, bounds, transform}
    """
    beijing_tile_patterns = [
        "beijing_sentinel2_10m-0000000000-0000000000.tif",
        "beijing_sentinel2_10m-0000000000-0000011776.tif",
        "beijing_sentinel2_10m-0000011776-0000000000.tif",
        "beijing_sentinel2_10m-0000011776-0000011776.tif"
    ]

    tiles_metadata = []
    for tile_file in beijing_tile_patterns:
        tile_path = Path(sentinel2_dir) / tile_file
        if not tile_path.exists():
            print(f"⚠ 警告: 北京分块文件不存在: {tile_file}")
            continue

        with rasterio.open(tile_path) as src:
            tiles_metadata.append({
                'file': tile_file,
                'path': str(tile_path),
                'bounds': src.bounds,  # (minx, miny, maxx, maxy)
                'transform': src.transform,
                'crs': src.crs
            })

    print(f"✓ 加载{len(tiles_metadata)}个北京分块元数据")
    return tiles_metadata


def find_matching_beijing_tile(grid_center_x, grid_center_y, tiles_metadata):
    """
    根据格网中心坐标找到包含它的北京分块

    参数:
        grid_center_x, grid_center_y: 格网中心坐标
        tiles_metadata: 分块元数据列表

    返回:
        tile_metadata: 匹配的分块元数据，如果未找到返回None
    """
    from shapely.geometry import Point, box

    point = Point(grid_center_x, grid_center_y)

    for tile in tiles_metadata:
        tile_box = box(*tile['bounds'])
        if tile_box.contains(point):
            return tile

    return None


def extract_tile_from_image(src, grid_bounds, tile_size=224, pixel_size=10.0):
    """
    从影像中提取一个tile

    参数:
        src: rasterio打开的影像对象
        grid_bounds: 格网边界 (minx, miny, maxx, maxy)
        tile_size: 目标tile尺寸 (224×224)
        pixel_size: 像素分辨率 (10m)

    返回:
        tile_data: (3, 224, 224) numpy array, RGB顺序
    """
    # 计算格网中心
    center_x = (grid_bounds[0] + grid_bounds[2]) / 2
    center_y = (grid_bounds[1] + grid_bounds[3]) / 2

    # 计算224×224像素窗口的地理范围
    # tile_size=224, pixel_size=10m -> 2240m×2240m
    half_extent = (tile_size * pixel_size) / 2  # 1120m

    tile_bounds = (
        center_x - half_extent,
        center_y - half_extent,
        center_x + half_extent,
        center_y + half_extent
    )

    # 读取窗口
    try:
        # 检查tile_bounds是否在影像范围内
        img_bounds = src.bounds
        if (tile_bounds[0] < img_bounds[0] or tile_bounds[1] < img_bounds[1] or
            tile_bounds[2] > img_bounds[2] or tile_bounds[3] > img_bounds[3]):
            # 窗口超出影像范围，跳过
            return None

        window = from_bounds(*tile_bounds, transform=src.transform)

        # 检查窗口大小
        if window.width <= 0 or window.height <= 0:
            return None

        tile_data = src.read(window=window)  # (C, H, W)

        # 检查读取的数据是否为空
        if tile_data.size == 0 or tile_data.shape[1] == 0 or tile_data.shape[2] == 0:
            return None

        # Sentinel2影像通道: B2(Blue), B3(Green), B4(Red), B8(NIR)
        # 提取RGB: 通道1(B2-蓝), 通道2(B3-绿), 通道3(B4-红)
        if tile_data.shape[0] == 4:
            # 提取前3个通道 (B2, B3, B4) = BGR顺序
            tile_bgr = tile_data[:3, :, :]
            # 转换为RGB顺序: [B2, B3, B4] -> [B4, B3, B2] (Red, Green, Blue)
            tile_data = np.array([tile_bgr[2], tile_bgr[1], tile_bgr[0]])
        elif tile_data.shape[0] == 3:
            # 已经是3通道，假设是RGB顺序
            pass
        else:
            print(f"⚠ 警告: 影像通道数为{tile_data.shape[0]}, 预期3或4通道")
            return None

        # 调整到224×224 (如果需要)
        if tile_data.shape[1:] != (tile_size, tile_size):
            tile_resized = np.zeros((3, tile_size, tile_size), dtype=np.float32)
            for c in range(3):
                tile_resized[c] = resize(
                    tile_data[c],
                    (tile_size, tile_size),
                    preserve_range=True,
                    anti_aliasing=True
                )
            tile_data = tile_resized

        return tile_data.astype(np.float32)

    except Exception as e:
        # 静默跳过错误（通常是窗口超出范围）
        return None


def extract_tiles_for_city(city_name, sentinel2_dir, grid_dir, output_dir, tile_size=224):
    """
    为一个城市提取所有格网的tiles

    参数:
        city_name: 'shenzhen' or 'beijing'
        sentinel2_dir: Sentinel2影像目录
        grid_dir: 格网数据目录
        output_dir: 输出目录
        tile_size: tile尺寸
    """
    print(f"\n{'='*60}")
    print(f"处理城市: {city_name.upper()}")
    print(f"{'='*60}")

    # 创建输出目录
    city_output_dir = Path(output_dir) / city_name
    tiles_dir = city_output_dir / 'tiles'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # 加载格网数据 (GDP+人口 和 土地利用)
    grid_gdp_pop_path = Path(grid_dir) / 'grid_gdp_pop_builtup' / f'{city_name}_grid_gdp_pop.geojson'
    grid_lu_path = Path(grid_dir) / 'grid_lu' / f'{city_name}_grid_lu.geojson'

    # 合并所有格网ID (去重)
    all_grid_ids = set()
    grid_data_dict = {}

    # 首先获取影像的CRS用于坐标转换
    if city_name == 'beijing':
        sentinel2_path = Path(sentinel2_dir) / f'{city_name}_sentinel2_10m-0000000000-0000000000.tif'
    else:
        sentinel2_path = Path(sentinel2_dir) / f'{city_name}_sentinel2_10m.tif'

    target_crs = None
    if sentinel2_path.exists():
        with rasterio.open(sentinel2_path) as src:
            target_crs = src.crs
            print(f"✓ 影像坐标系统: {target_crs}")

    if grid_gdp_pop_path.exists():
        gdf_gdp_pop = gpd.read_file(grid_gdp_pop_path)
        print(f"  原始格网CRS: {gdf_gdp_pop.crs}")

        # 检查CRS是否需要修正
        # 深圳格网实际是EPSG:32650但被读取为EPSG:4326
        if str(gdf_gdp_pop.crs).find("4326") != -1:
            # 检查坐标值范围
            bounds = gdf_gdp_pop.total_bounds
            if bounds[0] > 360 or bounds[1] > 90:  # 明显不是经纬度
                print(f"  ⚠ 检测到CRS标注错误，坐标值为投影坐标但标注为地理坐标")
                # 深圳格网实际为EPSG:32650 (UTM Zone 50N)
                if city_name == 'shenzhen' and bounds[0] < 300000:
                    print(f"  ⚠ 深圳格网: 设置为EPSG:32650并转换为{target_crs}")
                    gdf_gdp_pop = gdf_gdp_pop.set_crs("EPSG:32650", allow_override=True)
                    gdf_gdp_pop = gdf_gdp_pop.to_crs(target_crs)
                else:
                    print(f"  ⚠ 直接设置CRS为: {target_crs}")
                    gdf_gdp_pop = gdf_gdp_pop.set_crs(target_crs, allow_override=True)
            else:
                # 真正的地理坐标，需要转换
                gdf_gdp_pop = gdf_gdp_pop.to_crs(target_crs)
        elif target_crs and gdf_gdp_pop.crs != target_crs:
            print(f"  ⚠ 转换格网坐标系统: {gdf_gdp_pop.crs} -> {target_crs}")
            gdf_gdp_pop = gdf_gdp_pop.to_crs(target_crs)

        for _, row in gdf_gdp_pop.iterrows():
            grid_id = row['id']
            all_grid_ids.add(grid_id)
            grid_data_dict[grid_id] = row.geometry
        print(f"✓ 加载GDP/人口格网: {len(gdf_gdp_pop)}个")

    if grid_lu_path.exists():
        gdf_lu = gpd.read_file(grid_lu_path)

        # 检查CRS是否需要修正
        if str(gdf_lu.crs).find("4326") != -1:
            bounds = gdf_lu.total_bounds
            if bounds[0] > 360 or bounds[1] > 90:
                print(f"  ⚠ 检测到CRS标注错误，直接设置CRS为: {target_crs}")
                # 深圳格网实际为EPSG:32650 (UTM Zone 50N)
                if city_name == 'shenzhen' and bounds[0] < 300000:
                    print(f"  ⚠ 深圳格网: 设置为EPSG:32650并转换为{target_crs}")
                    gdf_lu = gdf_lu.set_crs("EPSG:32650", allow_override=True)
                    gdf_lu = gdf_lu.to_crs(target_crs)
                else:
                    gdf_lu = gdf_lu.set_crs(target_crs, allow_override=True)
            else:
                gdf_lu = gdf_lu.to_crs(target_crs)
        elif target_crs and gdf_lu.crs != target_crs:
            print(f"  ⚠ 转换格网坐标系统: {gdf_lu.crs} -> {target_crs}")
            gdf_lu = gdf_lu.to_crs(target_crs)

        for _, row in gdf_lu.iterrows():
            grid_id = row['id']
            all_grid_ids.add(grid_id)
            if grid_id not in grid_data_dict:
                grid_data_dict[grid_id] = row.geometry
        print(f"✓ 加载土地利用格网: {len(gdf_lu)}个")

    print(f"✓ 总共需要处理: {len(all_grid_ids)}个唯一格网")

    # 打开Sentinel2影像
    if city_name == 'beijing':
        # 北京: 加载4个分块元数据
        beijing_tiles = load_beijing_tiles_metadata(sentinel2_dir)
        if len(beijing_tiles) == 0:
            print("✗ 错误: 未找到北京分块影像")
            return
    else:
        # 深圳: 单个文件
        sentinel2_path = Path(sentinel2_dir) / f'{city_name}_sentinel2_10m.tif'
        if not sentinel2_path.exists():
            print(f"✗ 错误: 影像文件不存在: {sentinel2_path}")
            return

    # 提取tiles
    metadata_list = []
    tile_id = 0
    success_count = 0
    fail_count = 0

    for grid_id in tqdm(sorted(all_grid_ids), desc=f"提取{city_name}的tiles"):
        if grid_id not in grid_data_dict:
            continue

        geometry = grid_data_dict[grid_id]
        grid_bounds = geometry.bounds  # (minx, miny, maxx, maxy)
        center_x = (grid_bounds[0] + grid_bounds[2]) / 2
        center_y = (grid_bounds[1] + grid_bounds[3]) / 2

        # 根据城市选择影像源
        if city_name == 'beijing':
            # 找到包含此格网的分块
            tile_metadata = find_matching_beijing_tile(center_x, center_y, beijing_tiles)
            if tile_metadata is None:
                fail_count += 1
                continue
            src_path = tile_metadata['path']
        else:
            src_path = sentinel2_path

        # 提取tile
        with rasterio.open(src_path) as src:
            tile_data = extract_tile_from_image(src, grid_bounds, tile_size=tile_size)

        if tile_data is None:
            fail_count += 1
            continue

        # 保存tile
        tile_filename = f"{grid_id}_tile.npy"
        tile_path = tiles_dir / tile_filename
        np.save(tile_path, tile_data)

        # 记录元数据
        metadata_list.append({
            'tile_id': tile_id,
            'grid_id': grid_id,
            'tile_path': str(tile_path.relative_to(city_output_dir.parent)),
            'center_x': center_x,
            'center_y': center_y,
            'bounds_minx': grid_bounds[0],
            'bounds_miny': grid_bounds[1],
            'bounds_maxx': grid_bounds[2],
            'bounds_maxy': grid_bounds[3]
        })

        tile_id += 1
        success_count += 1

    # 保存元数据
    metadata_df = pd.DataFrame(metadata_list)
    metadata_csv_path = city_output_dir / 'tile_metadata.csv'
    metadata_df.to_csv(metadata_csv_path, index=False)

    print(f"\n✓ {city_name}提取完成!")
    print(f"  - 成功: {success_count}个tiles")
    print(f"  - 失败: {fail_count}个")
    print(f"  - 元数据保存到: {metadata_csv_path}")
    print(f"  - Tiles保存到: {tiles_dir}/")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从Sentinel2影像提取tiles')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--city', type=str, default=None, help='指定城市 (shenzhen/beijing), 默认处理所有')
    args = parser.parse_args()

    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # 加载配置
    config = load_config(args.config)
    sentinel2_dir = config['paths']['sentinel2_dir']
    grid_dir = config['paths']['grid_dir']
    output_dir = config['paths']['output_dir']
    tile_size = config['data']['tile_size']
    cities = [args.city] if args.city else config['data']['cities']

    print("="*60)
    print("步骤1: 从Sentinel2影像提取tiles")
    print("="*60)
    print(f"Sentinel2目录: {sentinel2_dir}")
    print(f"格网目录: {grid_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Tile尺寸: {tile_size}×{tile_size}")
    print(f"处理城市: {cities}")

    # 处理每个城市
    for city in cities:
        try:
            extract_tiles_for_city(
                city_name=city,
                sentinel2_dir=sentinel2_dir,
                grid_dir=grid_dir,
                output_dir=output_dir,
                tile_size=tile_size
            )
        except Exception as e:
            print(f"✗ 处理{city}时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("✓ 所有城市处理完成!")
    print("="*60)


if __name__ == '__main__':
    main()
