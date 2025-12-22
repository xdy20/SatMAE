"""
步骤3: 映射嵌入到格网

将tile级嵌入映射到grid级，并合并标签数据
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import yaml
import argparse


def load_config(config_path='config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def map_embeddings_to_gdp_pop_grid(city_name, embeddings_csv, grid_geojson, output_path):
    """
    将嵌入映射到建成区格网 (GDP+人口任务)

    参数:
        city_name: 城市名称
        embeddings_csv: 嵌入CSV路径
        grid_geojson: 格网GeoJSON路径
        output_path: 输出GeoJSON路径
    """
    print(f"\n映射GDP/人口格网: {city_name}")

    # 1. 加载嵌入
    embed_df = pd.read_csv(embeddings_csv)
    embed_cols = [col for col in embed_df.columns if col.startswith('embed_')]
    print(f"  ✓ 嵌入维度: {len(embed_cols)}")
    print(f"  ✓ tile数量: {len(embed_df)}")

    # 2. 加载格网
    grid_gdf = gpd.read_file(grid_geojson)
    print(f"  ✓ 格网数量: {len(grid_gdf)}")

    # 3. 合并 (使用grid_id匹配)
    result_gdf = grid_gdf.merge(
        embed_df[['grid_id'] + embed_cols],
        left_on='id',
        right_on='grid_id',
        how='inner'  # 只保留有嵌入的格网
    )
    print(f"  ✓ 合并后: {len(result_gdf)}个格网")

    # 4. 过滤有效样本 (有GDP和人口数据)
    before_filter = len(result_gdf)
    valid_mask = (
        result_gdf['gdp'].notna() &
        result_gdf['pop'].notna() &
        (result_gdf['gdp'] > 0) &
        (result_gdf['pop'] > 0)
    )
    result_gdf = result_gdf[valid_mask]
    print(f"  ✓ 过滤后: {len(result_gdf)}个有效格网 (过滤掉{before_filter - len(result_gdf)}个)")

    # 5. 检查NaN
    nan_count = result_gdf[embed_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"  ⚠ 警告: 发现{nan_count}个NaN嵌入值")

    # 6. 保存
    result_gdf.to_file(output_path, driver='GeoJSON')
    print(f"  ✓ 保存到: {output_path}")

    # 7. 统计
    stats = {
        'city': city_name,
        'task': 'gdp_pop',
        'grid_count': len(result_gdf),
        'embed_dim': len(embed_cols),
        'gdp_range': [float(result_gdf['gdp'].min()), float(result_gdf['gdp'].max())],
        'pop_range': [float(result_gdf['pop'].min()), float(result_gdf['pop'].max())],
        'gdp_mean': float(result_gdf['gdp'].mean()),
        'pop_mean': float(result_gdf['pop'].mean())
    }

    return result_gdf, stats


def map_embeddings_to_lu_grid(city_name, embeddings_csv, grid_geojson, output_path):
    """
    将嵌入映射到土地利用格网

    参数:
        city_name: 城市名称
        embeddings_csv: 嵌入CSV路径
        grid_geojson: 格网GeoJSON路径
        output_path: 输出GeoJSON路径
    """
    print(f"\n映射土地利用格网: {city_name}")

    # 1. 加载嵌入
    embed_df = pd.read_csv(embeddings_csv)
    embed_cols = [col for col in embed_df.columns if col.startswith('embed_')]
    print(f"  ✓ 嵌入维度: {len(embed_cols)}")
    print(f"  ✓ tile数量: {len(embed_df)}")

    # 2. 加载格网
    grid_gdf = gpd.read_file(grid_geojson)
    print(f"  ✓ 格网数量: {len(grid_gdf)}")

    # 3. 合并
    result_gdf = grid_gdf.merge(
        embed_df[['grid_id'] + embed_cols],
        left_on='id',
        right_on='grid_id',
        how='inner'
    )
    print(f"  ✓ 合并后: {len(result_gdf)}个格网")

    # 4. 过滤有效样本 (有土地利用标签)
    before_filter = len(result_gdf)
    valid_mask = result_gdf['lu'].notna()
    result_gdf = result_gdf[valid_mask]
    print(f"  ✓ 过滤后: {len(result_gdf)}个有效格网 (过滤掉{before_filter - len(result_gdf)}个)")

    # 5. 检查NaN
    nan_count = result_gdf[embed_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"  ⚠ 警告: 发现{nan_count}个NaN嵌入值")

    # 6. 保存
    result_gdf.to_file(output_path, driver='GeoJSON')
    print(f"  ✓ 保存到: {output_path}")

    # 7. 统计土地利用类型分布
    lu_counts = result_gdf['lu'].value_counts().to_dict()
    stats = {
        'city': city_name,
        'task': 'lu',
        'grid_count': len(result_gdf),
        'embed_dim': len(embed_cols),
        'lu_types': len(lu_counts),
        'lu_distribution': {str(k): int(v) for k, v in lu_counts.items()}
    }

    print(f"  ✓ 土地利用类型: {len(lu_counts)}种")
    for lu_type, count in sorted(lu_counts.items()):
        print(f"    - 类型{lu_type}: {count}个")

    return result_gdf, stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='映射嵌入到格网')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--city', type=str, default=None, help='指定城市 (shenzhen/beijing), 默认处理所有')
    args = parser.parse_args()

    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # 加载配置
    config = load_config(args.config)
    data_processed_dir = Path(config['paths']['output_dir'])
    grid_dir = Path(config['paths']['grid_dir'])
    downstream_dir = Path(config['paths']['downstream_dir'])
    cities = [args.city] if args.city else config['data']['cities']

    # 创建输出目录
    outputs_dir = downstream_dir / 'outputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("步骤3: 映射嵌入到格网")
    print("="*60)
    print(f"数据目录: {data_processed_dir}")
    print(f"格网目录: {grid_dir}")
    print(f"输出目录: {outputs_dir}")
    print(f"处理城市: {cities}")

    all_stats = []

    # 处理每个城市
    for city in cities:
        print(f"\n{'='*60}")
        print(f"处理城市: {city.upper()}")
        print(f"{'='*60}")

        try:
            # 加载嵌入CSV
            embeddings_csv = data_processed_dir / city / 'satmae_embeddings.csv'
            if not embeddings_csv.exists():
                print(f"⚠ 跳过{city}: 嵌入文件不存在")
                continue

            # GDP/人口任务
            grid_gdp_pop_path = grid_dir / 'grid_gdp_pop_builtup' / f'{city}_grid_gdp_pop.geojson'
            if grid_gdp_pop_path.exists():
                output_gdp_pop = outputs_dir / f'{city}_grid_satmae_gdp_pop.geojson'
                _, stats_gdp_pop = map_embeddings_to_gdp_pop_grid(
                    city, embeddings_csv, grid_gdp_pop_path, output_gdp_pop
                )
                all_stats.append(stats_gdp_pop)
            else:
                print(f"  ⚠ GDP/人口格网不存在: {grid_gdp_pop_path}")

            # 土地利用任务
            grid_lu_path = grid_dir / 'grid_lu' / f'{city}_grid_lu.geojson'
            if grid_lu_path.exists():
                output_lu = outputs_dir / f'{city}_grid_satmae_lu.geojson'
                _, stats_lu = map_embeddings_to_lu_grid(
                    city, embeddings_csv, grid_lu_path, output_lu
                )
                all_stats.append(stats_lu)
            else:
                print(f"  ⚠ 土地利用格网不存在: {grid_lu_path}")

        except Exception as e:
            print(f"✗ 处理{city}时出错: {e}")
            import traceback
            traceback.print_exc()

    # 保存汇总统计
    summary_path = outputs_dir / 'mapping_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 统计摘要已保存到: {summary_path}")

    print("\n" + "="*60)
    print("✓ 所有城市处理完成!")
    print("="*60)

    # 打印汇总
    print("\n汇总统计:")
    for stats in all_stats:
        print(f"\n{stats['city']} - {stats['task']}:")
        print(f"  - 格网数量: {stats['grid_count']}")
        print(f"  - 嵌入维度: {stats['embed_dim']}")
        if stats['task'] == 'gdp_pop':
            print(f"  - GDP范围: [{stats['gdp_range'][0]:.0f}, {stats['gdp_range'][1]:.0f}]")
            print(f"  - 人口范围: [{stats['pop_range'][0]:.0f}, {stats['pop_range'][1]:.0f}]")
        else:
            print(f"  - 土地利用类型: {stats['lu_types']}种")


if __name__ == '__main__':
    main()
