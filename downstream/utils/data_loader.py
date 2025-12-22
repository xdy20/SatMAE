"""
数据加载工具
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_grid_with_embeddings(geojson_path, embed_dim=1024):
    """
    加载带嵌入的格网数据

    参数:
        geojson_path: GeoJSON文件路径
        embed_dim: 嵌入维度

    返回:
        X: (N, embed_dim) 特征数组
        gdf: GeoDataFrame
    """
    import os
    file_size = os.path.getsize(geojson_path) / (1024**2)  # MB
    print(f"正在加载数据文件 ({file_size:.1f} MB)...")

    gdf = gpd.read_file(geojson_path)
    embed_cols = [f'embed_{i}' for i in range(embed_dim)]

    # 检查是否所有嵌入列都存在
    missing_cols = [col for col in embed_cols if col not in gdf.columns]
    if missing_cols:
        raise ValueError(f"缺少嵌入列: {missing_cols[:5]}... (共{len(missing_cols)}列)")

    X = gdf[embed_cols].values

    print(f"✓ 数据加载完成")
    print(f"  - 样本数量: {len(X)}")
    print(f"  - 特征维度: {X.shape[1]}")
    print(f"  - NaN数量: {np.isnan(X).sum()}")

    return X, gdf


def split_train_test(X, y, test_size=0.2, stratify=None, random_state=42):
    """
    统一的数据集划分

    参数:
        X: 特征数组
        y: 标签数组
        test_size: 测试集比例
        stratify: 分层抽样的依据（用于分类任务）
        random_state: 随机种子

    返回:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
