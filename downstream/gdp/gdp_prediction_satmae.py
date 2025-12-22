"""
GDP预测任务 - 使用SatMAE嵌入

网格级GDP回归预测
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 添加utils路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_grid_with_embeddings
from utils.model_utils import get_regression_models
from utils.evaluation import evaluate_regression


def load_gdp_data(geojson_path, embed_dim=1024):
    """
    加载GDP预测数据

    参数:
        geojson_path: GeoJSON文件路径
        embed_dim: 嵌入维度

    返回:
        X: (N, embed_dim) 特征数组
        y: (N,) GDP标签
        gdf: GeoDataFrame
    """
    X, gdf = load_grid_with_embeddings(geojson_path, embed_dim=embed_dim)

    # 提取GDP标签
    y = gdf['gdp'].values

    # 过滤有效样本
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & (y > 0)
    X = X[valid_mask]
    y = y[valid_mask]
    gdf = gdf[valid_mask]

    print(f"  - 有效样本: {len(X)}")
    print(f"  - GDP范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  - GDP均值: {y.mean():.2f}")
    print(f"  - GDP中位数: {np.median(y):.2f}")

    return X, y, gdf


def train_gdp_model(X_train, y_train, X_test, y_test, city_name, use_log=True):
    """
    训练GDP预测模型

    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        city_name: 城市名称
        use_log: 是否在对数空间训练

    返回:
        results_df: 结果DataFrame
    """
    print(f"\n训练GDP预测模型 ({city_name})...")

    # 对数变换 (处理偏斜分布)
    if use_log:
        print("  使用对数变换")
        y_train_transformed = np.log1p(y_train)
        y_test_transformed = np.log1p(y_test)
    else:
        y_train_transformed = y_train
        y_test_transformed = y_test

    # 获取模型
    models = get_regression_models(regularization='strong')

    results = []

    for name, model in models.items():
        print(f"\n  训练模型: {name}")

        # 训练
        model.fit(X_train, y_train_transformed)

        # 预测
        y_train_pred_transformed = model.predict(X_train)
        y_test_pred_transformed = model.predict(X_test)

        # 逆变换
        if use_log:
            y_train_pred = np.expm1(y_train_pred_transformed)
            y_test_pred = np.expm1(y_test_pred_transformed)
        else:
            y_train_pred = y_train_pred_transformed
            y_test_pred = y_test_pred_transformed

        # 确保非负
        y_train_pred = np.maximum(y_train_pred, 0)
        y_test_pred = np.maximum(y_test_pred, 0)

        # 评估
        train_metrics = evaluate_regression(y_train, y_train_pred)
        test_metrics = evaluate_regression(y_test, y_test_pred)

        result = {
            'City': city_name,
            'Model': name,
            'Train_R2': train_metrics['R2'],
            'Train_RMSE': train_metrics['RMSE'],
            'Train_MAE': train_metrics['MAE'],
            'Test_R2': test_metrics['R2'],
            'Test_RMSE': test_metrics['RMSE'],
            'Test_MAE': test_metrics['MAE'],
            'Test_MAPE': test_metrics['MAPE']
        }

        results.append(result)

        print(f"    训练集 R²: {train_metrics['R2']:.4f}")
        print(f"    测试集 R²: {test_metrics['R2']:.4f}")
        print(f"    测试集 RMSE: {test_metrics['RMSE']:.2f}")
        print(f"    测试集 MAE: {test_metrics['MAE']:.2f}")

    return pd.DataFrame(results)


def main():
    """主函数"""
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("="*60)
    print("GDP预测任务 - SatMAE")
    print("="*60)

    # 路径配置
    outputs_dir = Path('../outputs')
    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    cities = ['shenzhen', 'beijing']
    all_results = []

    # 处理每个城市
    for city in cities:
        print(f"\n{'='*60}")
        print(f"城市: {city.upper()}")
        print(f"{'='*60}")

        try:
            # 1. 加载数据
            geojson_path = outputs_dir / f'{city}_grid_satmae_gdp_pop.geojson'
            if not geojson_path.exists():
                print(f"⚠ 跳过{city}: 数据文件不存在")
                continue

            X, y, gdf = load_gdp_data(geojson_path, embed_dim=1024)

            # 2. 划分数据集 (80/20)
            print(f"\n划分数据集...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"  - 训练集: {len(X_train)}个样本")
            print(f"  - 测试集: {len(X_test)}个样本")

            # 3. 标准化特征
            print(f"\n标准化特征...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print(f"  ✓ 特征已标准化")

            # 4. 训练模型
            results = train_gdp_model(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                city_name=city,
                use_log=True
            )

            all_results.append(results)

        except Exception as e:
            print(f"✗ 处理{city}时出错: {e}")
            import traceback
            traceback.print_exc()

    # 汇总结果
    if len(all_results) > 0:
        final_results = pd.concat(all_results, ignore_index=True)

        # 保存结果
        results_csv = metrics_dir / 'gdp_prediction_satmae_results.csv'
        final_results.to_csv(results_csv, index=False)
        print(f"\n✓ 结果已保存到: {results_csv}")

        # 打印汇总
        print("\n" + "="*60)
        print("最终结果汇总")
        print("="*60)
        print(final_results.to_string(index=False))

        # 找出最佳模型
        print("\n" + "="*60)
        print("最佳模型 (按Test R²排序)")
        print("="*60)
        best_results = final_results.sort_values('Test_R2', ascending=False)
        print(best_results[['City', 'Model', 'Test_R2', 'Test_RMSE', 'Test_MAE']].to_string(index=False))

    print("\n" + "="*60)
    print("✓ GDP预测任务完成!")
    print("="*60)


if __name__ == '__main__':
    main()
