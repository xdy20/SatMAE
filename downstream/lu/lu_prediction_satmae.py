"""
土地利用预测任务 - 使用SatMAE嵌入

网格级土地利用多分类预测
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 添加utils路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_grid_with_embeddings
from utils.model_utils import get_classification_models
from utils.evaluation import evaluate_classification


def load_lu_data(geojson_path, embed_dim=1024):
    """
    加载土地利用分类数据

    参数:
        geojson_path: GeoJSON文件路径
        embed_dim: 嵌入维度

    返回:
        X: (N, embed_dim) 特征数组
        y: (N,) 土地利用类型标签 (编码后)
        le: LabelEncoder
        gdf: GeoDataFrame
    """
    X, gdf = load_grid_with_embeddings(geojson_path, embed_dim=embed_dim)

    # 提取土地利用标签
    y = gdf['lu'].values

    # 过滤有效样本
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    gdf = gdf[valid_mask]

    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"  - 有效样本: {len(X)}")
    print(f"  - 土地利用类型: {le.classes_}")
    print(f"  - 类型数量: {len(le.classes_)}")

    # 打印类别分布
    unique, counts = np.unique(y_encoded, return_counts=True)
    print(f"  - 类别分布:")
    for cls, count in zip(unique, counts):
        lu_type = le.inverse_transform([cls])[0]
        print(f"    类型{lu_type}: {count}个 ({count/len(y_encoded)*100:.1f}%)")

    return X, y_encoded, le, gdf


def train_lu_model(X_train, y_train, X_test, y_test, city_name, le):
    """
    训练土地利用分类模型

    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        city_name: 城市名称
        le: LabelEncoder

    返回:
        results_df: 结果DataFrame
    """
    print(f"\n训练土地利用分类模型 ({city_name})...")

    # 获取模型
    models = get_classification_models()

    results = []

    for name, model in models.items():
        print(f"\n  训练模型: {name}")

        # 训练
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 评估
        train_metrics, _ = evaluate_classification(y_train, y_train_pred)
        test_metrics, test_report = evaluate_classification(y_test, y_test_pred)

        result = {
            'City': city_name,
            'Model': name,
            'Train_Acc': train_metrics['Accuracy'],
            'Test_Acc': test_metrics['Accuracy'],
            'Test_F1_Macro': test_metrics['F1_Macro'],
            'Test_F1_Weighted': test_metrics['F1_Weighted']
        }

        results.append(result)

        print(f"    训练集 Accuracy: {train_metrics['Accuracy']:.4f}")
        print(f"    测试集 Accuracy: {test_metrics['Accuracy']:.4f}")
        print(f"    测试集 F1 (Macro): {test_metrics['F1_Macro']:.4f}")
        print(f"    测试集 F1 (Weighted): {test_metrics['F1_Weighted']:.4f}")

        # 打印详细分类报告
        print(f"\n    详细分类报告 ({name}):")
        print(test_report)

    return pd.DataFrame(results)


def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='土地利用预测任务 - SatMAE')
    parser.add_argument('--city', type=str, default='all',
                       help='城市名称: shenzhen, beijing, 或 all (默认)')
    args = parser.parse_args()

    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("="*60)
    print("土地利用预测任务 - SatMAE")
    print("="*60)

    # 路径配置
    outputs_dir = Path('../outputs')
    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    # 确定要处理的城市
    if args.city == 'all':
        cities = ['shenzhen', 'beijing']
    else:
        cities = [args.city.lower()]

    all_results = []

    # 处理每个城市
    for city in cities:
        print(f"\n{'='*60}")
        print(f"城市: {city.upper()}")
        print(f"{'='*60}")

        try:
            # 1. 加载数据
            geojson_path = outputs_dir / f'{city}_grid_satmae_lu.geojson'
            if not geojson_path.exists():
                print(f"⚠ 跳过{city}: 数据文件不存在")
                continue

            X, y, le, gdf = load_lu_data(geojson_path, embed_dim=1024)

            # 2. 划分数据集 (80/20, 分层抽样)
            print(f"\n划分数据集...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
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
            results = train_lu_model(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                city_name=city,
                le=le
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
        results_csv = metrics_dir / 'lu_prediction_satmae_results.csv'
        final_results.to_csv(results_csv, index=False)
        print(f"\n✓ 结果已保存到: {results_csv}")

        # 打印汇总
        print("\n" + "="*60)
        print("最终结果汇总")
        print("="*60)
        print(final_results.to_string(index=False))

        # 找出最佳模型
        print("\n" + "="*60)
        print("最佳模型 (按Test Accuracy排序)")
        print("="*60)
        best_results = final_results.sort_values('Test_Acc', ascending=False)
        print(best_results[['City', 'Model', 'Test_Acc', 'Test_F1_Macro', 'Test_F1_Weighted']].to_string(index=False))

    print("\n" + "="*60)
    print("✓ 土地利用预测任务完成!")
    print("="*60)


if __name__ == '__main__':
    main()
