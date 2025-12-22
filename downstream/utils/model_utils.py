"""
模型工具
"""
import torch
import sys
sys.path.insert(0, '/home/y/Code/GeoAI/ae/baseline/SatMAE')

from models_vit import vit_large_patch16
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet


def load_satmae(checkpoint_path, device='cuda'):
    """
    加载SatMAE模型 (统一接口)

    参数:
        checkpoint_path: fmow_pretrain.pth路径
        device: 'cuda' or 'cpu'

    返回:
        model: 加载好的SatMAE模型
    """
    # 检查CUDA可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA不可用，切换到CPU")
        device = 'cpu'

    # 创建ViT-Large模型 (num_classes=0表示不要分类头)
    model = vit_large_patch16(num_classes=0, global_pool=False)

    # 加载预训练权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

    # 设置为评估模式
    model = model.to(device)
    model.eval()

    print(f"✓ SatMAE模型加载完成 (设备: {device})")
    print(f"  - 嵌入维度: 1024")
    print(f"  - 输入尺寸: 224×224 RGB")

    return model


def get_regression_models(regularization='strong'):
    """
    获取预配置的回归模型

    参数:
        regularization: 'strong' / 'medium' / 'weak'

    返回:
        models: 字典，模型名称 -> 模型对象
    """
    if regularization == 'strong':
        return {
            'Ridge': Ridge(alpha=100.0),  # 增加正则化强度
            'Lasso': Lasso(alpha=10.0, max_iter=5000),
            'ElasticNet': ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=5000),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,  # 减少树数量
                max_depth=6,       # 降低树深度
                min_samples_split=20,  # 增加分裂最小样本数
                min_samples_leaf=10,   # 增加叶子最小样本数
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,      # 减少树数量
                max_depth=3,           # 降低树深度
                learning_rate=0.03,    # 降低学习率
                min_samples_split=20,  # 增加分裂最小样本数
                min_samples_leaf=10,   # 增加叶子最小样本数
                subsample=0.7,         # 降低子采样比例
                validation_fraction=0.1,  # 使用验证集
                n_iter_no_change=10,   # 早停
                random_state=42
            )
        }
    elif regularization == 'medium':
        return {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
    else:  # weak
        return {
            'Ridge': Ridge(alpha=0.1),
            'RandomForest': RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        }


def get_classification_models():
    """
    获取预配置的分类模型

    返回:
        models: 字典，模型名称 -> 模型对象
    """
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=10,
            random_state=42
        )
    }
