# SatMAE下游任务

使用SatMAE预训练权重完成GDP、人口和土地利用三个下游任务。

## 概述

本项目使用SatMAE (ViT-Large, RGB, 77.78% on fMoW)预训练权重提取1024维特征嵌入，然后使用传统机器学习模型进行下游任务训练。

**核心策略**:
- ✅ 不微调模型 - 仅提取特征
- ✅ 使用CLS token嵌入 (1024-dim)
- ✅ 传统ML模型 (Ridge, RandomForest, GradientBoosting)
- ✅ 强正则化防止过拟合

## 三个下游任务

1. **GDP预测** - 网格级回归 (建成区格网)
2. **人口预测** - 网格级回归 (建成区格网)
3. **土地利用预测** - 网格级多分类 (土地利用格网)

## 目录结构

```
downstream/
├── config.yaml                         # 配置文件
├── README.md                           # 本文档
│
├── utils/                              # 共享工具模块
│   ├── __init__.py
│   ├── data_loader.py                  # 数据加载
│   ├── model_utils.py                  # 模型工具
│   └── evaluation.py                   # 评估函数
│
├── extract_tiles_from_sentinel2.py    # 步骤1: 从Sentinel2提取tiles
├── extract_satmae_embeddings.py       # 步骤2: 提取SatMAE嵌入
├── map_embeddings_to_grids.py         # 步骤3: 映射嵌入到格网
│
├── gdp/                                # GDP预测任务
│   ├── gdp_prediction_satmae.py
│   ├── metrics/                        # 评估结果
│   └── figures/                        # 可视化图表
│
├── pop/                                # 人口预测任务
│   ├── pop_prediction_satmae.py
│   ├── metrics/
│   └── figures/
│
├── lu/                                 # 土地利用预测任务
│   ├── lu_prediction_satmae.py
│   ├── metrics/
│   └── figures/
│
└── outputs/                            # 映射后的格网数据
    ├── shenzhen_grid_satmae_gdp_pop.geojson
    ├── beijing_grid_satmae_gdp_pop.geojson
    ├── shenzhen_grid_satmae_lu.geojson
    ├── beijing_grid_satmae_lu.geojson
    └── mapping_summary.json
```

## 快速开始

### 环境要求

```bash
# Python 3.8+
# PyTorch 1.10+ (with CUDA if available)
# scikit-learn, pandas, geopandas, rasterio
# timm==0.3.2 (SatMAE要求)
```

### 完整流程

```bash
cd /home/y/Code/GeoAI/ae/baseline/SatMAE/downstream

# 步骤1: 从Sentinel2影像提取tiles (约30分钟)
python extract_tiles_from_sentinel2.py

# 步骤2: 使用SatMAE提取嵌入 (约1小时, 需要GPU)
python extract_satmae_embeddings.py

# 步骤3: 映射嵌入到格网 (约5分钟)
python map_embeddings_to_grids.py

# 步骤4: 下游任务训练
cd gdp && python gdp_prediction_satmae.py
cd ../pop && python pop_prediction_satmae.py
cd ../lu && python lu_prediction_satmae.py
```

### 快速测试 (仅单城市)

```bash
# 仅处理深圳
python extract_tiles_from_sentinel2.py --city shenzhen
python extract_satmae_embeddings.py --city shenzhen
python map_embeddings_to_grids.py --city shenzhen
```

## 详细说明

### 步骤1: 提取Tiles

**脚本**: `extract_tiles_from_sentinel2.py`

**功能**:
- 从Sentinel2大幅影像中提取224×224像素的tiles
- 自动处理北京分块影像 (4个TIF文件)
- 保存为NPY格式 + 元数据CSV

**输入**:
- Sentinel2影像: `../data/Sentinel2/{city}_sentinel2_10m.tif` (4通道: B2, B3, B4, B8)
- 格网数据: `../data/Grid/grid_gdp_pop_builtup/*.geojson`
- 格网数据: `../data/Grid/grid_lu/*.geojson`

**通道处理**:
- Sentinel2影像有4个波段: B2(蓝), B3(绿), B4(红), B8(近红外)
- 自动提取RGB通道: [B4, B3, B2] → RGB顺序
- 输出为标准RGB格式供SatMAE使用

**输出**:
```
../out/data_processed/
├── shenzhen/
│   ├── tiles/{grid_id}_tile.npy
│   └── tile_metadata.csv
└── beijing/
    ├── tiles/{grid_id}_tile.npy
    └── tile_metadata.csv
```

**参数**:
- `--config`: 配置文件路径 (默认: config.yaml)
- `--city`: 指定城市 (shenzhen/beijing), 默认处理所有

### 步骤2: 提取嵌入

**脚本**: `extract_satmae_embeddings.py`

**功能**:
- 加载预训练SatMAE模型
- 批量处理tiles并提取1024维CLS token嵌入
- 保存为CSV (格式: grid_id, tile_id, embed_0, ..., embed_1023)

**输入**:
- 预训练权重: `../out/fmow_pretrain.pth`
- Tiles: `../out/data_processed/{city}/tiles/*.npy`
- 元数据: `../out/data_processed/{city}/tile_metadata.csv`

**输出**:
```
../out/data_processed/
├── shenzhen/satmae_embeddings.csv
└── beijing/satmae_embeddings.csv
```

**参数**:
- `--config`: 配置文件路径
- `--city`: 指定城市
- `--batch_size`: 批大小 (默认: 32)

**GPU要求**: 推荐使用GPU (CUDA)，CPU会很慢

### 步骤3: 映射格网

**脚本**: `map_embeddings_to_grids.py`

**功能**:
- 将tile级嵌入映射到grid级
- 合并标签数据 (GDP, POP, LU)
- 过滤有效样本

**输入**:
- 嵌入: `../out/data_processed/{city}/satmae_embeddings.csv`
- 格网: `../data/Grid/grid_gdp_pop_builtup/*.geojson`
- 格网: `../data/Grid/grid_lu/*.geojson`

**输出**:
```
outputs/
├── shenzhen_grid_satmae_gdp_pop.geojson
├── beijing_grid_satmae_gdp_pop.geojson
├── shenzhen_grid_satmae_lu.geojson
├── beijing_grid_satmae_lu.geojson
└── mapping_summary.json
```

### 步骤4: 下游任务

#### GDP预测

**脚本**: `gdp/gdp_prediction_satmae.py`

**任务类型**: 回归

**数据**:
- 深圳: ~1,125个建成区格网
- 北京: ~3,220个建成区格网

**模型**:
- Ridge(alpha=10.0)
- RandomForest(n_estimators=200, max_depth=10)
- GradientBoosting(n_estimators=150, max_depth=4)

**特性**:
- 对数变换 (`log1p`) 处理偏斜分布
- 80/20划分训练集/测试集
- StandardScaler特征标准化

**评估指标**: R², RMSE, MAE, MAPE

**输出**: `gdp/metrics/gdp_prediction_satmae_results.csv`

#### 人口预测

**脚本**: `pop/pop_prediction_satmae.py`

**实现**: 与GDP预测相同，仅替换标签为人口

**输出**: `pop/metrics/pop_prediction_satmae_results.csv`

#### 土地利用预测

**脚本**: `lu/lu_prediction_satmae.py`

**任务类型**: 多分类

**数据**:
- 深圳: ~2,173个格网
- 北京: ~16,964个格网

**模型**:
- RandomForest(n_estimators=200, class_weight='balanced')
- GradientBoosting(n_estimators=150, max_depth=5)

**特性**:
- 分层抽样 (`stratify=y`) 保持类别比例
- LabelEncoder编码土地利用类型
- 类别权重平衡

**评估指标**: Accuracy, F1-Score (Macro/Weighted), Classification Report

**输出**: `lu/metrics/lu_prediction_satmae_results.csv`

## 预期性能

基于Tile2Vec baseline和SatMAE更强的特征表示能力:

### GDP预测
- 深圳: Test R² ≈ 0.30-0.40 (Tile2Vec: 0.296)
- 北京: Test R² ≈ 0.20-0.30 (Tile2Vec: 0.199)

### 人口预测
- 深圳: Test R² ≈ 0.05-0.15 (Tile2Vec: 0.033)
- 北京: Test R² ≈ 0.43-0.55 (Tile2Vec: 0.427)

### 土地利用预测
- 深圳: Accuracy ≈ 60-75%
- 北京: Accuracy ≈ 70-85%

## 技术细节

### SatMAE模型配置

- **架构**: ViT-Large
- **嵌入维度**: 1024
- **层数**: 24
- **注意力头数**: 16
- **输入尺寸**: 224×224 RGB
- **特征提取**: CLS token (第0位)

### 数据预处理

**Sentinel2通道提取**:
```python
# Sentinel2: 4通道 [B2(蓝), B3(绿), B4(红), B8(近红外)]
# 提取RGB: [B4, B3, B2] -> (3, 224, 224) RGB顺序
tile_bgr = tile_data[:3, :, :]  # 取前3个通道
tile_rgb = np.array([tile_bgr[2], tile_bgr[1], tile_bgr[0]])  # BGR->RGB
```

**Sentinel2归一化**:
```python
# 原始DN值 (0-10000) -> 反射率 (0-1)
tile_normalized = np.clip(tile_data / 10000.0, 0, 1)
```

**GDP/人口对数变换**:
```python
# 训练时
y_train_log = np.log1p(y_train)

# 预测后逆变换
y_pred = np.expm1(model.predict(X_test))
```

### 北京分块影像处理

北京影像分为4个分块，根据格网中心坐标自动选择正确分块:
```
beijing_sentinel2_10m-0000000000-0000000000.tif  # 左下
beijing_sentinel2_10m-0000000000-0000011776.tif  # 右下
beijing_sentinel2_10m-0000011776-0000000000.tif  # 左上
beijing_sentinel2_10m-0000011776-0000011776.tif  # 右上
```

## 配置文件

`config.yaml`:
```yaml
paths:
  satmae_checkpoint: "/path/to/fmow_pretrain.pth"
  sentinel2_dir: "/path/to/Sentinel2"
  grid_dir: "/path/to/Grid"
  output_dir: "/path/to/data_processed"

model:
  embed_dim: 1024
  use_global_pool: false  # 使用CLS token
  device: "cuda"

data:
  tile_size: 224
  batch_size: 32
  cities: ["shenzhen", "beijing"]

training:
  test_size: 0.2
  random_state: 42
  use_log_transform: true
```

## 故障排除

### 问题1: CUDA out of memory

**解决**: 减小batch_size
```bash
python extract_satmae_embeddings.py --batch_size 16
```

### 问题2: 北京分块文件找不到

**检查**: 确认4个分块TIF文件都在 `data/Sentinel2/` 目录下

### 问题3: 嵌入值全是NaN

**原因**: tile数据可能有问题
**检查**:
1. 验证tile shape为 (3, 224, 224)
2. 验证像素值范围 (0-10000)
3. 检查Sentinel2影像波段顺序

### 问题4: 模型性能很差

**可能原因**:
1. 高维过拟合 - 尝试PCA降维
2. 数据归一化问题 - 检查Sentinel2预处理
3. 样本太少 - 考虑数据增强或简化模型

## 引用

如果使用本代码，请引用SatMAE论文:
```
@inproceedings{satmae2022,
  title={SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery},
  author={Yezhen Cong and Samar Khanna and Chenlin Meng and Patrick Liu and Erik Rozi and Yutong He and Marshall Burke and David B. Lobell and Stefano Ermon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## 许可

遵循SatMAE原始许可证。

## 联系

如有问题，请查看[计划文档](/home/y/.claude/plans/floating-giggling-harbor.md)。
