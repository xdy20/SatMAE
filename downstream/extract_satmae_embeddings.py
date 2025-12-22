"""
步骤2: 使用SatMAE提取特征嵌入

加载预训练的SatMAE模型，批量处理tiles并提取1024维嵌入
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml
import argparse

# 添加SatMAE路径
sys.path.insert(0, '/home/y/Code/GeoAI/ae/baseline/SatMAE')
from models_vit import vit_large_patch16


def load_config(config_path='config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_satmae_model(checkpoint_path, device='cuda'):
    """
    加载SatMAE预训练模型

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

    print(f"\n加载SatMAE模型...")
    print(f"  - 权重路径: {checkpoint_path}")
    print(f"  - 设备: {device}")

    # 创建ViT-Large模型 (num_classes=0表示不要分类头)
    model = vit_large_patch16(num_classes=0, global_pool=False)

    # 加载预训练权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

    # 设置为评估模式
    model = model.to(device)
    model.eval()

    print(f"✓ SatMAE模型加载完成")
    print(f"  - 嵌入维度: 1024")
    print(f"  - 输入尺寸: 224×224 RGB")
    print(f"  - 使用CLS token")

    return model


def preprocess_tile(tile_data):
    """
    预处理tile数据以适配SatMAE输入

    参数:
        tile_data: (3, 224, 224) numpy array, 原始DN值 (0-10000)

    返回:
        tile_tensor: (1, 3, 224, 224) tensor, 归一化到[0,1]
    """
    # 归一化: DN值除以10000转为反射率
    tile_normalized = np.clip(tile_data / 10000.0, 0, 1)

    # 转为tensor
    tile_tensor = torch.from_numpy(tile_normalized).float()

    # 添加batch维度
    tile_tensor = tile_tensor.unsqueeze(0)  # (1, 3, 224, 224)

    return tile_tensor


def extract_embeddings_for_city(city_name, tiles_dir, metadata_csv, checkpoint_path,
                                  batch_size=32, device='cuda'):
    """
    为一个城市的所有tiles提取嵌入

    参数:
        city_name: 城市名称
        tiles_dir: tiles目录
        metadata_csv: tile元数据CSV路径
        checkpoint_path: SatMAE权重路径
        batch_size: 批大小
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"处理城市: {city_name.upper()}")
    print(f"{'='*60}")

    # 加载元数据
    metadata_df = pd.read_csv(metadata_csv)
    print(f"✓ 加载元数据: {len(metadata_df)}个tiles")

    # 加载SatMAE模型
    model = load_satmae_model(checkpoint_path, device=device)

    # 提取嵌入
    all_embeddings = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(metadata_df), batch_size),
                                 desc=f"提取{city_name}的嵌入"):
            batch_end = min(batch_start + batch_size, len(metadata_df))
            batch_metadata = metadata_df.iloc[batch_start:batch_end]

            # 加载批次tiles
            batch_tiles = []
            valid_indices = []

            for idx, row in batch_metadata.iterrows():
                grid_id = row['grid_id']
                tile_path = tiles_dir / f"{grid_id}_tile.npy"

                try:
                    # 加载tile
                    tile_data = np.load(tile_path)  # (3, 224, 224)

                    # 检查shape
                    if tile_data.shape != (3, 224, 224):
                        print(f"⚠ 跳过tile {grid_id}: shape={tile_data.shape}")
                        continue

                    # 预处理
                    tile_tensor = preprocess_tile(tile_data)
                    batch_tiles.append(tile_tensor)
                    valid_indices.append(idx)

                except Exception as e:
                    print(f"⚠ 加载tile {grid_id}失败: {e}")
                    continue

            if len(batch_tiles) == 0:
                continue

            # 拼接为batch
            batch_tensor = torch.cat(batch_tiles, dim=0).to(device)  # (B, 3, 224, 224)

            # 前向传播提取特征
            # SatMAE ViT返回CLS token嵌入 (B, 1024)
            batch_embeddings = model(batch_tensor)  # (B, 1024)

            # 转到CPU并保存
            all_embeddings.append(batch_embeddings.cpu().numpy())

            # 清空GPU缓存
            del batch_tensor, batch_embeddings
            if device == 'cuda':
                torch.cuda.empty_cache()

    # 合并所有嵌入
    if len(all_embeddings) == 0:
        print("✗ 错误: 未提取到任何嵌入")
        return

    embeddings = np.vstack(all_embeddings)  # (N, 1024)

    print(f"\n✓ 嵌入提取完成")
    print(f"  - 嵌入shape: {embeddings.shape}")
    print(f"  - 均值: {embeddings.mean():.6f}")
    print(f"  - 标准差: {embeddings.std():.6f}")
    print(f"  - 范围: [{embeddings.min():.6f}, {embeddings.max():.6f}]")
    print(f"  - NaN数量: {np.isnan(embeddings).sum()}")

    # 保存为CSV
    output_dir = Path(metadata_csv).parent
    embeddings_csv = output_dir / 'satmae_embeddings.csv'

    # 创建DataFrame
    embed_cols = [f'embed_{i}' for i in range(embeddings.shape[1])]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols)

    # 添加grid_id和tile_id
    embed_df['grid_id'] = metadata_df['grid_id'].values[:len(embeddings)]
    embed_df['tile_id'] = metadata_df['tile_id'].values[:len(embeddings)]

    # 重排列列顺序
    cols = ['grid_id', 'tile_id'] + embed_cols
    embed_df = embed_df[cols]

    # 保存
    embed_df.to_csv(embeddings_csv, index=False)

    print(f"✓ 嵌入已保存到: {embeddings_csv}")
    print(f"  - 格网数量: {len(embed_df)}")
    print(f"  - 嵌入维度: {embeddings.shape[1]}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用SatMAE提取嵌入')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--city', type=str, default=None, help='指定城市 (shenzhen/beijing), 默认处理所有')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小 (覆盖配置文件)')
    args = parser.parse_args()

    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # 加载配置
    config = load_config(args.config)
    checkpoint_path = config['paths']['satmae_checkpoint']
    output_dir = Path(config['paths']['output_dir'])
    batch_size = args.batch_size if args.batch_size else config['data']['batch_size']
    device = config['model']['device']
    cities = [args.city] if args.city else config['data']['cities']

    print("="*60)
    print("步骤2: 使用SatMAE提取嵌入")
    print("="*60)
    print(f"权重路径: {checkpoint_path}")
    print(f"输出目录: {output_dir}")
    print(f"批大小: {batch_size}")
    print(f"设备: {device}")
    print(f"处理城市: {cities}")

    # 检查权重文件
    if not Path(checkpoint_path).exists():
        print(f"✗ 错误: 权重文件不存在: {checkpoint_path}")
        return

    # 处理每个城市
    for city in cities:
        try:
            city_dir = output_dir / city
            tiles_dir = city_dir / 'tiles'
            metadata_csv = city_dir / 'tile_metadata.csv'

            if not tiles_dir.exists():
                print(f"⚠ 跳过{city}: tiles目录不存在")
                continue

            if not metadata_csv.exists():
                print(f"⚠ 跳过{city}: 元数据文件不存在")
                continue

            extract_embeddings_for_city(
                city_name=city,
                tiles_dir=tiles_dir,
                metadata_csv=metadata_csv,
                checkpoint_path=checkpoint_path,
                batch_size=batch_size,
                device=device
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
