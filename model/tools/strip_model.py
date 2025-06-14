import argparse
import os
import sys
from pathlib import Path
import torch

def strip_optimizer(f='best.pt', s=''):
    """
    移除模型权重中的优化器状态，并保存简化后的模型。

    参数:
        f (str): 原始模型权重文件路径。
        s (str): 简化后模型的保存路径。如果为空，将覆盖原文件。
    """
    print(f'加载模型权重文件：{f}')
    try:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
    except Exception as e:
        print(f'无法加载模型权重文件：{e}')
        sys.exit(1)

    # 移除优化器状态和其他不必要的信息
    keys_to_remove = ['optimizer', 'best_fitness', 'ema', 'updates', 'training_results', 'wandb_id']
    for k in keys_to_remove:
        if k in checkpoint:
            del checkpoint[k]
            print(f'已移除键：{k}')

    # 重置 epoch 信息
    if 'epoch' in checkpoint:
        checkpoint['epoch'] = -1
        print('重置 epoch 信息为 -1')
    else:
        print('未找到 "epoch" 键，跳过重置。')

    # 确定保存路径
    save_path = s if s else f
    print(f'保存简化后的模型到：{save_path}')

    # 保存简化后的模型
    try:
        torch.save(checkpoint, save_path)
        print('优化器状态已成功移除并保存简化后的模型。')
    except Exception as e:
        print(f'保存简化后的模型失败：{e}')
        sys.exit(1)

def parse_opt():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Strip optimizer from YOLO model.')
    parser.add_argument('--weights', type=str, default='../weights/val/visdrone_400.pt', help='模型权重文件路径，例如：path/to/model.pt')
    parser.add_argument('--save_dir', type=str, default='', help='保存简化后模型的目录，默认覆盖原文件')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    return parser.parse_args()

def main():
    """主函数"""
    opt = parse_opt()

    # 如果需要详细日志，可以在此处设置
    if opt.verbose:
        print('启用详细日志输出。')

    weights_path = Path(opt.weights).resolve()
    if not weights_path.is_file():
        print(f'错误：权重文件不存在：{weights_path}')
        sys.exit(1)

    # 确定保存路径
    if opt.save_dir:
        save_dir = Path(opt.save_dir)
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f'创建保存目录：{save_dir}')
        except Exception as e:
            print(f'无法创建保存目录 {save_dir}：{e}')
            sys.exit(1)
        stripped_weights_path = save_dir / weights_path.name
    else:
        # 默认覆盖原文件
        stripped_weights_path = weights_path

    print(f'正在移除优化器状态：{weights_path}')
    print(f'保存简化后模型到：{stripped_weights_path}')

    # 调用 strip_optimizer 函数
    try:
        strip_optimizer(str(weights_path), str(stripped_weights_path))
    except Exception as e:
        print(f'移除优化器状态失败：{e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
