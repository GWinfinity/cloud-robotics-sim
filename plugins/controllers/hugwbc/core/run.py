#!/usr/bin/env python3
"""
HugWBC Genesis - Main Entry Point

HugWBC 在 Genesis 物理引擎中的实现

用法:
    python run.py train --task h1_loco
    python run.py eval --task h1_loco --checkpoint logs/h1_loco/checkpoints/best_model.pt
    python run.py visualize --task h1_loco --checkpoint logs/h1_loco/checkpoints/best_model.pt
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print("""
HugWBC Genesis - Humanoid Whole-Body Controller

用法:
    python run.py [command] [options]

Commands:
    train       训练策略
    eval        评估策略
    visualize   可视化策略
    
Examples:
    # 训练平地行走任务
    python run.py train --task h1_loco --num-envs 4096
    
    # 训练上下楼梯任务
    python run.py train --task h1_stairs --config configs/h1_stairs.yaml
    
    # 评估策略
    python run.py eval --task h1_loco --checkpoint logs/h1_loco/checkpoints/best_model.pt
    
    # 可视化策略
    python run.py visualize --task h1_loco --checkpoint logs/h1_loco/checkpoints/best_model.pt --command 1.0 0.0 0.0
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # 移除命令参数
    
    try:
        if command == 'train':
            from scripts.train import parse_args, train
            args = parse_args()
            train(args)
        
        elif command == 'eval':
            from scripts.eval import parse_args, evaluate
            args = parse_args()
            evaluate(args)
        
        elif command == 'visualize':
            from scripts.visualize import parse_args, visualize
            args = parse_args()
            visualize(args)
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: train, eval, visualize")
            sys.exit(1)
    except Exception as e:
        print(f"Error executing command '{command}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
