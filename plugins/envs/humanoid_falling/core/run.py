#!/usr/bin/env python3
"""
Genesis Humanoid Falling Protection - 主入口

简化命令:
  python run.py train
  python run.py eval --checkpoint checkpoints/best_model.pt
  python run.py visualize --checkpoint checkpoints/best_model.pt
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [train|eval|visualize] [options]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # 移除命令参数，保留其他参数
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'train':
        from scripts.train import parse_args, train
        args = parse_args()
        train(args)
    
    elif command == 'eval':
        from scripts.eval import parse_args, evaluate
        args = parse_args()
        evaluate(args)
    
    elif command == 'visualize':
        from scripts.visualize import parse_args, main as visualize_main
        args = parse_args()
        visualize_main(args)
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, eval, visualize")
        sys.exit(1)


if __name__ == '__main__':
    main()
