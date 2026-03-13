#!/usr/bin/env python3
"""
Genesis SLAC - Main Entry
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [pretrain|finetune] [options]")
        print("\nCommands:")
        print("  pretrain    Pretrain latent action space in simulation")
        print("  finetune    Fine-tune on real-world downstream tasks")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'pretrain':
        from scripts.pretrain import parse_args, pretrain
        args = parse_args()
        pretrain(args)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
