#!/usr/bin/env python3
"""
BFM-Zero Main Entry
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [pretrain|zero_shot|few_shot] [options]")
        print("\nCommands:")
        print("  pretrain    Pre-train the FB model (unsupervised)")
        print("  zero_shot   Run zero-shot inference")
        print("  few_shot    Few-shot adaptation")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'pretrain':
        from scripts.train import parse_args, train
        args = parse_args()
        train(args)
    
    elif command == 'zero_shot':
        from scripts.zero_shot import parse_args, main as zero_shot_main
        args = parse_args()
        zero_shot_main(args)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
