#!/usr/bin/env python3
"""
Genesis Table Tennis - Main Entry
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [train|eval|demo] [options]")
        print("\nCommands:")
        print("  train    Train the table tennis policy")
        print("  eval     Evaluate a trained policy")
        print("  demo     Run demo")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'train':
        from scripts.train import parse_args, train
        args = parse_args()
        train(args)
    
    elif command == 'eval':
        from scripts.eval import parse_args, evaluate
        args = parse_args()
        evaluate(args)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
