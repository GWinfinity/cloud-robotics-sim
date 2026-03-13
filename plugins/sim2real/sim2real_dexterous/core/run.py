#!/usr/bin/env python3
"""
Genesis Sim-to-Real Dexterous Manipulation - Main Entry
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [train_expert|distill|tune|eval] [options]")
        print("\nCommands:")
        print("  train_expert    Train expert policy for a specific task")
        print("  distill         Distill experts into multi-task policy")
        print("  tune            Real-to-sim parameter tuning")
        print("  eval            Evaluate trained policy")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'train_expert':
        from scripts.train_experts import parse_args, train_expert
        args = parse_args()
        train_expert(args)
    
    elif command == 'distill':
        from scripts.distill_policy import parse_args, distill
        args = parse_args()
        distill(args)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
