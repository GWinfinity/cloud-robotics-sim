"""
Logger for HugWBC Training
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json


class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件
        self.log_file = os.path.join(log_dir, 'train.log')
        
        # 尝试导入 tensorboard
        self.use_tensorboard = False
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.use_tensorboard = True
        except ImportError:
            print("Warning: tensorboard not available, logging to console only")
        
        # 记录启动信息
        self.info(f"Logger initialized: {log_dir}")
    
    def info(self, message: str):
        """记录信息日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        # 打印到控制台
        print(log_message)
        
        # 写入文件
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """记录多个标量值"""
        if self.use_tensorboard:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """记录直方图"""
        if self.use_tensorboard:
            self.writer.add_histogram(tag, values, step)
    
    def save_config(self, config: Dict[str, Any]):
        """保存配置"""
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.info(f"Config saved to {config_path}")
    
    def close(self):
        """关闭日志记录器"""
        if self.use_tensorboard:
            self.writer.close()
