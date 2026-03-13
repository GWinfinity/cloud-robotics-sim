"""
训练日志记录器
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any
import numpy as np


class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 指标历史
        self.metrics_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'curriculum_level': []
        }
        
        # 训练统计
        self.start_time = time.time()
        self.total_steps = 0
        
        # 打开日志文件
        self.log_file = open(os.path.join(self.log_dir, 'train.log'), 'w')
        
        # 保存配置
        self.config = None
        
        self.print_and_log(f"Logger initialized. Log directory: {self.log_dir}")
    
    def set_config(self, config: Dict):
        """设置并保存配置"""
        self.config = config
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.print_and_log(f"Config saved to {config_path}")
    
    def print_and_log(self, message: str):
        """打印并记录消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_file.write(log_message + '\n')
        self.log_file.flush()
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """
        记录训练指标
        
        Args:
            step: 当前训练步数
            metrics: 指标字典
        """
        self.total_steps = step
        
        # 更新时间戳
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        # 构建日志消息
        message = f"Step {step} | Time {hours:02d}:{minutes:02d}:{seconds:02d}\n"
        
        for key, value in metrics.items():
            if isinstance(value, float):
                message += f"  {key}: {value:.4f}\n"
            else:
                message += f"  {key}: {value}\n"
            
            # 保存到历史
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((step, value))
        
        self.print_and_log(message)
    
    def log_episode(self, episode: int, reward: float, length: int, success: bool):
        """记录回合信息"""
        self.metrics_history['episode_rewards'].append((episode, reward))
        self.metrics_history['episode_lengths'].append((episode, length))
        
        if episode % 10 == 0:
            recent_rewards = [r for _, r in self.metrics_history['episode_rewards'][-100:]]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            
            self.print_and_log(
                f"Episode {episode} | Reward: {reward:.2f} | "
                f"Length: {length} | Avg Reward (100): {avg_reward:.2f}"
            )
    
    def save_metrics(self):
        """保存指标历史到文件"""
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        
        # 转换为可序列化的格式
        serializable_metrics = {}
        for key, values in self.metrics_history.items():
            serializable_metrics[key] = [
                {'step': step, 'value': float(value)} 
                for step, value in values
            ]
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        self.print_and_log(f"Metrics saved to {metrics_path}")
    
    def close(self):
        """关闭日志记录器"""
        self.save_metrics()
        self.log_file.close()
        self.print_and_log("Logger closed.")


class TensorBoardLogger(Logger):
    """TensorBoard 日志记录器"""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        super().__init__(log_dir, experiment_name)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            self.has_tensorboard = True
        except ImportError:
            self.print_and_log("Warning: tensorboard not installed, using file logging only")
            self.has_tensorboard = False
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """记录指标到 TensorBoard"""
        super().log_metrics(step, metrics)
        
        if self.has_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """记录直方图"""
        if self.has_tensorboard:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """关闭日志记录器"""
        if self.has_tensorboard:
            self.writer.close()
        super().close()
