"""
Video recording wrapper.
"""

import os
import numpy as np
from typing import Optional
import gymnasium as gym
import imageio


class RecordVideo(gym.Wrapper):
    """
    Record video of environment episodes.
    
    Args:
        env: Environment to wrap
        video_folder: Folder to save videos
        episode_trigger: Function to determine which episodes to record
        step_trigger: Function to determine which steps to record
        video_length: Maximum length of video in steps
        name_prefix: Prefix for video filenames
    """
    
    def __init__(
        self,
        env,
        video_folder: str = "videos",
        episode_trigger: Optional[callable] = None,
        step_trigger: Optional[callable] = None,
        video_length: int = 1000,
        name_prefix: str = "video",
    ):
        super().__init__(env)
        
        self.video_folder = video_folder
        self.episode_trigger = episode_trigger or (lambda ep: ep % 10 == 0)
        self.step_trigger = step_trigger
        self.video_length = video_length
        self.name_prefix = name_prefix
        
        # Create folder
        os.makedirs(video_folder, exist_ok=True)
        
        # Recording state
        self.episode_id = 0
        self.step_id = 0
        self.recording = False
        self.frames = []
        
    def reset(self, **kwargs):
        """Reset environment and start recording if triggered."""
        # Save previous episode if recording
        if self.recording and len(self.frames) > 0:
            self._save_video()
        
        obs, info = self.env.reset(**kwargs)
        
        # Check if should record this episode
        self.recording = self.episode_trigger(self.episode_id)
        self.frames = []
        self.step_id = 0
        
        if self.recording:
            frame = self._get_frame()
            if frame is not None:
                self.frames.append(frame)
        
        self.episode_id += 1
        return obs, info
    
    def step(self, action):
        """Step environment and record frame if needed."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.recording:
            frame = self._get_frame()
            if frame is not None:
                self.frames.append(frame)
            
            self.step_id += 1
            
            # Stop recording if max length reached
            if self.step_id >= self.video_length:
                self._save_video()
                self.recording = False
        
        return obs, reward, terminated, truncated, info
    
    def _get_frame(self):
        """Get frame from environment render."""
        frame = self.env.render()
        return frame
    
    def _save_video(self):
        """Save recorded video."""
        if len(self.frames) == 0:
            return
        
        filename = f"{self.name_prefix}_episode_{self.episode_id}.mp4"
        filepath = os.path.join(self.video_folder, filename)
        
        # Convert frames to video
        frames_array = np.array(self.frames)
        imageio.mimsave(filepath, frames_array, fps=30)
        
        print(f"Saved video: {filepath}")
        self.frames = []
    
    def close(self):
        """Close environment and save final video."""
        if self.recording and len(self.frames) > 0:
            self._save_video()
        super().close()
