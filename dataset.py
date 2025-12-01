import os
import cv2
import torch
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Helper function
def resize_keep_ar(frame, target_size=112):
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    frame = cv2.resize(frame, (nw, nh))

    # pad to square
    top = (target_size - nh) // 2
    bottom = target_size - nh - top
    left = (target_size - nw) // 2
    right = target_size - nw - left

    frame = cv2.copyMakeBorder(frame, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=0)
    return frame

class VideoDataset(Dataset):
    def __init__(self, root, subset='train', val_split=0.2, target_size=112, seed=42, transform=None, target_fps=15, max_duration=8):
        self.root = root
        self.target_size = target_size
        self.subset = subset
        self.target_fps = target_fps
        self.max_duration = max_duration
        self.samples = []
        
        # Ensure root exists
        if not os.path.exists(root):
             print(f"Warning: Root directory '{root}' does not exist.")
             self.classes = []
        else:
             self.classes = sorted(os.listdir(root))
             
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        random.seed(seed)

        for cl in self.classes:
            folder = os.path.join(root, cl)
            if not os.path.isdir(folder):
                continue
                
            video_files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            
            random.shuffle(video_files)
            split_idx = int(len(video_files) * (1 - val_split))
            
            if self.subset == 'train':
                selected_files = video_files[:split_idx]
            else: 
                selected_files = video_files[split_idx:]
                
            for f in selected_files:
                self.samples.append((os.path.join(folder, f), cl))

        print(f"Loaded {len(self.samples)} samples for subset: {self.subset}")

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if original_fps <= 0 or math.isnan(original_fps):
            original_fps = 30.0

        # --- RANDOM SAMPLING LOGIC START ---
        start_frame = 0
        end_frame = frame_count # Default to reading the whole video

        if self.max_duration is not None:
            max_frames = int(self.max_duration * original_fps)
            
            # If video is longer than max_duration, pick a random start point
            if frame_count > max_frames:
                # Random integer between 0 and (total - duration)
                start_frame = random.randint(0, frame_count - max_frames)
                end_frame = start_frame + max_frames
                
                # OPTIMIZATION: Jump directly to the start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # --- RANDOM SAMPLING LOGIC END ---

        step = original_fps / self.target_fps
        
        frames = []
        # Since we jumped (or started at 0), our current read position corresponds to start_frame
        current_frame_idx = start_frame 
        frames_collected = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Stop reading if we've reached the end of our sampled clip
            if current_frame_idx >= end_frame:
                break

            # Calculate the target frame index we need next
            # It is relative to where we started: start_frame + (n * step)
            target_idx_absolute = start_frame + (frames_collected * step)
            
            if current_frame_idx >= target_idx_absolute:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = resize_keep_ar(frame, self.target_size)
                
                # Normalize and add channel dim
                frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0) / 255.0
                frames.append(frame)
                frames_collected += 1
            
            current_frame_idx += 1

        cap.release()
        
        if not frames:
             # Return a placeholder if video failed
             return [torch.zeros((1, self.target_size, self.target_size))], self.target_fps
             
        return frames, self.target_fps

    def __getitem__(self, idx):
        path, cl = self.samples[idx]
        frames, fps = self.load_video(path)
        label = self.class_to_idx[cl]
        return frames, label

    def __len__(self):
        return len(self.samples)

# Collate function
def video_collate_fn(batch):
    videos, labels = zip(*batch)
    
    videos = [v for v in videos if len(v) > 0]
    
    if not videos:
        return torch.tensor([]), torch.tensor([])

    max_len = max(len(v) for v in videos)

    padded = []
    for v in videos:
        # v is a list of tensors [(1,H,W), ...]
        if len(v) < max_len:
            pad = [torch.zeros_like(v[0])] * (max_len - len(v))
            v = v + pad
        
        # Stack list of tensors -> (1, T, H, W)
        v_stacked = torch.stack(v, dim=1) 
        padded.append(v_stacked)

    # Stack batch -> (B, C, T, H, W)
    return torch.stack(padded, dim=0), torch.tensor(labels)

def save_video(tensor, path, fps=25):
    """
    tensor: shape [1, T, H, W], grayscale float tensor 0..1
    """
    tensor = tensor.squeeze(0)  # [T, H, W]

    frames = tensor.cpu().numpy()
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8)

    T, H, W = frames.shape

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    for i in range(T):
        frame = frames[i]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)

    writer.release()
    print("Saved video:", path)
    
    

    
if __name__ == '__main__':
    # Test block
    dataset = VideoDataset("datafolder", subset='val', target_size=112, target_fps=5, max_duration=8)
    print(f"Number of classessssss: {len(dataset.classes)}")
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=video_collate_fn)

        for videos, labels in loader:
            print(videos.shape)  # (B,C,T,H,W)
            print(labels)
            
            if videos.size(2) > 0:
                plt.imshow(videos[0, 0, 0].cpu().numpy(), cmap='gray')
                plt.show()
                video = videos[0]
                save_video(video, "video0.mp4", fps=5)
            break
    else:
        print("No videos found.")