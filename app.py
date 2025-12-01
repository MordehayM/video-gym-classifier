import gradio as gr
import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
import json
import random
import math
import os

# --- 1. HELPER FUNCTIONS ---

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

def load_video_frames(path, max_duration=8, target_fps=5):
    target_size = 200
    cap = cv2.VideoCapture(path)
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if original_fps <= 0 or math.isnan(original_fps):
        original_fps = 30.0

    # --- RANDOM SAMPLING LOGIC START ---
    start_frame = 0
    end_frame = frame_count # Default to reading the whole video

    if max_duration is not None:
        max_frames = int(max_duration * original_fps)
        
        # If video is longer than max_duration, pick a random start point
        if frame_count > max_frames:
            # Random integer between 0 and (total - duration)
            start_frame = random.randint(0, frame_count - max_frames)
            end_frame = start_frame + max_frames
            
            # OPTIMIZATION: Jump directly to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # --- RANDOM SAMPLING LOGIC END ---

    step = original_fps / target_fps
    
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
            frame = resize_keep_ar(frame, target_size)
            
            # Normalize and add channel dim
            frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0) / 255.0
            frames.append(frame)
            frames_collected += 1
        
        current_frame_idx += 1

    cap.release()
    
    if not frames:
        # Return a placeholder if video failed
        return torch.zeros((1, 1, target_size, target_size)), target_fps
            
    return torch.stack(frames, dim=1) # (C, T, H, W)

# --- 2. MODEL SETUP ---

# Load Mappings
with open("class_mappings.json", "r") as f:
    data = json.load(f)

idx_to_class = data["idx_to_class"]
num_classes = len(idx_to_class)

# Define Model Architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.video.r2plus1d_18(weights=None) # No need for internet weights, we load ours

# Modify first layer for Grayscale
old_layer = model.stem[0]
new_layer = nn.Conv3d(
    in_channels=1, 
    out_channels=old_layer.out_channels, 
    kernel_size=old_layer.kernel_size, 
    stride=old_layer.stride, 
    padding=old_layer.padding, 
    bias=old_layer.bias
)
model.stem[0] = new_layer

# Modify final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load Weights
if os.path.exists("r2plus1d_18_best_model.pt"):
    checkpoint = torch.load("r2plus1d_18_best_model.pt", map_location=device)
    # Check if checkpoint is full dict or state_dict
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
else:
    print("Warning: Model weights not found.")

model = model.to(device)
model.eval()

# --- 3. INFERENCE FUNCTION ---

def predict(video_path):
    if video_path is None:
        return None
    
    # Process Video
    video_tensor = load_video_frames(video_path)
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        # Add batch dimension: (1, C, T, H, W)
        output = model(video_tensor.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Create a dictionary for Gradio Label {Class: Probability}
    confidences = {idx_to_class[str(i)]: float(probabilities[i]) for i in range(len(probabilities))}
    
    return confidences

# --- 4. GRADIO INTERFACE ---

interface = gr.Interface(
    fn=predict,
    inputs=gr.Video(
        label="Upload or Record Video",
        sources=["upload", "webcam"], # <--- THIS ENABLES THE CAMERA
        format="mp4" # Ensures the recorded video is saved in a format OpenCV likes
    ),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Video Action Classifier",
    description="Upload a video or record yourself to classify the action."
)

if __name__ == "__main__":
    interface.launch()