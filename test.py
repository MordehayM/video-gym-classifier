import torch
import torchvision
import torch.nn as nn
from dataset import load_video  # or write a small video loader
import os
import json

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

def load_video(path, max_duration=8, target_fps=5):
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
            return [torch.zeros((1, target_size, target_size))], target_fps
            
    return frames, target_fps


with open("class_mappings.json", "r") as f:
    data = json.load(f)

idx_to_class = data["idx_to_class"]

num_classes = len(idx_to_class)
print("Number of classes:", num_classes)

model_name = 'r2plus1d_18'
# 1. Load the model normally
if model_name == 'r2plus1d_18':
    model = torchvision.models.video.r2plus1d_18(weights="DEFAULT")
elif model_name == 'mc3_18':
    model = torchvision.models.video.mc3_18(weights="DEFAULT")

old_layer = model.stem[0]

new_layer = nn.Conv3d(
    in_channels=1, 
    out_channels=old_layer.out_channels, 
    kernel_size=old_layer.kernel_size, 
    stride=old_layer.stride, 
    padding=old_layer.padding, 
    bias=old_layer.bias
)

# 3. Copy pre-trained weights (Sum RGB channels to Grayscale)
with torch.no_grad():
    new_layer.weight[:] = old_layer.weight.sum(dim=1, keepdim=True)

# Replace the layer
model.stem[0] = new_layer

# 4. Update the final classification layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

checkpoint = torch.load("r2plus1d_18_best_model.pt", map_location=device)


model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Loaded model with best val acc:", checkpoint["val_acc"])


video, _ = load_video("/home/dsi/moradim/video-classifier/datafolder/pull Up/pull up_2.mp4")   # list of frames
video = torch.stack(video, dim=1).to(device)

with torch.no_grad():
    output = model(video.unsqueeze(0))    # add batch dim (1, C, T, H, W)
    pred_class = output.argmax(dim=1).item()
    print(f"The predicted class is: {pred_class} with probability {output.softmax(dim=1)[0, pred_class].item():.4f}")
    pred_class_name = idx_to_class[str(pred_class)]

print("Predicted class:", pred_class_name)