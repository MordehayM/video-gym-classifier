from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import VideoDataset, video_collate_fn # Assuming these exist in your local files
from tqdm import tqdm

# --- CONFIGURATION ---
# Batch size must be larger than the number of GPUs. 
# E.g., if you have 2 GPUs, batch_size should be at least 2 (ideally 8, 16, etc.)
batch_size = 3
model_name = 'r2plus1d_18'  # Options: 'r2plus1d_18', 'mc3_18'
num_workers = 0 # Increased from 0 to speed up data loading for multiple GPUs
target_fps = 5
max_duration = 8  # seconds
transform = Compose([
    Resize((112, 112)),
    ToTensor()
])

# Ensure dataset_dir is correct for your environment
train_dataset = VideoDataset("datafolder", subset='train', target_size=200, transform=transform, target_fps=target_fps, max_duration=max_duration)
val_dataset   = VideoDataset("datafolder", subset='val', target_size=200, transform=transform, target_fps=target_fps, max_duration=max_duration)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=video_collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=video_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.classes)

# --- MODEL SETUP ---

# 1. Load the model normally
if model_name == 'r2plus1d_18':
    model = torchvision.models.video.r2plus1d_18(weights="DEFAULT")
elif model_name == 'mc3_18':
    model = torchvision.models.video.mc3_18(weights="DEFAULT")

# # 2. Freeze all layers
# for param in model.parameters():
#     param.requires_grad = False
# 2. Modify the first convolutional layer to accept 1 channel instead of 3
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

# # Unfreeze ONLY the first layer
# for param in model.stem[0].parameters():
#     param.requires_grad = True

# # Unfreeze final layer
# for param in model.fc.parameters():
#     param.requires_grad = True
    
# --- MULTI-GPU WRAPPING ---
if torch.cuda.device_count() > 1:
    print(f"🔥 Using {torch.cuda.device_count()} GPUs!")
    # This wraps the model to split batches across devices
    # model = nn.DataParallel(model)
else:
    print("Using single device:", device)

model = model.to(device)

# --- TRAINING SETUP ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses, val_accs = [], [], []

best_val_acc = 0.0
checkpoint_path = f"{model_name}_best_model.pt"
num_epochs = 10

# --- LOOP ---
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Training
    model.train()
    running_loss = 0.0
    
    # Added tqdm wrapper
    pbar = tqdm(train_loader, desc="Training")
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="Validating"):
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, pred = outputs.max(1)
            val_correct += pred.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Results - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print("🔼 Improvement! Saving checkpoint...")

        # HANDLE DATA PARALLEL SAVING
        # If wrapped in DataParallel, we must access .module to get the actual model weights
        # otherwise loading this model later on a CPU or single GPU will fail due to key mismatches.
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        torch.save({
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
        }, checkpoint_path)

# --- PLOTTING ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(val_accs, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig('training_curves.png') # Save plot to file so you can view it
print("Training complete. Plot saved to training_curves.png")