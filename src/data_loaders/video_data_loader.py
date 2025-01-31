import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from configs.V_config import Config

class VideoDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self.class_sample_counts = {}
        
        for class_name in Config.CLASSES:
            class_path = os.path.join(root_dir, class_name, split)
            
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist")
                continue
            
            video_count = 0
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
                
                self.samples.append({
                    'path': video_path,
                    'frames': frames,
                    'class': Config.CLASS_TO_IDX[class_name]
                })
                video_count += 1
            
            self.class_sample_counts[class_name] = video_count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['path']                             
        frames = sample['frames']
        
        num_frames = len(frames)
        
        # Check if the video has fewer frames than expected
        if num_frames < Config.FRAMES:
            # Pad the frames with the last frame to match Config.FRAMES
            frames = frames + [frames[-1]] * (Config.FRAMES - num_frames)
            frame_indices = np.arange(Config.FRAMES)
        else:
            # If the video has more frames, sample uniformly from them
            frame_indices = np.linspace(0, num_frames - 1, Config.FRAMES).astype(int)
        
        video_tensor = torch.zeros(Config.FRAMES, 3, 112, 112)
        
        for i, frame_idx in enumerate(frame_indices):
            frame_path = os.path.join(video_path, frames[frame_idx])
            frame = Image.open(frame_path).convert('RGB')
            frame_tensor = self.transform(frame)
            video_tensor[i] = frame_tensor
        
        # Permute the tensor to match the expected format: (C, T, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor, sample['class']

def get_dataloaders(root_dir, batch_size=Config.BATCH_SIZE):
    train_dataset = VideoDataset(root_dir, split='train')
    test_dataset = VideoDataset(root_dir, split='test')
    
    print("Dataset Information:")
    print(f"Total Classes: {len(Config.CLASSES)}")
    print("\nTrain Dataset:")
    for cls, count in train_dataset.class_sample_counts.items():
        print(f"- {cls}: {count} samples")
    
    print("\nTest Dataset:")
    for cls, count in test_dataset.class_sample_counts.items():
        print(f"- {cls}: {count} samples")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader
