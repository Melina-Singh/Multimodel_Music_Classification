import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from src.configs.A_config import Config

class SpectrogramPreprocessor:
    def __init__(self, root_dir, output_size=(224, 224), normalize=True):
        self.root_dir = root_dir
        self.output_size = output_size
        self.normalize = normalize

        self.transform = transforms.Compose([
            transforms.Resize(self.output_size),
            transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel (grayscale)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) if self.normalize else transforms.ToTensor()
        ])

    def preprocess(self):
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for split in ["train", "test"]:
                split_path = os.path.join(class_path, split)
                if not os.path.isdir(split_path):
                    continue

                for file_name in os.listdir(split_path):
                    file_path = os.path.join(split_path, file_name)
                    if file_name.endswith(".png"):
                        image = Image.open(file_path).convert("L")  # Convert directly to grayscale ('L')

                        processed_image = self.transform(image)
                        torch.save(processed_image, file_path.replace(file_name.split('.')[-1], "pt"))

                        # Save processed image back (optional; keeping it as a tensor in memory is possible too)
                        # torch.save(processed_image, file_path.replace(".png", ".pt"))

        print("Preprocessing complete.")

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.samples = []
        self.labels = []

        for cls in Config.CLASSES:
            cls_path = os.path.join(data_dir, cls, split)
            if os.path.exists(cls_path):
                for file_name in os.listdir(cls_path):
                    if file_name.endswith(".pt"):
                        self.samples.append(os.path.join(cls_path, file_name))
                        self.labels.append(Config.CLASS_TO_IDX[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spec_path = self.samples[idx]
        spectrogram = torch.load(spec_path)
        label = self.labels[idx]

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

def get_data_loaders(data_dir, batch_size=None):
    batch_size = batch_size or Config.BATCH_SIZE
    
    train_dataset = SpectrogramDataset(data_dir, split='train')
    val_dataset = SpectrogramDataset(data_dir, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    # Preprocessing step
    preprocessor = SpectrogramPreprocessor(root_dir=Config.DATA_DIR)
    preprocessor.preprocess()

    # DataLoader setup
    train_loader, val_loader = get_data_loaders(Config.DATA_DIR)

    # Example usage
    for inputs, labels in train_loader:
        print("Input shape:", inputs.shape)
        print("Label shape:", labels.shape)
        break