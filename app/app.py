import torch
import torch.nn as nn
import gradio as gr
import numpy as np
import cv2
import librosa
import librosa.display
import os
import time
from functools import lru_cache
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F

# Video Model (Existing Code)
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AdvancedVideoClassifier(nn.Module):
    def __init__(self, num_classes=5, input_channels=3):
        super(AdvancedVideoClassifier, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.adaptpool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock3D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Audio Model (Provided Code)
# class AudioSpectrogramTransformer(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         # Load pre-trained Vision Transformer
#         weights = ViT_B_16_Weights.DEFAULT
#         self.backbone = vit_b_16(weights=weights)
#         # Modify input layer for single-channel spectrograms
#         self.backbone.conv_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
#         # Replace classification head
#         self.backbone.heads = nn.Sequential(
#             nn.Linear(self.backbone.hidden_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         return self.backbone(x)

class VGGNetAudio(nn.Module):
    def __init__(self, config):
        """
        VGG-like architecture for audio classification with spectrograms (4D input).
        
        :param config: Configuration object that includes hyperparameters such as 
                        number of classes, input channels, dropout rate, etc.
        """
        super(VGGNetAudio, self).__init__()
        
        self.num_classes = config.NUM_CLASSES
        self.input_channels = config.INPUT_CHANNELS
        self.dropout_rate = 0.5
        self.hidden_units = 4096
        
        # 2D Convolutions for spectrogram data (4D input)
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)

        # Max pooling after each convolution
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)

        # Fully connected layers
        # Update the input size of the fully connected layer to match the flattened size
        # Assuming input size of (224, 224) for spectrograms
        # After 4 pooling layers, the size becomes (14, 14)
        self.fc1 = nn.Linear(512 * 7 * 7, self.hidden_units)  # Correct size after pooling
        self.fc2 = nn.Linear(self.hidden_units, self.num_classes)

    def forward(self, x):
        """
        Forward pass for the model
        
        :param x: Input tensor with shape (batch_size, channels, height, width)
        :return: Output tensor (classification logits)
        """
        # Convolution + BatchNorm + ReLU + Pooling + Dropout
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # Conv5 -> ReLU -> MaxPool

        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten (batch_size, 512 * 7 * 7)

        # Fully connected layers with ReLU activation and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout regularization
        x = self.fc2(x)  # Final output layer

        return x


# Load Video Model
def load_video_model(best_model_path):
    try:
        best_model_checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
        model = AdvancedVideoClassifier(
            num_classes=best_model_checkpoint.get('num_classes', 5),
            input_channels=best_model_checkpoint.get('input_shape', (3, 30, 112, 112))[0]
        )
        model.load_state_dict(best_model_checkpoint.get('model_state_dict', best_model_checkpoint))
        model.eval()
        class_names = best_model_checkpoint.get('class_mapping', ["bansuri", "deuda", "jhora", "kartik", "lahare"])
        return model, class_names
    except Exception as e:
        print(f"Error loading video model: {e}")
        raise

# Load Audio Model
def load_audio_model(audio_model_path, num_classes):
    try:
        model = VGGNetAudio(num_classes=num_classes)
        checkpoint = torch.load(audio_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading audio model: {e}")
        raise

# Preprocess Video
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)
    cap.release()
    frame_count = len(frames)
    if frame_count >= 30:
        indices = np.linspace(0, frame_count - 1, 30).astype(int)
        frames = [frames[i] for i in indices]
    else:
        padding = 30 - frame_count
        frames.extend([np.zeros_like(frames[0]) for _ in range(padding)])
    frames = np.stack(frames, axis=0)
    frames = frames.transpose(3, 0, 1, 2)
    frames = torch.tensor(frames, dtype=torch.float32) / 255.0
    frames = frames.unsqueeze(0)
    return frames

# Cache Spectrogram Generation
@lru_cache(maxsize=10)  # Cache up to 10 spectrograms
def generate_spectrogram(video_path):
    # Extract audio from video
    audio, sr = librosa.load(video_path, sr=None)
    # Generate spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    # Resize spectrogram to match model input size
    spectrogram = cv2.resize(spectrogram, (224, 224))  # ViT input size
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return spectrogram

# Predict with Ensemble
def predict_ensemble(video_model, audio_model, class_names, video_path):
    # Video prediction
    video_tensor = preprocess_video(video_path)
    with torch.no_grad():
        video_outputs = video_model(video_tensor)
        video_probs = torch.softmax(video_outputs, dim=1).numpy()[0]

    # Audio prediction
    spectrogram = generate_spectrogram(video_path)  # Use cached spectrogram
    with torch.no_grad():
        audio_outputs = audio_model(spectrogram)
        audio_probs = torch.softmax(audio_outputs, dim=1).numpy()[0]

    # Compare confidence scores
    video_confidence = max(video_probs)
    audio_confidence = max(audio_probs)

    if video_confidence > audio_confidence:
        predictions = {class_names[i]: float(video_probs[i]) for i in range(len(class_names))}
    else:
        predictions = {class_names[i]: float(audio_probs[i]) for i in range(len(class_names))}

    return predictions

# Gradio Interface
def create_gradio_interface(video_model, audio_model, class_names):
    def prediction_fn(video_path):
        return predict_ensemble(video_model, audio_model, class_names, video_path)

    iface = gr.Interface(
        fn=prediction_fn,
        inputs=gr.Video(),
        outputs=gr.Label(num_top_classes=5),
        title="Video Classification with Audio Ensemble",
        description="Upload a video to classify it using both video and audio models."
    )
    return iface



# Main Execution
if __name__ == "__main__":
    video_model_path = r'D:\.CV_Projects\only_vids\runs\instrument_classification\checkpoints\best_model.pth'
    audio_model_path = r"D:\.CV_Projects\Only_audio\output\checkpoints\best_model.pth"  # Update with your audio model path
    num_classes = 5  # Update with the number of classes

    # Load models
    video_model, class_names = load_video_model(video_model_path)
    audio_model = load_audio_model(audio_model_path, num_classes)

    # Create and launch Gradio interface
    iface = create_gradio_interface(video_model, audio_model, class_names)
    iface.launch(server_port=7862, share=False, debug=True)  # Use a different port
    
# # Main Execution
# if __name__ == "__main__":
#     video_model_path = r'D:\.CV_Projects\only_vids\runs\instrument_classification\checkpoints\best_model.pth'
#     audio_model_path =  r"D:\.CV_Projects\Only_audio\output\checkpoints\best_model.pth"  # Update with your audio model path
#     num_classes = 5  # Update with the number of classes

#     # Load models
#     video_model, class_names = load_video_model(video_model_path)
#     audio_model = load_audio_model(audio_model_path, num_classes)

#     # Create and launch Gradio interface
#     iface = create_gradio_interface(video_model, audio_model, class_names)
#     iface.launch(server_port=7861, share=False, debug=True)  # Updated launch settings