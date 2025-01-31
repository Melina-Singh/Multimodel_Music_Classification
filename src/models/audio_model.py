import torch
import torch.nn as nn
import torch.nn.functional as F

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
