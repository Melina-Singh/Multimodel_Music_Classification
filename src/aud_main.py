import torch
from configs.A_config import Config
from data_loaders.audio_data_loader import get_data_loaders
from trainers.audio_train_pipe import train_model
from models.audio_model import VGGNetAudio


def main():
    # Preprocessing (optional if already done)
    # preprocessor = SpectrogramPreprocessor(root_dir=Config.DATA_DIR)
    # preprocessor.preprocess()
    
    # Load data
    train_loader, val_loader = get_data_loaders(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE
    )
    
    # Initialize model
    model = VGGNetAudio(num_classes=Config.NUM_CLASSES)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Run training
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs= Config.NUM_EPOCHS
    )

if __name__ == '__main__':
    main()