import torch
from configs.V_config import Config
from data_loaders.video_data_loader import get_dataloaders
from models.video_model import AdvancedVideoClassifier
from trainers.video_train_pipe import TrainingPipeline

def main():
    # Create necessary directories
    Config.create_directories()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load data
    train_loader, test_loader = get_dataloaders(
        Config.DATA_DIR, 
        batch_size=Config.BATCH_SIZE
    )
    
    # Initialize model
    model = AdvancedVideoClassifier(
        num_classes=Config.NUM_CLASSES
    )
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        num_classes=Config.NUM_CLASSES,
        experiment_name=Config.EXPERIMENT_NAME
    )
    
    # Run training
    pipeline.run_training(epochs=Config.NUM_EPOCHS)

if __name__ == '__main__':
    main()