
import os
import torch

class Config:
    # Dataset Configuration
    DATA_DIR = r'D:\.CV_Projects\Only_audio\Spectrograms_data'
    
    # Classes and Mapping
   
    
    # CLASSES =  ["bansuri", "deuda","jhora","kartik", "lahare"] 

    CLASSES = [
        "Asare", "Astamatrika", "basuri", "Bhairab_dance", "Bhote_lhomi",
        "Bhume_nach", "Chaliya", "Charya", "Chyabrung", "Damaha", 
        "deuda", "Dhami", "Dhime", "Ghatu", "Gunla", "jhapre", "jhijya", 
        "jhora", "kartik", "Kaura", "khukuri", "Kumari", "lahare", "Lakhe", 
        "lathi", "Mahakali", "Maruni", "mayur", "Monkey_dance", "murchunga", 
        "Palam", "Panchebaja", "rajbanshi", "Sakela", "Sarangi", "saraya", 
        "Sherpa", "Singura", "Sorathi", "Tharu"
    ]

    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
    IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
    
    # Training Hyperparameters
    BATCH_SIZE = 16 
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    NUM_CLASSES = len(CLASSES)
    # PATCH_SIZE = 16
    # EMBEDDING_DIM = 128
    
    # Model Configuration
    INPUT_CHANNELS = 1 # RGB spectrograms
    SPECTROGRAM_SIZE = (224, 224)  # Resize dimension
    # SPECTROGRAM_HEIGHT = 224
    # SPECTROGRAM_WIDTH = 224
    
    # Paths
    EXPERIMENT_NAME = 'spectrogram_classification'
    OUTPUT_DIR = os.path.join('runs', EXPERIMENT_NAME)
    
    # Logging and Checkpointing
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create necessary directories
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

# Initialize directories when the config is imported
Config.create_directories()