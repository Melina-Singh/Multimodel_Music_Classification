
import os

class Config:
    # Dataset Configuration
    DATA_DIR = 'b_video_data/'
    
    # Classes and Mapping
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
    BATCH_SIZE = 8 
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    NUM_CLASSES = len(CLASSES)
    
    # Model Configuration
    INPUT_CHANNELS = 3
    FRAMES = 30
    
    # Paths
    EXPERIMENT_NAME = 'instrument_classification'
    OUTPUT_DIR = os.path.join('runs', EXPERIMENT_NAME)
    
    # Logging and Checkpointing
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    
    # Create necessary directories
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)