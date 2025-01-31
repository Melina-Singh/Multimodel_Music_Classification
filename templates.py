import os
from pathlib import Path

list_of_files = [

   ".github/workflows/.gitkeep",
   "src/data_loaders/audio_data_loader.py",
    "src/data_loaders/video_data_loader.py",
    "src/models/audio_model.py",  
    "src/models/video_model.py",  
    "src/trainers/audio_model_training.py",  
    "src/trainers/video_model_training.py",  
    "src/configs/A_config.py",
    "src/configs/V_config.py",
    "src/utils/utils.py",
    "src/logger/logging.py",
    "src/exception/exception.py",
    "src/aud_main.py",
    "src/vid_main.py",
    "app/app.py",
    "app/components/file_uploader.py",
    "app/components/display.py",
    "experiment/experiments.ipynb",
    "setup.py",
    "check.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        # logging.info(f"Creating directory: {filedir} for file: {filename}")


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass
