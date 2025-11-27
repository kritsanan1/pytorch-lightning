# Configuration settings for the PyTorch Lightning project

class Config:
    # Model settings
    INPUT_DIM = 784
    HIDDEN_DIM = 128
    OUTPUT_DIM = 10
    LEARNING_RATE = 1e-3
    
    # Training settings
    BATCH_SIZE = 64
    MAX_EPOCHS = 10
    NUM_WORKERS = 4
    
    # Data settings
    DATA_DIR = './data'
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = 'lightning_logs'
    
    # Device settings
    ACCELERATOR = 'auto'  # 'cpu', 'gpu', 'tpu', 'auto'
    DEVICES = 'auto'