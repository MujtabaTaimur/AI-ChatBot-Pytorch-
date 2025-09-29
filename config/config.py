"""
Configuration settings for the AI ChatBot
"""

# Model hyperparameters
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_LENGTH = 50
MIN_LENGTH = 3

# Training parameters  
NUM_EPOCHS = 100
TEACHER_FORCING_RATIO = 0.5
CLIP_GRAD = 50.0
PRINT_EVERY = 100
SAVE_EVERY = 1000

# Data parameters
SOS_TOKEN = 0  # Start of sentence token
EOS_TOKEN = 1  # End of sentence token
PAD_TOKEN = 2  # Padding token

# File paths
MODEL_PATH = "models/chatbot_model.pth"
VOCAB_PATH = "models/vocab.pkl"
DATA_PATH = "data/conversations.txt"

# Device configuration
USE_CUDA = True  # Set to False if you don't have CUDA