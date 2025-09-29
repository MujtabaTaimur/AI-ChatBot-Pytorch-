# AI ChatBot - PyTorch

A sophisticated AI chatbot built with PyTorch using sequence-to-sequence learning with attention mechanism. This chatbot can engage in natural conversations and learn from training data.

## Features

- **Sequence-to-Sequence Architecture**: Uses encoder-decoder model with LSTM/GRU cells
- **Attention Mechanism**: Luong attention for better context understanding
- **Flexible Training**: Configurable hyperparameters and training settings
- **Interactive Chat**: Real-time conversation interface
- **Extensible**: Easy to add new training data and modify behavior

## Architecture

The chatbot uses a sequence-to-sequence model with the following components:

1. **Encoder**: Processes input sentences and creates context vectors
2. **Decoder**: Generates response tokens using attention mechanism
3. **Attention**: Helps the model focus on relevant parts of the input
4. **Embedding Layer**: Converts words to dense vector representations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MujtabaTaimur/AI-ChatBot-Pytorch-.git
cd AI-ChatBot-Pytorch-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training the Model

Train the chatbot with the provided conversation data:

```bash
python main.py train
```

This will:
- Load conversation pairs from `data/conversations.txt`
- Create vocabulary from the training data
- Train the encoder-decoder model
- Save the trained model to `models/chatbot_model.pth`

### Starting a Chat Session

Once trained, start chatting with the bot:

```bash
python main.py chat
```

Example conversation:
```
You: hello
ChatBot: hi there how are you

You: what is your name
ChatBot: i am an ai chatbot here to help you

You: quit
ChatBot: Goodbye!
```

## Configuration

Model hyperparameters can be adjusted in `config/config.py`:

```python
# Model hyperparameters
HIDDEN_SIZE = 256        # Hidden layer size
EMBEDDING_SIZE = 128     # Word embedding dimension
NUM_LAYERS = 2          # Number of RNN layers
DROPOUT = 0.1           # Dropout probability
LEARNING_RATE = 0.001   # Learning rate
BATCH_SIZE = 32         # Training batch size
MAX_LENGTH = 50         # Maximum sentence length
```

## Project Structure

```
AI-ChatBot-Pytorch-/
├── config/
│   ├── __init__.py
│   └── config.py           # Configuration settings
├── data/
│   └── conversations.txt   # Training conversation pairs
├── models/                 # Saved models directory
├── src/
│   ├── __init__.py
│   ├── data_utils.py      # Data preprocessing utilities
│   ├── model.py           # Neural network models
│   ├── train.py           # Training script
│   └── chat.py            # Chat interface
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Training Data Format

The training data should be in `data/conversations.txt` with alternating lines of input and response:

```
hello
hi there how are you
what is your name
i am an ai chatbot
goodbye
see you later
```

## Advanced Usage

### Custom Training Data

Replace `data/conversations.txt` with your own conversation pairs:

1. Each input-response pair should be on alternating lines
2. Keep responses conversational and natural
3. Ensure good coverage of topics you want the bot to handle

### Modifying Model Architecture

You can modify the model in `src/model.py`:

- Change attention mechanism (dot, general, concat)
- Adjust layer sizes and depths
- Add regularization techniques
- Implement different RNN variants

### Training Parameters

Adjust training in `config/config.py`:

- `NUM_EPOCHS`: Number of training epochs
- `TEACHER_FORCING_RATIO`: Probability of using teacher forcing
- `CLIP_GRAD`: Gradient clipping threshold
- `LEARNING_RATE`: Optimizer learning rate

## Model Performance

The model performance depends on:

1. **Training Data Quality**: Better conversation pairs lead to better responses
2. **Model Size**: Larger hidden sizes can capture more complex patterns
3. **Training Time**: More epochs generally improve performance
4. **Vocabulary Size**: Larger vocabulary allows more diverse responses

## Troubleshooting

### CUDA Issues
If you encounter CUDA problems, set `USE_CUDA = False` in `config/config.py`.

### Memory Issues
- Reduce `BATCH_SIZE` in config
- Decrease `HIDDEN_SIZE` or `MAX_LENGTH`
- Use gradient checkpointing for very large models

### Poor Responses
- Add more diverse training data
- Increase training time
- Adjust attention mechanism
- Fine-tune hyperparameters

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- NLTK
- tqdm
- scikit-learn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Feel free to use and modify as needed.

## Acknowledgments

- Based on PyTorch seq2seq tutorials
- Inspired by neural machine translation research
- Uses Luong attention mechanism