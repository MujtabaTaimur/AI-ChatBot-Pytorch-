"""
Chat interface for the AI ChatBot
"""

import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
from src.data_utils import normalize_string, load_vocabulary
from config.config import *

class ChatBot:
    """ChatBot class for inference"""
    
    def __init__(self, model_path=MODEL_PATH, vocab_path=VOCAB_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
        print(f'Using device: {self.device}')
        
        # Load vocabulary and model
        self.vocab = None
        self.searcher = None
        self.load_model(model_path, vocab_path)
        
    def load_model(self, model_path, vocab_path):
        """Load trained model and vocabulary"""
        try:
            # Load vocabulary
            if os.path.exists(vocab_path):
                self.vocab = load_vocabulary(vocab_path)
                print(f"Loaded vocabulary with {self.vocab.num_words} words")
            else:
                print(f"Vocabulary file not found at {vocab_path}")
                return False
                
            # Load model
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Extract model configuration
                config = checkpoint.get('model_config', {})
                hidden_size = config.get('hidden_size', HIDDEN_SIZE)
                encoder_n_layers = config.get('encoder_n_layers', NUM_LAYERS)
                decoder_n_layers = config.get('decoder_n_layers', NUM_LAYERS)
                attn_model = config.get('attn_model', 'dot')
                embedding_size = config.get('embedding_size', EMBEDDING_SIZE)
                
                # Initialize models
                encoder = EncoderRNN(self.vocab.num_words, hidden_size, encoder_n_layers)
                decoder = LuongAttnDecoderRNN(attn_model, embedding_size, hidden_size, 
                                            self.vocab.num_words, decoder_n_layers)
                
                # Load model states
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                
                # Move to device
                encoder = encoder.to(self.device)
                decoder = decoder.to(self.device)
                
                # Set to evaluation mode
                encoder.eval()
                decoder.eval()
                
                # Initialize searcher module
                self.searcher = GreedySearchDecoder(encoder, decoder)
                
                print("Model loaded successfully!")
                return True
            else:
                print(f"Model file not found at {model_path}")
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate(self, sentence, max_length=MAX_LENGTH):
        """Evaluate input sentence and generate response"""
        if not self.vocab or not self.searcher:
            return "Model not loaded. Please train the model first."
            
        # Normalize sentence
        sentence = normalize_string(sentence)
        
        # Convert to word indexes
        indexes_batch = [self.indexes_from_sentence(sentence)]
        
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        
        # Use appropriate device
        input_batch = input_batch.to(self.device)
        lengths = lengths.to('cpu')
        
        # Decode sentence with searcher
        tokens, scores = self.searcher(input_batch, lengths, max_length)
        
        # Convert indexes to words
        decoded_words = [self.vocab.index2word[token.item()] for token in tokens]
        
        # Filter out special tokens and return response
        response_words = []
        for word in decoded_words:
            if word == 'EOS':
                break
            elif word not in ['SOS', 'PAD']:
                response_words.append(word)
                
        return ' '.join(response_words) if response_words else "I don't understand."
    
    def indexes_from_sentence(self, sentence):
        """Convert sentence to list of word indexes"""
        return [self.vocab.word2index.get(word, self.vocab.word2index.get('UNK', 0)) 
                for word in sentence.split(' ')] + [EOS_TOKEN]
    
    def chat(self):
        """Interactive chat session"""
        print("ChatBot: Hello! I'm your AI assistant. Type 'quit' to exit.")
        
        while True:
            try:
                # Get input from user
                input_sentence = input('You: ').strip()
                
                # Check for quit command
                if input_sentence.lower() in ['quit', 'exit', 'bye']:
                    print("ChatBot: Goodbye!")
                    break
                    
                # Skip empty inputs
                if not input_sentence:
                    continue
                    
                # Generate response
                response = self.evaluate(input_sentence)
                print(f'ChatBot: {response}')
                
            except KeyboardInterrupt:
                print("\nChatBot: Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function for chat interface"""
    print("=== AI ChatBot ===")
    
    # Initialize chatbot
    chatbot = ChatBot()
    
    # Check if model is loaded
    if chatbot.vocab is None or chatbot.searcher is None:
        print("\nNo trained model found. Please run training first:")
        print("python src/train.py")
        return
        
    # Start chat session
    chatbot.chat()

if __name__ == '__main__':
    main()