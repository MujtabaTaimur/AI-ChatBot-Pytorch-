#!/usr/bin/env python3
"""
Simple chat interface that works with the trained model
"""

import torch
import torch.nn as nn
import random
import os
from src.model import EncoderRNN, LuongAttnDecoderRNN
from src.data_utils import normalize_string, load_vocabulary
from config.config import *

class SimpleChatBot:
    """Simple ChatBot class for inference"""
    
    def __init__(self, model_path=MODEL_PATH, vocab_path=VOCAB_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
        print(f'Using device: {self.device}')
        
        self.vocab = None
        self.encoder = None
        self.decoder = None
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
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Extract model configuration
                config = checkpoint.get('model_config', {})
                hidden_size = config.get('hidden_size', HIDDEN_SIZE)
                encoder_n_layers = config.get('encoder_n_layers', NUM_LAYERS)
                decoder_n_layers = config.get('decoder_n_layers', NUM_LAYERS)
                attn_model = config.get('attn_model', 'dot')
                embedding_size = config.get('embedding_size', EMBEDDING_SIZE)
                
                # Initialize models
                self.encoder = EncoderRNN(self.vocab.num_words, hidden_size, encoder_n_layers)
                self.decoder = LuongAttnDecoderRNN(attn_model, embedding_size, hidden_size, 
                                                 self.vocab.num_words, decoder_n_layers)
                
                # Load model states
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                
                # Move to device and set to eval mode
                self.encoder = self.encoder.to(self.device)
                self.decoder = self.decoder.to(self.device)
                self.encoder.eval()
                self.decoder.eval()
                
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
        if not self.vocab or not self.encoder or not self.decoder:
            return "Model not loaded. Please train the model first."
            
        # Normalize sentence
        sentence = normalize_string(sentence)
        words = sentence.split()
        
        # Convert to word indexes
        indexes = []
        for word in words:
            if word in self.vocab.word2index:
                indexes.append(self.vocab.word2index[word])
            # Skip unknown words instead of failing
        
        if not indexes:
            return "I don't understand those words."
            
        indexes.append(EOS_TOKEN)
        
        # Create input tensor
        input_tensor = torch.LongTensor(indexes).view(-1, 1).to(self.device)
        input_length = torch.LongTensor([len(indexes)]).to('cpu')
        
        # Encode
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_length)
            
            # Initialize decoder
            decoder_input = torch.LongTensor([[SOS_TOKEN]]).to(self.device)
            decoder_hidden = encoder_hidden[:self.decoder.num_layers]
            
            # Generate response
            decoded_words = []
            
            for di in range(max_length):
                # Simple decoder forward pass
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs.unsqueeze(0)
                )
                
                # Get most likely word
                topv, topi = decoder_output.topk(1)
                if topi.item() == EOS_TOKEN:
                    break
                elif topi.item() in self.vocab.index2word:
                    word = self.vocab.index2word[topi.item()]
                    if word not in ['SOS', 'PAD']:
                        decoded_words.append(word)
                
                # Use output as next input
                decoder_input = topi.detach()
                
        return ' '.join(decoded_words) if decoded_words else "I don't know what to say."
    
    def chat(self):
        """Interactive chat session"""
        print("=" * 50)
        print("Simple AI ChatBot")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                # Get input from user
                user_input = input('\nYou: ').strip()
                
                # Check for quit command
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("ChatBot: Goodbye! Thanks for chatting!")
                    break
                    
                # Skip empty inputs
                if not user_input:
                    continue
                    
                # Generate response
                response = self.evaluate(user_input)
                print(f'ChatBot: {response}')
                
            except KeyboardInterrupt:
                print("\nChatBot: Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function"""
    import os
    
    print("=== Simple AI ChatBot ===")
    
    # Initialize chatbot
    chatbot = SimpleChatBot()
    
    # Check if model is loaded
    if not chatbot.vocab or not chatbot.encoder or not chatbot.decoder:
        print("\nNo trained model found. Please run training first:")
        print("python quick_train.py")
        return
        
    # Test with some examples first
    print("\n--- Testing the bot ---")
    test_cases = ['hello', 'how are you', 'thanks']
    for test in test_cases:
        response = chatbot.evaluate(test)
        print(f"Test - You: {test}")
        print(f"Test - Bot: {response}")
    
    print("\n--- Ready for chat ---")
    # Start interactive chat
    chatbot.chat()

if __name__ == '__main__':
    main()