#!/usr/bin/env python3
"""
Quick training script for demonstration purposes
"""

import sys
import os
import torch
import torch.nn as nn
from torch import optim
import random
from tqdm import tqdm

# Import our modules
from src.model import EncoderRNN, LuongAttnDecoderRNN
from src.data_utils import prepare_data, save_vocabulary, tensors_from_pair
from config.config import *

def quick_train():
    """Quick training function with simplified parameters"""
    
    # Quick training configuration
    QUICK_HIDDEN_SIZE = 64
    QUICK_EMBEDDING_SIZE = 64
    QUICK_NUM_LAYERS = 1
    QUICK_NUM_ITERATIONS = 200
    QUICK_PRINT_EVERY = 50
    
    print('=== Quick Training Session ===')
    print('Loading data and preparing vocabulary...')
    
    # Load data
    vocab, pairs = prepare_data(DATA_PATH)
    print(f'Loaded {len(pairs)} conversation pairs')
    print(f'Vocabulary size: {vocab.num_words}')
    
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize models
    print('Initializing models...')
    attn_model = 'dot'
    encoder = EncoderRNN(vocab.num_words, QUICK_HIDDEN_SIZE, QUICK_NUM_LAYERS, DROPOUT)
    decoder = LuongAttnDecoderRNN(attn_model, QUICK_EMBEDDING_SIZE, QUICK_HIDDEN_SIZE, 
                                vocab.num_words, QUICK_NUM_LAYERS, DROPOUT)
    
    # Move to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Initialize optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE * 2.0)
    
    print('Models initialized successfully!')
    
    # Simple training function
    def train_iteration(input_var, target_var):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        input_length = input_var.size(0)
        target_length = target_var.size(0)
        
        loss = 0
        
        # Encoder forward pass - simplified
        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        encoder_hidden = torch.zeros(QUICK_NUM_LAYERS, 1, encoder.hidden_size, device=device)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_var[ei].view(1, 1), 
                                                   torch.tensor([1]), encoder_hidden)
        
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO
        
        if use_teacher_forcing:
            for di in range(target_length):
                # Create encoder outputs for attention (simplified)
                encoder_outputs_dummy = torch.zeros(1, 1, QUICK_HIDDEN_SIZE, device=device)
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs_dummy)
                loss += nn.NLLLoss()(torch.log(decoder_output + 1e-10), target_var[di])
                decoder_input = target_var[di].view(1, 1)
        else:
            for di in range(target_length):
                encoder_outputs_dummy = torch.zeros(1, 1, QUICK_HIDDEN_SIZE, device=device)
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs_dummy)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()
                
                loss += nn.NLLLoss()(torch.log(decoder_output + 1e-10), target_var[di])
                
                if decoder_input.item() == EOS_TOKEN:
                    break
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP_GRAD)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP_GRAD)
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        return loss.item() / target_length
    
    # Training loop
    print(f'Training for {QUICK_NUM_ITERATIONS} iterations...')
    print_loss_total = 0
    
    for iteration in tqdm(range(1, QUICK_NUM_ITERATIONS + 1)):
        # Get random training pair
        training_pair = random.choice(pairs)
        input_variable, target_variable = tensors_from_pair(vocab, training_pair, device)
        
        # Train iteration
        loss = train_iteration(input_variable, target_variable)
        print_loss_total += loss
        
        if iteration % QUICK_PRINT_EVERY == 0:
            print_loss_avg = print_loss_total / QUICK_PRINT_EVERY
            print(f'Iteration {iteration}/{QUICK_NUM_ITERATIONS}: Avg Loss = {print_loss_avg:.4f}')
            print_loss_total = 0
    
    print('Training completed!')
    
    # Create models directory and save the model
    print('Saving model...')
    os.makedirs('models', exist_ok=True)
    
    # Save model with proper structure
    model_data = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'vocab': vocab,
        'model_config': {
            'hidden_size': QUICK_HIDDEN_SIZE,
            'encoder_n_layers': QUICK_NUM_LAYERS,
            'decoder_n_layers': QUICK_NUM_LAYERS,
            'attn_model': attn_model,
            'embedding_size': QUICK_EMBEDDING_SIZE
        }
    }
    
    torch.save(model_data, MODEL_PATH)
    save_vocabulary(vocab, VOCAB_PATH)
    
    print(f'✓ Model saved to {MODEL_PATH}')
    print(f'✓ Vocabulary saved to {VOCAB_PATH}')
    print('\n=== Training Session Complete ===')
    print('You can now chat with the bot using: python main.py chat')

if __name__ == '__main__':
    quick_train()