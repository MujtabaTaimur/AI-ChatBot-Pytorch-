"""
Neural Network Models for the ChatBot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import *

class EncoderRNN(nn.Module):
    """Encoder RNN for sequence-to-sequence model"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, 
                         dropout=(0 if num_layers == 1 else dropout), 
                         batch_first=True)
        
    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, 
                                                  batch_first=True, 
                                                  enforce_sorted=False)
        
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden

class Attn(nn.Module):
    """Attention mechanism"""
    
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
            
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)
    
    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)
    
    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), 
                                     encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)
    
    def forward(self, hidden, encoder_outputs):
        # Calculate attention weights
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
            
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        
        # Return the softmax normalized probability scores
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    """Luong attention-based decoder RNN"""
    
    def __init__(self, attn_model, embedding_size, hidden_size, output_size, 
                 num_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        
        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, 
                         dropout=(0 if num_layers == 1 else dropout), 
                         batch_first=True)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.attn = Attn(attn_model, hidden_size)
        
    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        # Calculate attention weights from current RNN output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        
        # Concatenate weighted context vector and RNN output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        
        # Return output and final hidden state
        return output, hidden

class GreedySearchDecoder(nn.Module):
    """Greedy search decoder for inference"""
    
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        # Prepare encoder's final hidden layer to be first hidden input to decoder
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        
        # Initialize decoder input with SOS_TOKEN
        decoder_input = torch.ones(1, 1, device=input_seq.device, dtype=torch.long) * SOS_TOKEN
        
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=input_seq.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=input_seq.device)
        
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
            
        # Return collections of word tokens and scores
        return all_tokens, all_scores