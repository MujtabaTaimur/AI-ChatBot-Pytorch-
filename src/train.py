"""
Training script for the ChatBot
"""

import torch
import torch.nn as nn
from torch import optim
import random
import os
import sys
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import EncoderRNN, LuongAttnDecoderRNN
from src.data_utils import prepare_data, tensors_from_pair, save_vocabulary
from config.config import *

def mask_nll_loss(inp, target, mask):
    """Calculate loss with masking for padded sequences"""
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(inp.device)
    return loss, n_total.item()

def train_iteration(input_variable, lengths, target_variable, mask, max_target_len,
                   encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, 
                   batch_size, clip, device, max_length=MAX_LENGTH):
    """Single training iteration"""
    
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    lengths = lengths.to('cpu')
    
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0
    
    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    
    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.num_layers]
    
    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
    
    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, n_total = mask_nll_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, n_total = mask_nll_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    
    # Perform backpropatation
    loss.backward()
    
    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return sum(print_losses) / n_totals

def batch_2_train_data(vocab, pair_batch, device):
    """Convert batch of conversation pairs to training data"""
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    
    inp, lengths = input_var(input_batch, vocab, device)
    output, mask, max_target_len = output_var(output_batch, vocab, device)
    return inp, lengths, output, mask, max_target_len

def input_var(l, vocab, device):
    """Create input variable from list of sentences"""
    indexes_batch = [indexes_from_sentence(vocab, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padded_seqs = torch.zeros(len(indexes_batch), max(lengths)).long()
    
    for i, seq in enumerate(indexes_batch):
        padded_seqs[i, :lengths[i]] = torch.LongTensor(seq)
        
    return padded_seqs, lengths

def output_var(l, vocab, device):
    """Create output variable from list of sentences"""
    indexes_batch = [indexes_from_sentence(vocab, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padded_seqs = torch.zeros(max_target_len, len(indexes_batch)).long()
    mask = torch.zeros(max_target_len, len(indexes_batch)).byte()
    
    for i, seq in enumerate(indexes_batch):
        for j, index in enumerate(seq):
            padded_seqs[j, i] = index
            mask[j, i] = 1
            
    return padded_seqs, mask, max_target_len

def indexes_from_sentence(vocab, sentence):
    """Convert sentence to list of word indexes"""
    return [vocab.word2index[word] for word in sentence.split(' ') if word in vocab.word2index] + [EOS_TOKEN]

def train_model(model_name, vocab, pairs, encoder, decoder, encoder_optimizer, 
               decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, 
               save_dir, n_iteration, batch_size, print_every, save_every, clip, 
               corpus_name, load_filename, device):
    """Main training function"""
    
    # Load batches for each iteration
    training_batches = [batch_2_train_data(vocab, [random.choice(pairs) for _ in range(batch_size)], device)
                       for _ in range(n_iteration)]
    
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    
    if load_filename:
        start_iteration = checkpoint['iteration'] + 1
    
    # Training loop
    print("Training...")
    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        
        # Run a training iteration with batch
        loss = train_iteration(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                              decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, device)
        print_loss += loss
        
        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
            
        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, 
                                   '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, HIDDEN_SIZE))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'vocab_dict': vocab.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

def main():
    """Main training function"""
    # Configure models
    model_name = 'chatbot_model'
    attn_model = 'dot'
    hidden_size = HIDDEN_SIZE
    encoder_n_layers = NUM_LAYERS
    decoder_n_layers = NUM_LAYERS
    dropout = DROPOUT
    batch_size = BATCH_SIZE
    
    # Configure training/optimization
    clip = CLIP_GRAD
    teacher_forcing_ratio = TEACHER_FORCING_RATIO
    learning_rate = LEARNING_RATE
    decoder_learning_ratio = 5.0
    n_iteration = 4000
    print_every = 100
    save_every = 500
    
    # Set checkpoint to load from; set to None if starting from scratch
    load_filename = None
    checkpoint_iter = 4000
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    print(f'Using device: {device}')
    
    # Load/Assemble vocab and pairs
    corpus_name = "chatbot"
    vocab, pairs = prepare_data(DATA_PATH)
    
    # Print some pairs to validate
    print("\nSample pairs:")
    for i in range(min(10, len(pairs))):
        print(pairs[i])
    
    # Load model if a load_filename is provided
    if load_filename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(load_filename, map_location=device)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        vocab.__dict__ = checkpoint['vocab_dict']
    
    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(vocab.num_words, hidden_size)
    if load_filename:
        embedding.load_state_dict(embedding_sd)
        
    # Initialize encoder & decoder models
    encoder = EncoderRNN(vocab.num_words, hidden_size, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, EMBEDDING_SIZE, hidden_size, vocab.num_words, decoder_n_layers, dropout)
    
    if load_filename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    
    # Configure training/optimization
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    
    if load_filename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)
    
    # Run training iterations
    print("Starting Training!")
    train_model(model_name, vocab, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, "models", n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, load_filename, device)
    
    # Save final model
    print("Saving final model...")
    os.makedirs("models", exist_ok=True)
    
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'vocab': vocab,
        'embedding': embedding.state_dict(),
        'model_config': {
            'hidden_size': hidden_size,
            'encoder_n_layers': encoder_n_layers,
            'decoder_n_layers': decoder_n_layers,
            'attn_model': attn_model,
            'embedding_size': EMBEDDING_SIZE
        }
    }, MODEL_PATH)
    
    # Save vocabulary
    save_vocabulary(vocab, VOCAB_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vocabulary saved to {VOCAB_PATH}")

if __name__ == '__main__':
    main()