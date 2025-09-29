"""
Data preprocessing utilities for the chatbot
"""

import re
import unicodedata
import pickle
import torch
from collections import Counter
from config.config import *

class Vocabulary:
    """Vocabulary class for managing word-to-index mappings"""
    
    def __init__(self):
        self.word2index = {"SOS": SOS_TOKEN, "EOS": EOS_TOKEN, "PAD": PAD_TOKEN}
        self.word2count = {}
        self.index2word = {SOS_TOKEN: "SOS", EOS_TOKEN: "EOS", PAD_TOKEN: "PAD"}
        self.num_words = 3  # Count SOS, EOS, PAD
        
    def add_sentence(self, sentence):
        """Add all words in a sentence to vocabulary"""
        for word in sentence.split(' '):
            self.add_word(word)
    
    def add_word(self, word):
        """Add a word to vocabulary"""
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

def unicode_to_ascii(s):
    """Convert unicode string to ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    """Normalize string by converting to lowercase and removing punctuation"""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def read_conversations(file_path):
    """Read conversation pairs from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        
        pairs = []
        for i in range(0, len(lines)-1, 2):
            if i+1 < len(lines):
                input_line = normalize_string(lines[i])
                target_line = normalize_string(lines[i+1])
                if len(input_line) >= MIN_LENGTH and len(target_line) >= MIN_LENGTH:
                    pairs.append([input_line, target_line])
        
        return pairs
    except FileNotFoundError:
        print(f"Conversation file not found at {file_path}")
        return []

def filter_pairs(pairs, max_length=MAX_LENGTH):
    """Filter pairs based on length constraints"""
    return [pair for pair in pairs if 
            len(pair[0].split(' ')) < max_length and 
            len(pair[1].split(' ')) < max_length]

def prepare_data(file_path):
    """Prepare training data and vocabulary"""
    print("Reading conversation pairs...")
    pairs = read_conversations(file_path)
    
    if not pairs:
        # Create some default conversation pairs if no file exists
        pairs = [
            ["hello", "hi there"],
            ["how are you", "i am fine thank you"],
            ["what is your name", "i am a chatbot"],
            ["goodbye", "see you later"],
            ["thanks", "you are welcome"],
            ["help", "how can i assist you"],
            ["yes", "great"],
            ["no", "okay"],
            ["maybe", "i understand"],
            ["good morning", "good morning to you too"]
        ]
        print(f"Using {len(pairs)} default conversation pairs")
    
    print(f"Read {len(pairs)} conversation pairs")
    
    pairs = filter_pairs(pairs)
    print(f"Filtered to {len(pairs)} pairs")
    
    # Create vocabulary
    vocab = Vocabulary()
    for pair in pairs:
        vocab.add_sentence(pair[0])
        vocab.add_sentence(pair[1])
    
    print(f"Vocabulary size: {vocab.num_words}")
    
    return vocab, pairs

def indexes_from_sentence(vocab, sentence):
    """Convert sentence to list of word indexes"""
    return [vocab.word2index[word] for word in sentence.split(' ') if word in vocab.word2index]

def tensor_from_sentence(vocab, sentence, device):
    """Convert sentence to tensor"""
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(vocab, pair, device):
    """Convert input/target pair to tensors"""
    input_tensor = tensor_from_sentence(vocab, pair[0], device)
    target_tensor = tensor_from_sentence(vocab, pair[1], device)
    return input_tensor, target_tensor

def save_vocabulary(vocab, path):
    """Save vocabulary to file"""
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocabulary(path):
    """Load vocabulary from file"""
    with open(path, 'rb') as f:
        return pickle.load(f)