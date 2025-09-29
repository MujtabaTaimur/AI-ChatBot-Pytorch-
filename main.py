#!/usr/bin/env python3
"""
Main script for the AI ChatBot
Usage: 
    python main.py train    # Train the model
    python main.py chat     # Start chat interface
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='AI ChatBot with PyTorch')
    parser.add_argument('mode', choices=['train', 'chat'], 
                       help='Mode: train the model or start chat interface')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training...")
        from src.train import main as train_main
        train_main()
        
    elif args.mode == 'chat':
        print("Starting chat interface...")
        from src.chat import main as chat_main  
        chat_main()

if __name__ == '__main__':
    main()