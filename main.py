#!/usr/bin/env python3
"""
Main script for the AI ChatBot
Usage: 
    python main.py train     # Train the neural network model
    python main.py chat      # Start neural network chat interface  
    python main.py demo      # Start demo chat (pattern-based)
    python main.py quick     # Quick training session
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='AI ChatBot with PyTorch')
    parser.add_argument('mode', choices=['train', 'chat', 'demo', 'quick'], 
                       help='Mode: train/chat with neural model, demo chat, or quick training')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting full neural network training...")
        from src.train import main as train_main
        train_main()
        
    elif args.mode == 'chat':
        print("Starting neural network chat interface...")
        from src.chat import main as chat_main  
        chat_main()
        
    elif args.mode == 'demo':
        print("Starting demo chat interface...")
        from demo_chat import main as demo_main
        demo_main()
        
    elif args.mode == 'quick':
        print("Starting quick training session...")
        import quick_train
        quick_train.quick_train()

if __name__ == '__main__':
    main()