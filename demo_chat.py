#!/usr/bin/env python3
"""
Demo chat interface for the AI ChatBot
This shows a working chatbot using rule-based responses based on training data
"""

import random
from src.data_utils import prepare_data, normalize_string
from config.config import *

class DemoChatBot:
    """Demo ChatBot that uses pattern matching with training data"""
    
    def __init__(self):
        print('Loading conversation data...')
        self.vocab, self.pairs = prepare_data(DATA_PATH)
        
        # Create response mappings from training data
        self.responses = {}
        for pair in self.pairs:
            input_text = pair[0]
            output_text = pair[1]
            
            # Create mapping from input to output
            self.responses[input_text] = output_text
            
            # Also create mappings for individual words
            for word in input_text.split():
                if word not in self.responses:
                    self.responses[word] = output_text
        
        print(f'Loaded {len(self.pairs)} conversation pairs')
        print(f'Created {len(self.responses)} response patterns')
        
        # Default responses
        self.default_responses = [
            "That's interesting, tell me more.",
            "I see, can you elaborate?",
            "How does that make you feel?",
            "That sounds important to you.",
            "I understand what you mean.",
            "Can you tell me more about that?",
            "That's a good point.",
            "I'm listening, please continue."
        ]
    
    def evaluate(self, sentence):
        """Generate response based on input"""
        sentence = normalize_string(sentence)
        
        # First try exact match
        if sentence in self.responses:
            return self.responses[sentence]
        
        # Try partial matches
        words = sentence.split()
        for word in words:
            if word in self.responses:
                return self.responses[word]
        
        # Try to find responses that contain similar words
        for response_key in self.responses:
            response_words = response_key.split()
            for word in words:
                if word in response_words:
                    return self.responses[response_key]
        
        # Fallback to default response
        return random.choice(self.default_responses)
    
    def chat(self):
        """Interactive chat session"""
        print("=" * 60)
        print("AI ChatBot Demo - PyTorch Implementation")
        print("This demo shows the chatbot concept using training data")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("=" * 60)
        
        # Show some example responses first
        print("\n--- Example Responses ---")
        examples = ['hello', 'how are you', 'what is your name', 'thanks', 'help']
        for example in examples:
            if example in [pair[0] for pair in self.pairs]:
                response = self.evaluate(example)
                print(f"Example - You: {example}")
                print(f"Example - Bot: {response}")
        
        print("\n--- Start Chatting ---")
        
        while True:
            try:
                # Get input from user
                user_input = input('\nYou: ').strip()
                
                # Check for quit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("ChatBot: Goodbye! Thanks for chatting with me!")
                    break
                    
                # Skip empty inputs
                if not user_input:
                    print("ChatBot: I'm here, please say something.")
                    continue
                    
                # Generate response
                response = self.evaluate(user_input)
                print(f'ChatBot: {response}')
                
            except KeyboardInterrupt:
                print("\nChatBot: Goodbye! Thanks for chatting with me!")
                break
            except Exception as e:
                print(f"ChatBot: Sorry, I had a technical issue: {e}")

def show_training_data():
    """Show available training conversation pairs"""
    print("=" * 60)
    print("Available Training Conversations")
    print("=" * 60)
    
    vocab, pairs = prepare_data(DATA_PATH)
    
    print(f"Total conversation pairs: {len(pairs)}")
    print(f"Vocabulary size: {vocab.num_words}")
    print("\nSample conversations:")
    
    for i, pair in enumerate(pairs[:15]):  # Show first 15 pairs
        print(f"{i+1:2d}. You: {pair[0]}")
        print(f"    Bot: {pair[1]}")
        print()

def main():
    """Main function"""
    print("=== AI ChatBot Demo ===")
    print("\nOptions:")
    print("1. Chat with the bot")
    print("2. View training data")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '2':
        show_training_data()
    else:
        # Initialize and start chatbot
        chatbot = DemoChatBot()
        chatbot.chat()

if __name__ == '__main__':
    main()