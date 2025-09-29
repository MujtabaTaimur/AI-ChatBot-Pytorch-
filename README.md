# 🤖 AI ChatBot (PyTorch)

A simple intent-based chatbot built with **PyTorch** and **NLTK**.  
This project demonstrates natural language understanding (NLU) using a neural network with bag-of-words features. You can customize intents and even map them to functions (e.g., fetching stocks).

---

## 🚀 Features
- Trainable PyTorch model (fully connected NN).
- Tokenization + lemmatization with NLTK.
- Bag-of-words representation for text.
- Loads training data from `intents.json`.
- Function mappings (run Python functions when specific intents are detected).
- Saves/loads trained model for inference.

---

## 📂 Project Structure
- main.py # Main chatbot assistant code (training + inference)
- intents.json # Training data (intents, patterns, responses)
- README.md
