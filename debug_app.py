#!/usr/bin/env python3
"""Test script to debug vocabulary and model loading"""

import json
import torch
import os

def test_vocab_loading():
    """Test loading vocabularies"""
    print("Testing vocabulary loading...")
    
    try:
        # Load vocabularies
        with open("vocab_src_wmt14.json", 'r') as f:
            vocab_src_data = json.load(f)
        with open("vocab_tgt_wmt14.json", 'r') as f:
            vocab_tgt_data = json.load(f)
        
        print(f"Source vocab size: {len(vocab_src_data['stoi'])}")
        print(f"Target vocab size: {len(vocab_tgt_data['stoi'])}")
        
        # Test a few lookups
        print(f"Source 'die' -> {vocab_src_data['stoi'].get('die', 'NOT_FOUND')}")
        print(f"Target 'the' -> {vocab_tgt_data['stoi'].get('the', 'NOT_FOUND')}")
        
        return vocab_src_data, vocab_tgt_data, len(vocab_src_data['stoi']), len(vocab_tgt_data['stoi'])
        
    except Exception as e:
        print(f"Error loading vocabularies: {e}")
        return None, None, None, None

def test_model_loading():
    """Test loading the model"""
    print("\nTesting model loading...")
    
    try:
        # Check if model file exists
        if not os.path.exists("transformer_best_model.pt"):
            print("Model file transformer_best_model.pt not found!")
            return None
            
        # Try to load the state dict
        state_dict = torch.load("transformer_best_model.pt", map_location='cpu')
        print(f"Model state dict loaded successfully")
        print(f"Number of parameters: {len(state_dict)}")
        
        # Print some parameter shapes to verify architecture
        for key in list(state_dict.keys())[:5]:
            print(f"  {key}: {state_dict[key].shape}")
        
        return state_dict
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_simple_vocab():
    """Test the SimpleVocab class"""
    print("\nTesting SimpleVocab class...")
    
    # Create simple vocabulary classes
    class SimpleVocab:
        def __init__(self, vocab_data):
            self.stoi = vocab_data['stoi']
            self.itos = vocab_data['itos']
            
        def __getitem__(self, token):
            return self.stoi.get(token, self.stoi.get('<unk>', 0))
        
        def __len__(self):
            return len(self.stoi)
        
        def lookup_tokens(self, indices):
            return [self.itos[str(idx)] if str(idx) in self.itos else '<unk>' for idx in indices]
    
    try:
        with open("vocab_src_wmt14.json", 'r') as f:
            vocab_src_data = json.load(f)
        
        vocab = SimpleVocab(vocab_src_data)
        
        # Test token lookup
        test_tokens = ['die', 'katze', 'sitzt', '<unknown_token>']
        for token in test_tokens:
            idx = vocab[token]
            print(f"'{token}' -> {idx}")
        
        # Test index lookup
        test_indices = [6, 7, 8, 0]  # Some common indices
        tokens = vocab.lookup_tokens(test_indices)
        print(f"Indices {test_indices} -> {tokens}")
        
        return True
        
    except Exception as e:
        print(f"Error testing SimpleVocab: {e}")
        return False

if __name__ == "__main__":
    print("=== Vocabulary and Model Loading Test ===")
    
    # Test vocabulary loading
    vocab_src_data, vocab_tgt_data, src_size, tgt_size = test_vocab_loading()
    
    # Test model loading
    state_dict = test_model_loading()
    
    # Test SimpleVocab class
    test_simple_vocab()
    
    print(f"\n=== Summary ===")
    print(f"Vocabularies loaded: {vocab_src_data is not None}")
    print(f"Model loaded: {state_dict is not None}")
    print(f"Source vocab size: {src_size}")
    print(f"Target vocab size: {tgt_size}")
