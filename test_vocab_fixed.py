#!/usr/bin/env python3
"""Test the corrected vocabulary loading"""

import json

def test_corrected_vocab():
    print("Testing corrected vocabulary loading...")
    
    # Load the correct vocabulary files
    with open("vocab_src.json", 'r') as f:
        vocab_src_data = json.load(f)
    with open("vocab_tgt.json", 'r') as f:
        vocab_tgt_data = json.load(f)
    
    print(f"Source vocab size: {len(vocab_src_data['stoi'])}")
    print(f"Target vocab size: {len(vocab_tgt_data['stoi'])}")
    
    # Create SimpleVocab class
    class SimpleVocab:
        def __init__(self, vocab_data):
            self.stoi = vocab_data['stoi']
            self.itos = vocab_data['itos']
            
        def __getitem__(self, token):
            return self.stoi.get(token, self.stoi.get('<unk>', 0))
        
        def __len__(self):
            return len(self.stoi)
        
        def lookup_tokens(self, indices):
            result = []
            for idx in indices:
                if 0 <= idx < len(self.itos):
                    result.append(self.itos[idx])
                else:
                    result.append('<unk>')
            return result
    
    # Test vocabularies
    vocab_src = SimpleVocab(vocab_src_data)
    vocab_tgt = SimpleVocab(vocab_tgt_data)
    
    print(f"\nTesting source vocabulary:")
    test_words = ['die', 'katze', 'sitzt', 'unknown_word']
    for word in test_words:
        idx = vocab_src[word]
        print(f"'{word}' -> {idx}")
    
    print(f"\nTesting target vocabulary:")
    test_words = ['the', 'cat', 'sits', 'unknown_word']
    for word in test_words:
        idx = vocab_tgt[word]
        print(f"'{word}' -> {idx}")
    
    # Test reverse lookup
    print(f"\nTesting reverse lookup:")
    test_indices = [0, 1, 2, 3, 4, 5]
    src_tokens = vocab_src.lookup_tokens(test_indices)
    tgt_tokens = vocab_tgt.lookup_tokens(test_indices)
    print(f"Source indices {test_indices} -> {src_tokens}")
    print(f"Target indices {test_indices} -> {tgt_tokens}")

if __name__ == "__main__":
    test_corrected_vocab()
