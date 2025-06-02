#!/usr/bin/env python3
"""Test loading the recent M4-optimized model with WMT14 vocabularies"""

import json
import torch
from model import Transformer

def test_recent_model():
    print("Testing recent M4-optimized model...")
    
    # Load WMT14 vocabularies
    try:
        with open("vocab_src_wmt14.json", 'r') as f:
            vocab_src_data = json.load(f)
        with open("vocab_tgt_wmt14.json", 'r') as f:
            vocab_tgt_data = json.load(f)
            
        src_vocab_size = len(vocab_src_data['stoi'])
        tgt_vocab_size = len(vocab_tgt_data['stoi'])
        
        print(f"✅ Vocabularies loaded:")
        print(f"   Source vocab size: {src_vocab_size}")
        print(f"   Target vocab size: {tgt_vocab_size}")
        
    except Exception as e:
        print(f"❌ Error loading vocabularies: {e}")
        return False
    
    # Create model with M4-optimized configuration
    config = {
        "d_model": 256,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3, 
        "num_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "max_seq_len": 5000  # Fixed: match the saved model
    }
    
    try:
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config["d_model"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            max_seq_len=config["max_seq_len"]
        )
        
        print(f"✅ Model created successfully with configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False
    
    # Load the recent model state
    try:
        state_dict = torch.load("transformer_model_wmt14_epoch_3.pth", map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✅ Model state loaded successfully from transformer_model_wmt14_epoch_3.pth")
        
    except Exception as e:
        print(f"❌ Error loading model state: {e}")
        return False
    
    print(f"\n🎉 Success! Recent M4-optimized model is ready for inference.")
    return True

if __name__ == "__main__":
    test_recent_model()
