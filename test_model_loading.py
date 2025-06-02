#!/usr/bin/env python3

import torch
import json
from model import Transformer

# Load config
with open('config_local.json', 'r') as f:
    config = json.load(f)

print("Configuration loaded:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Load vocabularies
with open('vocab_src_wmt14.json', 'r') as f:
    vocab_src_data = json.load(f)
with open('vocab_tgt_wmt14.json', 'r') as f:
    vocab_tgt_data = json.load(f)

src_vocab_size = len(vocab_src_data['stoi'])
tgt_vocab_size = len(vocab_tgt_data['stoi'])

print(f"\nVocabulary sizes:")
print(f"  Source: {src_vocab_size}")
print(f"  Target: {tgt_vocab_size}")

# Create model
print(f"\nCreating model with:")
print(f"  d_model: {config['d_model']}")
print(f"  d_ff: {config['d_ff']}")
print(f"  max_seq_len: {config['max_seq_len']}")
print(f"  num_encoder_layers: {config['num_encoder_layers']}")
print(f"  num_decoder_layers: {config['num_decoder_layers']}")

try:
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    )
    print("\n✓ Model created successfully!")
    
    # Try to load the state dict
    print("\nLoading saved model...")
    state_dict = torch.load('transformer_model_wmt14_epoch_3.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    print("✓ Model state loaded successfully!")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Positional encoding shape: {model.positional_encoding.pe.shape}")
    print(f"  Feed-forward layer 1 shape: {model.encoder.layers[0].feed_forward.linear1.weight.shape}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
