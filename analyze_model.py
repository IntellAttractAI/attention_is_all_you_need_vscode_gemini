#!/usr/bin/env python3
"""Analyze the saved model to extract correct parameters"""

import torch

def analyze_model():
    state_dict = torch.load("transformer_best_model.pt", map_location='cpu')
    
    print("=== Model Architecture Analysis ===")
    
    # Extract vocabulary sizes from embedding layers
    src_vocab_size = state_dict['src_embedding.embedding.weight'].shape[0]
    tgt_vocab_size = state_dict['tgt_embedding.embedding.weight'].shape[0]
    d_model = state_dict['src_embedding.embedding.weight'].shape[1]
    
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")
    print(f"d_model: {d_model}")
    
    # Count encoder and decoder layers
    encoder_layers = len([k for k in state_dict.keys() if k.startswith('encoder.layers.')])
    decoder_layers = len([k for k in state_dict.keys() if k.startswith('decoder.layers.')])
    
    # Count attention heads by looking at key projection size
    if 'encoder.layers.0.self_attn.key_linear.weight' in state_dict:
        attn_dim = state_dict['encoder.layers.0.self_attn.key_linear.weight'].shape[0]
        num_heads = attn_dim // (d_model // 8) if d_model >= 64 else attn_dim // 64
        # More direct way: assume standard head_dim of 64
        num_heads = attn_dim // 64
    else:
        num_heads = "unknown"
    
    # Get feed-forward dimension
    if 'encoder.layers.0.feed_forward.fc1.weight' in state_dict:
        d_ff = state_dict['encoder.layers.0.feed_forward.fc1.weight'].shape[0]
    else:
        d_ff = "unknown"
    
    print(f"Encoder layers: {encoder_layers // 3}")  # Each layer has 3 components (self_attn, feed_forward, norm)
    print(f"Decoder layers: {decoder_layers // 4}")  # Each layer has 4 components
    print(f"Number of heads: {num_heads}")
    print(f"d_ff: {d_ff}")
    
    # Look for layer structure
    print("\n=== Layer Structure ===")
    layer_keys = [k for k in state_dict.keys() if 'layers.' in k]
    unique_layers = set()
    for key in layer_keys:
        parts = key.split('.')
        if 'encoder' in key:
            layer_num = parts[2] if len(parts) > 2 else 'unknown'
            unique_layers.add(f"encoder.{layer_num}")
        elif 'decoder' in key:
            layer_num = parts[2] if len(parts) > 2 else 'unknown'
            unique_layers.add(f"decoder.{layer_num}")
    
    encoder_layer_nums = [l for l in unique_layers if l.startswith('encoder')]
    decoder_layer_nums = [l for l in unique_layers if l.startswith('decoder')]
    
    print(f"Encoder layers found: {sorted(encoder_layer_nums)}")
    print(f"Decoder layers found: {sorted(decoder_layer_nums)}")
    
    return {
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'd_model': d_model,
        'num_encoder_layers': len(encoder_layer_nums),
        'num_decoder_layers': len(decoder_layer_nums),
        'num_heads': num_heads,
        'd_ff': d_ff
    }

if __name__ == "__main__":
    params = analyze_model()
    print(f"\n=== Corrected Configuration ===")
    for key, value in params.items():
        print(f"{key}: {value}")
