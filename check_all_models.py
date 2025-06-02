#!/usr/bin/env python3
"""Check the architecture of the recent .pth model files"""

import torch

def check_model_architecture(model_path):
    print(f"\n=== Analyzing {model_path} ===")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Get architecture info
        src_vocab_size = state_dict['src_embedding.embedding.weight'].shape[0]
        tgt_vocab_size = state_dict['tgt_embedding.embedding.weight'].shape[0]
        d_model = state_dict['src_embedding.embedding.weight'].shape[1]
        
        print(f"Source vocab size: {src_vocab_size}")
        print(f"Target vocab size: {tgt_vocab_size}")
        print(f"d_model: {d_model}")
        
        # Check if generator exists and its shape
        if 'generator.weight' in state_dict:
            gen_shape = state_dict['generator.weight'].shape
            print(f"Generator shape: {gen_shape}")
        
        # Count layers
        encoder_layers = len([k for k in state_dict.keys() if k.startswith('encoder.layers.') and k.endswith('.self_attn.query_linear.weight')])
        decoder_layers = len([k for k in state_dict.keys() if k.startswith('decoder.layers.') and k.endswith('.self_attn.query_linear.weight')])
        
        print(f"Encoder layers: {encoder_layers}")
        print(f"Decoder layers: {decoder_layers}")
        
        # Check feed forward dimension
        if 'encoder.layers.0.feed_forward.linear1.weight' in state_dict:
            d_ff = state_dict['encoder.layers.0.feed_forward.linear1.weight'].shape[0]
            print(f"d_ff: {d_ff}")
        
        return {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'd_model': d_model,
            'num_encoder_layers': encoder_layers,
            'num_decoder_layers': decoder_layers,
            'd_ff': d_ff if 'encoder.layers.0.feed_forward.linear1.weight' in state_dict else None
        }
        
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

if __name__ == "__main__":
    models = [
        "transformer_best_model.pt",
        "transformer_final_model.pt", 
        "transformer_model_wmt14_epoch_3.pth"
    ]
    
    for model in models:
        check_model_architecture(model)
