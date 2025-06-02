#!/usr/bin/env python3
"""Get the exact model parameters"""

import torch

def get_model_params():
    state_dict = torch.load("transformer_best_model.pt", map_location='cpu')
    
    # Get d_ff from feed forward layer
    d_ff = None
    for key in state_dict.keys():
        if 'feed_forward.fc1.weight' in key:
            d_ff = state_dict[key].shape[0]
            break
    
    if d_ff is None:
        # Try alternative naming
        for key in state_dict.keys():
            if 'linear1.weight' in key or 'fc1.weight' in key:
                d_ff = state_dict[key].shape[0]
                break
    
    print(f"d_ff: {d_ff}")
    
    # Print all keys to understand structure
    print("\nAll model keys:")
    for key in sorted(state_dict.keys()):
        print(f"  {key}: {state_dict[key].shape}")

if __name__ == "__main__":
    get_model_params()
