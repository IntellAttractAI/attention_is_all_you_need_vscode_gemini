#!/usr/bin/env python3
"""Check vocabulary sizes"""

import json

def check_vocab_sizes():
    files = [
        ("vocab_src.json", "Source"),
        ("vocab_tgt.json", "Target"),
        ("vocab_src_wmt14.json", "Source WMT14"),
        ("vocab_tgt_wmt14.json", "Target WMT14")
    ]
    
    for filename, label in files:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            size = len(data['stoi'])
            print(f"{label} ({filename}): {size} tokens")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

if __name__ == "__main__":
    check_vocab_sizes()
