#!/usr/bin/env python3

import torch
import json
import re
from model import Transformer, generate_square_subsequent_mask, create_padding_mask

# Load local config
with open('config_local.json', 'r') as f:
    config = json.load(f)

print('Local config:', config)

# Load local vocabularies
with open('vocab_src_wmt14.json', 'r') as f:
    vocab_src_data = json.load(f)
with open('vocab_tgt_wmt14.json', 'r') as f:
    vocab_tgt_data = json.load(f)

# Create vocab class (exactly like app_local.py)
class SimpleVocab:
    def __init__(self, vocab_data):
        self.stoi = vocab_data['stoi']
        self.itos = vocab_data['itos']  # This is a list, not a dict
        
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get('<unk>', 0))
    
    def __len__(self):
        return len(self.stoi)
    
    def lookup_tokens(self, indices):
        # itos is a list, so we can index directly
        result = []
        for idx in indices:
            if 0 <= idx < len(self.itos):
                result.append(self.itos[idx])
            else:
                result.append('<unk>')
        return result

vocab_src = SimpleVocab(vocab_src_data)
vocab_tgt = SimpleVocab(vocab_tgt_data)

vocab_transform = {
    config['src_language']: vocab_src,
    config['tgt_language']: vocab_tgt
}

def simple_tokenize(text):
    """Simple tokenization by splitting on whitespace and punctuation"""
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load model (exactly like app_local.py)
model = Transformer(
    src_vocab_size=len(vocab_src),
    tgt_vocab_size=len(vocab_tgt),
    d_model=config['d_model'],
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    num_heads=config['num_heads'],
    d_ff=config['d_ff'],
    dropout=config['dropout'],
    max_seq_len=config['max_seq_len']
).to(DEVICE)

# Load state dict
state_dict = torch.load("transformer_model_wmt14_epoch_3.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully")

# Translation function (exactly like app_local.py)
def translate_sentence(model, sentence: str, max_length: int = 50):
    if model is None or vocab_transform is None:
        return "Model or vocabularies not loaded."
    
    model.eval()
    src_lang = config['src_language']
    tgt_lang = config['tgt_language']
    
    # Tokenize source sentence
    src_tokens = simple_tokenize(sentence)
    print(f"Source tokens: {src_tokens}")
    
    # Convert to indices using vocabulary
    src_indices = [config['bos_idx']]
    for token in src_tokens:
        try:
            idx = vocab_transform[src_lang][token]
        except KeyError:
            idx = config['unk_idx']  # Use unknown token
        src_indices.append(idx)
        print(f"Token '{token}' -> index {idx}")
    src_indices.append(config['eos_idx'])
    
    print(f"Source indices: {src_indices}")
    
    # Create source tensor
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(DEVICE)  # (1, src_len)
    
    # Create source padding mask
    src_padding_mask = create_padding_mask(src_tensor, config['pad_idx']).to(DEVICE)
    
    with torch.no_grad():
        # Encode source
        memory = model.encode(src_tensor, src_padding_mask)
        
        # Start with BOS token for target
        tgt_tokens = [config['bos_idx']]
        
        for i in range(max_length):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(DEVICE)
            tgt_mask = generate_square_subsequent_mask(len(tgt_tokens), device=DEVICE)
            
            # Decode
            decoder_output = model.decode(tgt_tensor, memory, tgt_mask, src_padding_mask)
            
            # Get next token prediction
            next_token_logits = model.generator(decoder_output[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1).item()
            
            print(f"Step {i}: Generated token index {next_token}")
            
            # Check what this token corresponds to
            if 0 <= next_token < len(vocab_tgt.itos):
                token_word = vocab_tgt.itos[next_token]
                print(f"  -> Word: '{token_word}'")
            
            tgt_tokens.append(next_token)
            
            # Stop if EOS token is generated
            if next_token == config['eos_idx']:
                print("EOS token generated, stopping")
                break
    
    # Convert indices back to tokens
    output_tokens = tgt_tokens[1:]  # Remove BOS
    if output_tokens and output_tokens[-1] == config['eos_idx']:
        output_tokens = output_tokens[:-1]  # Remove EOS
    
    # Convert to words using vocabulary
    try:
        translated_words = vocab_transform[tgt_lang].lookup_tokens(output_tokens)
        return " ".join(translated_words)
    except Exception as e:
        return f"Error in translation: {e}"

# Test translation
test_sentences = [
    "Eine Katze sitzt auf dem Tisch.",
    "Der Mann geht zur Arbeit.",
    "Das Wetter ist heute schön."
]

for sentence in test_sentences:
    print(f"\n--- Testing: {sentence} ---")
    result = translate_sentence(model, sentence)
    print(f"Translation: {result}")
