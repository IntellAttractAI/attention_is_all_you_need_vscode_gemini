
# attention_is_all_you_need_de_en_wmt14

This is a **demonstration** Transformer model for machine translation from de to en, based on the "Attention Is All You Need" paper.

## ⚠️ Important Limitation Notice

**This model has only been trained for 3 epochs and produces very limited output patterns.** This is a demonstration of the transformer architecture rather than a production-ready translation system.

### Expected Behavior:
- The model will produce short, repetitive phrases like "The Union ." or "We have ." regardless of the input
- This is due to insufficient training (only 3 epochs vs. hundreds typically needed)
- The model architecture is correct, but the weights have not converged to meaningful translation patterns

## Model Details

- **Architecture:** Custom Transformer (Encoder-Decoder)
- **Framework:** PyTorch
- **Source Language:** de
- **Target Language:** en
- **Training Data:** WMT14 German-English (subset)

### Configuration

```json
{
    "model_type": "transformer_mt",
    "d_model": 256,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "num_heads": 4,
    "d_ff": 512,
    "dropout": 0.1,
    "max_seq_len": 5000,
    "src_language": "de",
    "tgt_language": "en",
    "src_vocab_size": 357611,
    "tgt_vocab_size": 191005,
    "pad_idx": 1,
    "bos_idx": 2,
    "eos_idx": 3,
    "unk_idx": 0,
    "architecture": "TransformerSeq2Seq",
    "framework": "pytorch"
}
```

## How to Use

This model can be used with the accompanying Streamlit application or directly in PyTorch.
To load the model, config, and vocabularies from the Hub:

```python
from huggingface_hub import hf_hub_download
import torch
import json

# Define your Transformer class (copy from model.py)
# class Transformer(...): ...

REPO_ID = "posity/attention_is_all_you_need_de_en_wmt14"

# Download files
model_weights_path = hf_hub_download(repo_id=REPO_ID, filename="transformer_model_wmt14_epoch_3.pth")
config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
vocab_src_path = hf_hub_download(repo_id=REPO_ID, filename="vocab_src_wmt14.json")
vocab_tgt_path = hf_hub_download(repo_id=REPO_ID, filename="vocab_tgt_wmt14.json")

# Load config
with open(config_path, 'r') as f:
    config = json.load(f)

# Load vocabularies
with open(vocab_src_path, 'r') as f:
    vocab_src_data = json.load(f)
with open(vocab_tgt_path, 'r') as f:
    vocab_tgt_data = json.load(f)

# Create vocabulary objects
class SimpleVocab:
    def __init__(self, vocab_data):
        self.stoi = vocab_data['stoi']
        self.itos = vocab_data['itos']
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get('<unk>', 0))
    
    def __len__(self):
        return len(self.stoi)

vocab_src = SimpleVocab(vocab_src_data)
vocab_tgt = SimpleVocab(vocab_tgt_data)

# Instantiate model
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
)
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()
```

## Model Architecture

- **Model size**: ~144M parameters
- **d_model**: 256
- **Encoder layers**: 3
- **Decoder layers**: 3  
- **Attention heads**: 4
- **Feed-forward dimension**: 512
- **Max sequence length**: 5000
- **Vocabulary sizes**: 357,611 (German), 191,005 (English)

## Limitations

This model was trained on WMT14 data and may have limitations in vocabulary and translation quality for out-of-domain sentences. The model was optimized for M4 GPU training with reduced parameters.
