
# attention_is_all_you_need_gb_en

This is a Transformer model for machine translation from de to en, based on the "Attention Is All You Need" paper.

## Model Details

- **Architecture:** Custom Transformer (Encoder-Decoder)
- **Framework:** PyTorch
- **Source Language:** de
- **Target Language:** en
- **Training Data:** Multi30k (subset)

### Configuration

```json
{
    "model_type": "transformer_mt",
    "d_model": 512,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_seq_len": 100,
    "src_language": "de",
    "tgt_language": "en",
    "src_vocab_size": 19214,
    "tgt_vocab_size": 10837,
    "pad_idx": 1,
    "bos_idx": 2,
    "eos_idx": 3,
    "unk_idx": 0,
    "architecture": "TransformerSeq2Seq",
    "framework": "pytorch"
}
```

## How to Use (Example with Streamlit app or direct PyTorch)

This model can be used with the accompanying Streamlit application or directly in PyTorch.
To load the model, config, and vocabularies from the Hub:

```python
from huggingface_hub import hf_hub_download
import torch
import json

# Example: Define your Transformer class here (or import it)
# class Transformer(...): ...

REPO_ID = "posity/attention_is_all_you_need_gb_en"

# Download files
model_weights_path = hf_hub_download(repo_id=REPO_ID, filename="transformer_best_model.pt")
config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
vocab_src_path = hf_hub_download(repo_id=REPO_ID, filename="vocab_src.pth")
vocab_tgt_path = hf_hub_download(repo_id=REPO_ID, filename="vocab_tgt.pth")

# Load config
with open(config_path, 'r') as f:
    config = json.load(f)

# Load vocabularies
vocab_src = torch.load(vocab_src_path)
vocab_tgt = torch.load(vocab_tgt_path)

# Instantiate model (ensure your Transformer class definition is available)
# model = Transformer(
#     src_vocab_size=config['src_vocab_size'],
#     tgt_vocab_size=config['tgt_vocab_size'],
#     d_model=config['d_model'],
#     num_encoder_layers=config['num_encoder_layers'],
#     num_decoder_layers=config['num_decoder_layers'],
#     num_heads=config['num_heads'],
#     d_ff=config['d_ff'],
#     dropout=config['dropout'],
#     max_seq_len=config['max_seq_len']
# )
# model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
# model.eval()

print("Model, config, and vocabs would be loaded here.")
```

## Limitations

This model was trained on a subset of Multi30k and may have limitations in vocabulary and translation quality for out-of-domain sentences.
