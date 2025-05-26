import torch
import json
import os
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file

# --- Configuration (Import from train.py or define here) ---
# These should match the model you trained and want to deploy
from train import (
    D_MODEL, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_SEQ_LEN,
    SRC_LANGUAGE, TRG_LANGUAGE,
    vocab_transform, # This will be loaded by train.py when imported
    PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX # Needed for config if not inferred from vocab
)

# --- Hugging Face Configuration ---
HUGGING_FACE_USERNAME = "posity"
HUGGING_FACE_REPO_NAME = "attention_is_all_you_need_gb_en"
MODEL_PATH = "transformer_best_model.pt" # Path to your trained model weights
LOCAL_REPO_PATH = "hf_deployment_repo" # Temporary local folder to stage files

# Ensure the local repo path exists and is empty or doesn't exist
if os.path.exists(LOCAL_REPO_PATH):
    # Basic cleanup - for a real script, you might want more robust handling
    for f in os.listdir(LOCAL_REPO_PATH):
        os.remove(os.path.join(LOCAL_REPO_PATH, f))
else:
    os.makedirs(LOCAL_REPO_PATH, exist_ok=True)

def deploy():
    print(f"Starting deployment to Hugging Face Hub: {HUGGING_FACE_USERNAME}/{HUGGING_FACE_REPO_NAME}")

    # 1. Create or get the repository on Hugging Face Hub
    repo_id = f"{HUGGING_FACE_USERNAME}/{HUGGING_FACE_REPO_NAME}"
    try:
        create_repo(repo_id, private=False, exist_ok=True) # Set private=True if you want
        print(f"Repository '{repo_id}' created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # 2. Prepare config.json
    # vocab_transform should be loaded when train.py is imported
    src_vocab_size = len(vocab_transform[SRC_LANGUAGE])
    tgt_vocab_size = len(vocab_transform[TRG_LANGUAGE])

    config = {
        "model_type": "transformer_mt", # Custom model type
        "d_model": D_MODEL,
        "num_encoder_layers": NUM_ENCODER_LAYERS,
        "num_decoder_layers": NUM_DECODER_LAYERS,
        "num_heads": NUM_HEADS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "max_seq_len": MAX_SEQ_LEN,
        "src_language": SRC_LANGUAGE,
        "tgt_language": TRG_LANGUAGE,
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "pad_idx": PAD_IDX,
        "bos_idx": BOS_IDX,
        "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX,
        "architecture": "TransformerSeq2Seq", # A name for your architecture
        "framework": "pytorch"
    }
    config_path = os.path.join(LOCAL_REPO_PATH, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Saved config.json to {config_path}")

    # 3. Save vocabularies
    vocab_src_path = os.path.join(LOCAL_REPO_PATH, "vocab_src.pth")
    torch.save(vocab_transform[SRC_LANGUAGE], vocab_src_path)
    print(f"Saved source vocabulary to {vocab_src_path}")

    vocab_tgt_path = os.path.join(LOCAL_REPO_PATH, "vocab_tgt.pth")
    torch.save(vocab_transform[TRG_LANGUAGE], vocab_tgt_path)
    print(f"Saved target vocabulary to {vocab_tgt_path}")

    # 4. Prepare Model Card (README.md)
    readme_content = f"""
# {HUGGING_FACE_REPO_NAME}

This is a Transformer model for machine translation from {SRC_LANGUAGE} to {TRG_LANGUAGE}, based on the "Attention Is All You Need" paper.

## Model Details

- **Architecture:** Custom Transformer (Encoder-Decoder)
- **Framework:** PyTorch
- **Source Language:** {SRC_LANGUAGE}
- **Target Language:** {TRG_LANGUAGE}
- **Training Data:** Multi30k (subset)

### Configuration

```json
{json.dumps(config, indent=4)}
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

REPO_ID = "{repo_id}"

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
"""
    readme_path = os.path.join(LOCAL_REPO_PATH, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Saved README.md to {readme_path}")

    # 5. Upload files
    api = HfApi()
    try:
        print(f"Uploading {MODEL_PATH} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="transformer_best_model.pt", # Keep original name or use pytorch_model.bin
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Uploading {config_path} to {repo_id}...")
        api.upload_file(path_or_fileobj=config_path, path_in_repo="config.json", repo_id=repo_id, repo_type="model")
        print(f"Uploading {vocab_src_path} to {repo_id}...")
        api.upload_file(path_or_fileobj=vocab_src_path, path_in_repo="vocab_src.pth", repo_id=repo_id, repo_type="model")
        print(f"Uploading {vocab_tgt_path} to {repo_id}...")
        api.upload_file(path_or_fileobj=vocab_tgt_path, path_in_repo="vocab_tgt.pth", repo_id=repo_id, repo_type="model")
        print(f"Uploading {readme_path} to {repo_id}...")
        api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=repo_id, repo_type="model")
        print("All files uploaded successfully!")
        print(f"Visit your model at: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"Error uploading files: {e}")

    finally:
        # Clean up local staging folder (optional)
        # for f in os.listdir(LOCAL_REPO_PATH):
        #     os.remove(os.path.join(LOCAL_REPO_PATH, f))
        # os.rmdir(LOCAL_REPO_PATH)
        # print(f"Cleaned up {LOCAL_REPO_PATH}")
        pass 

if __name__ == "__main__":
    # Ensure you are logged in: `huggingface-cli login`
    if HfFolder.get_token() is None:
        print("Hugging Face token not found. Please log in using 'huggingface-cli login'.")
    else:
        # This script assumes train.py can be imported to get vocab_transform and constants.
        # If train.py has a main execution block (if __name__ == "__main__"), 
        # ensure it doesn't run automatically when imported.
        # For this to work, vocab_transform must be populated at the module level in train.py,
        # which it is (it's built when train.py is loaded).
        
        # Check if the model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"Model file {MODEL_PATH} not found. Please ensure the model is trained and saved.")
        else:
            deploy()
