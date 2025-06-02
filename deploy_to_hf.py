import torch
import json
import os
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file

# --- Configuration (Load from our current setup) ---
# Load configuration from config_local.json and vocabularies from JSON files
def load_config_and_vocabularies():
    # Load config
    with open('config_local.json', 'r') as f:
        config = json.load(f)
    
    # Load vocabularies
    with open('vocab_src_wmt14.json', 'r') as f:
        vocab_src_data = json.load(f)
    with open('vocab_tgt_wmt14.json', 'r') as f:
        vocab_tgt_data = json.load(f)
    
    # Create simple vocabulary classes
    class SimpleVocab:
        def __init__(self, vocab_data):
            self.stoi = vocab_data['stoi']
            self.itos = vocab_data['itos']
            
        def __getitem__(self, token):
            return self.stoi.get(token, self.stoi.get('<unk>', 0))
        
        def __len__(self):
            return len(self.stoi)
        
        def lookup_tokens(self, indices):
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
    
    return config, vocab_transform

# Load current configuration
config_data, vocab_transform = load_config_and_vocabularies()

# Extract configuration values
D_MODEL = config_data['d_model']
NUM_ENCODER_LAYERS = config_data['num_encoder_layers']
NUM_DECODER_LAYERS = config_data['num_decoder_layers']
NUM_HEADS = config_data['num_heads']
D_FF = config_data['d_ff']
DROPOUT = config_data['dropout']
MAX_SEQ_LEN = config_data['max_seq_len']
SRC_LANGUAGE = config_data['src_language']
TRG_LANGUAGE = config_data['tgt_language']
PAD_IDX = config_data['pad_idx']
BOS_IDX = config_data['bos_idx']
EOS_IDX = config_data['eos_idx']
UNK_IDX = config_data['unk_idx']

# --- Hugging Face Configuration ---
HUGGING_FACE_USERNAME = "posity"
HUGGING_FACE_REPO_NAME = "attention_is_all_you_need_de_en_wmt14"
MODEL_PATH = "transformer_model_wmt14_epoch_3.pth" # Path to your trained model weights
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

    # 3. Save vocabularies (save JSON files instead of .pth)
    vocab_src_path = os.path.join(LOCAL_REPO_PATH, "vocab_src_wmt14.json")
    with open(vocab_src_path, 'w') as f:
        json.dump({
            'stoi': vocab_transform[SRC_LANGUAGE].stoi,
            'itos': vocab_transform[SRC_LANGUAGE].itos
        }, f, indent=2)
    print(f"Saved source vocabulary to {vocab_src_path}")

    vocab_tgt_path = os.path.join(LOCAL_REPO_PATH, "vocab_tgt_wmt14.json")
    with open(vocab_tgt_path, 'w') as f:
        json.dump({
            'stoi': vocab_transform[TRG_LANGUAGE].stoi,
            'itos': vocab_transform[TRG_LANGUAGE].itos
        }, f, indent=2)
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
- **Training Data:** WMT14 German-English (subset)

### Configuration

```json
{json.dumps(config, indent=4)}
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

REPO_ID = "{repo_id}"

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
"""
    readme_path = os.path.join(LOCAL_REPO_PATH, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Saved README.md to {readme_path}")

    # 6. Copy the model.py file so users can load the model
    import shutil
    model_py_path = os.path.join(LOCAL_REPO_PATH, "model.py")
    shutil.copy("model.py", model_py_path)
    print(f"Copied model.py to {model_py_path}")

    # 5. Upload files
    api = HfApi()
    try:
        print(f"Uploading {MODEL_PATH} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="transformer_model_wmt14_epoch_3.pth", # Use the actual model name
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Uploading {config_path} to {repo_id}...")
        api.upload_file(path_or_fileobj=config_path, path_in_repo="config.json", repo_id=repo_id, repo_type="model")
        print(f"Uploading {vocab_src_path} to {repo_id}...")
        api.upload_file(path_or_fileobj=vocab_src_path, path_in_repo="vocab_src_wmt14.json", repo_id=repo_id, repo_type="model")
        print(f"Uploading {vocab_tgt_path} to {repo_id}...")
        api.upload_file(path_or_fileobj=vocab_tgt_path, path_in_repo="vocab_tgt_wmt14.json", repo_id=repo_id, repo_type="model")
        print(f"Uploading model.py to {repo_id}...")
        api.upload_file(path_or_fileobj=model_py_path, path_in_repo="model.py", repo_id=repo_id, repo_type="model")
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
        # Check if the model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"Model file {MODEL_PATH} not found. Please ensure the model is trained and saved.")
        else:
            deploy()
