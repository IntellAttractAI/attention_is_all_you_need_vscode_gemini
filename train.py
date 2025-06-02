import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import spacy
import os
import json
# Ensure your model.py has Transformer, generate_square_subsequent_mask, create_padding_mask
from model import Transformer, generate_square_subsequent_mask, create_padding_mask
import math
import time
from tqdm import tqdm
import random
from datasets import load_dataset # Added for Hugging Face Datasets

# --- Configuration ---
# Configure device for Apple Silicon (M1/M2/M3/M4) to use Metal Performance Shaders (MPS)
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Using CUDA GPU")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU")

# Dataset Configuration (WMT14 from Hugging Face)
DATASET_NAME = "wmt14"
DATASET_CONFIG = "de-en" # German-English pair

# Paths for saving models and vocabs (specific to WMT14)
MODEL_SAVE_PATH = "transformer_model_wmt14_epoch_{epoch}.pth"
VOCAB_SRC_FILE_NAME = "vocab_src_wmt14.json"
VOCAB_TGT_FILE_NAME = "vocab_tgt_wmt14.json"
VOCAB_SAVE_DIR = "." # Save vocabs in the current project directory

# Languages and Spacy Models
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
# Ensure these spacy models are downloaded or listed in requirements.txt for cloud environments
SRC_LANGUAGE_MODEL = "de_core_news_sm"
TGT_LANGUAGE_MODEL = "en_core_web_sm"

# Model Hyperparameters (optimized for fast development on M4 MacBook)
SRC_VOCAB_SIZE = 0 # Will be set after loading/building vocab
TGT_VOCAB_SIZE = 0 # Will be set after loading/building vocab
EMB_SIZE = 256 # Reduced from 512 for faster training
NHEAD = 4 # Reduced from 8 for faster training
FFN_HID_DIM = 512 # Reduced from 2048 for faster training
NUM_ENCODER_LAYERS = 3 # Reduced from 6 for faster training
NUM_DECODER_LAYERS = 3 # Reduced from 6 for faster training
DROPOUT = 0.1
MAX_SEQ_LEN_CONFIG = 5000 # For Transformer model's PositionalEncoding

# Training Hyperparameters (optimized for fast development)
NUM_EPOCHS = 3 # Reduced for faster experimentation
BATCH_SIZE = 64 # Increased for M4 MacBook GPU utilization (adjust if memory issues occur)
LEARNING_RATE = 0.001 # Increased learning rate for faster convergence
BETAS = (0.9, 0.98)
EPS = 1e-9
GRAD_CLIP_NORM = 1.0 # Gradient clipping value

# Fast training optimizations
MAX_SAMPLES_TRAIN = 50000 # Limit training samples for faster experimentation (set to None for full dataset)
MAX_SAMPLES_VAL = 3000 # Limit validation samples
VOCAB_MIN_FREQ = 20 # Increased from 5 to reduce vocabulary size significantly
MAX_SEQ_LEN = 100 # Reduced from default for faster processing

# Special token definitions
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
SPECIAL_SYMBOLS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

# --- Helper Class for JSON Vocabularies ---
class JsonVocab:
    def __init__(self, vocab_data=None, stoi=None, itos=None):
        if vocab_data:
            self._stoi = vocab_data['stoi']
            self._itos = vocab_data['itos']
        elif stoi is not None and itos is not None:
            self._stoi = stoi
            self._itos = itos
        else:
            raise ValueError("Must provide either vocab_data or both stoi and itos.")

        # Ensure special tokens are in stoi and get their indices
        self.UNK_IDX = self._stoi.get(UNK_TOKEN)
        self.PAD_IDX = self._stoi.get(PAD_TOKEN)
        self.BOS_IDX = self._stoi.get(BOS_TOKEN)
        self.EOS_IDX = self._stoi.get(EOS_TOKEN)

        if None in [self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX]:
            missing = [s for s, i in zip(SPECIAL_SYMBOLS, [self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX]) if i is None]
            raise ValueError(f"Special tokens {missing} not found in vocabulary's stoi map.")


    def get_stoi(self):
        return self._stoi

    def get_itos(self):
        return self._itos

    def __getitem__(self, token):
        return self._stoi.get(token, self.UNK_IDX)

    def __len__(self):
        return len(self._itos)

    @classmethod
    def build_from_iterator(cls, iterator, specials, min_freq=VOCAB_MIN_FREQ): # Use optimized min_freq
        counter = {}
        # Wrap iterator with tqdm for progress display
        for tokens in tqdm(iterator, desc="Counting tokens for vocab build"):
            for token in tokens:
                counter[token] = counter.get(token, 0) + 1
        
        stoi = {}
        itos = list(specials) # Start with special symbols

        for s_idx, s_tok in enumerate(specials):
            stoi[s_tok] = s_idx
        
        # Sort tokens by frequency (descending), then alphabetically for tie-breaking
        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

        for token, freq in tqdm(sorted_tokens, desc="Building vocab from sorted tokens"):
            if freq >= min_freq and token not in stoi: # Add if freq is sufficient and not a special token
                stoi[token] = len(itos)
                itos.append(token)
        return cls(stoi=stoi, itos=itos)

# --- Data Loading and Preprocessing ---

def limit_dataset_size(dataset, max_samples=None):
    """Limit dataset size for faster experimentation"""
    if max_samples is None:
        return dataset
    total_samples = len(dataset)
    if total_samples <= max_samples:
        return dataset
    print(f"Limiting dataset from {total_samples} to {max_samples} samples for faster training")
    return dataset.select(range(max_samples))

# Load Spacy tokenizers
try:
    spacy_src = spacy.load(SRC_LANGUAGE_MODEL)
    spacy_tgt = spacy.load(TGT_LANGUAGE_MODEL)
except OSError:
    print(f"Spacy models not found. Please run:\n"
          f"python -m spacy download {SRC_LANGUAGE_MODEL}\n"
          f"python -m spacy download {TGT_LANGUAGE_MODEL}\n"
          f"Or ensure they are listed in requirements.txt for cloud environments.")
    exit(1)

token_transform = {
    SRC_LANGUAGE: spacy_src.tokenizer,
    TGT_LANGUAGE: spacy_tgt.tokenizer
}

# Load WMT14 dataset from Hugging Face
print(f"Loading WMT14 dataset ('{DATASET_NAME}', config '{DATASET_CONFIG}'). This may take time for the first download...")
try:
    # Using trust_remote_code=True might be necessary for some datasets on the Hub
    # For wmt14, it's generally not needed but good to be aware of.
    wmt_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
except Exception as e:
    print(f"Failed to load dataset {DATASET_NAME} with config {DATASET_CONFIG}: {e}")
    print("Ensure you have an internet connection, the 'datasets' library is installed correctly,")
    print("and the dataset identifier is correct. You might need 'trust_remote_code=True'.")
    exit(1)

def yield_tokens_from_hf_dataset(dataset_split, language_key, tokenizer_fn):
    """Yields tokenized text from a Hugging Face dataset split."""
    # Accessing the text: example['translation'][language_key]
    for example in dataset_split: # No need for tqdm here, build_from_iterator will use it
        text = example['translation'][language_key]
        yield [token.text for token in tokenizer_fn(text.strip())]

# Load or build vocabularies
vocab_src_path = os.path.join(VOCAB_SAVE_DIR, VOCAB_SRC_FILE_NAME)
vocab_tgt_path = os.path.join(VOCAB_SAVE_DIR, VOCAB_TGT_FILE_NAME)

if os.path.exists(vocab_src_path) and os.path.exists(vocab_tgt_path):
    print("Loading existing WMT14 vocabularies...")
    with open(vocab_src_path, 'r', encoding='utf-8') as f:
        vocab_src_data = json.load(f)
    with open(vocab_tgt_path, 'r', encoding='utf-8') as f:
        vocab_tgt_data = json.load(f)
    vocab_src = JsonVocab(vocab_data=vocab_src_data)
    vocab_tgt = JsonVocab(vocab_data=vocab_tgt_data)
else:
    print("Building WMT14 vocabularies from 'train' split...")
    
    src_token_iterator = yield_tokens_from_hf_dataset(wmt_dataset['train'], SRC_LANGUAGE, token_transform[SRC_LANGUAGE])
    vocab_src = JsonVocab.build_from_iterator(src_token_iterator, SPECIAL_SYMBOLS)
    
    tgt_token_iterator = yield_tokens_from_hf_dataset(wmt_dataset['train'], TGT_LANGUAGE, token_transform[TGT_LANGUAGE])
    vocab_tgt = JsonVocab.build_from_iterator(tgt_token_iterator, SPECIAL_SYMBOLS)

    print(f"Saving source vocabulary to {vocab_src_path}")
    with open(vocab_src_path, 'w', encoding='utf-8') as f:
        json.dump({'stoi': vocab_src.get_stoi(), 'itos': vocab_src.get_itos()}, f, ensure_ascii=False, indent=4)
    
    print(f"Saving target vocabulary to {vocab_tgt_path}")
    with open(vocab_tgt_path, 'w', encoding='utf-8') as f:
        json.dump({'stoi': vocab_tgt.get_stoi(), 'itos': vocab_tgt.get_itos()}, f, ensure_ascii=False, indent=4)

SRC_VOCAB_SIZE = len(vocab_src)
TGT_VOCAB_SIZE = len(vocab_tgt)
PAD_IDX = vocab_src.PAD_IDX # Get PAD_IDX from the loaded/built vocab

print(f"Source Vocab Size (WMT14): {SRC_VOCAB_SIZE}")
print(f"Target Vocab Size (WMT14): {TGT_VOCAB_SIZE}")
print(f"PAD_IDX: {PAD_IDX}")


# Text transformation function
def text_transform_fn(vocab, tokenizer_fn):
    def func(text_input):
        tokens = [token.text for token in tokenizer_fn(text_input.strip())]
        return [vocab.BOS_IDX] + [vocab[token] for token in tokens] + [vocab.EOS_IDX]
    return func

text_transform_src = text_transform_fn(vocab_src, token_transform[SRC_LANGUAGE])
text_transform_tgt = text_transform_fn(vocab_tgt, token_transform[TGT_LANGUAGE])

class HFTranslationDataset(Dataset):
    def __init__(self, hf_dataset_split, src_lang_key, tgt_lang_key, src_transform_fn, tgt_transform_fn):
        self.dataset_split = hf_dataset_split
        self.src_lang_key = src_lang_key
        self.tgt_lang_key = tgt_lang_key
        self.src_transform_fn = src_transform_fn
        self.tgt_transform_fn = tgt_transform_fn

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, idx):
        example = self.dataset_split[idx]['translation']
        src_text = example[self.src_lang_key]
        tgt_text = example[self.tgt_lang_key]
        
        src_sample = self.src_transform_fn(src_text)
        tgt_sample = self.tgt_transform_fn(tgt_text)
        return torch.tensor(src_sample, dtype=torch.long), torch.tensor(tgt_sample, dtype=torch.long)

# Collate function (expects batch_first=True for your Transformer)
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        # Filter out very long sequences for faster training
        if len(src_sample) <= MAX_SEQ_LEN and len(tgt_sample) <= MAX_SEQ_LEN:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)
    
    # Skip batch if all samples were filtered out
    if not src_batch:
        return None, None
    
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True) 
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

# --- Model, Loss, Optimizer ---
transformer = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=EMB_SIZE,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    num_heads=NHEAD,
    d_ff=FFN_HID_DIM,
    dropout=DROPOUT,
    max_seq_len=MAX_SEQ_LEN_CONFIG # Passed to PositionalEncoding within Transformer
)
transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS)

# --- Training and Evaluation Functions ---
# CRITICAL: Ensure your model.py's mask functions and Transformer.forward are compatible
# with the mask creation logic below.
# Assumptions:
# - generate_square_subsequent_mask(sz, device) -> (sz, sz) boolean, True for allowed.
# - create_padding_mask(seq, pad_idx, device) -> (B, 1, S) boolean, True for non-pad.
# - Transformer.forward(src, tgt, src_mask, tgt_mask) expects:
#   - src_mask: (B, 1, S_src) boolean, True for non-pad.
#   - tgt_mask: (B, S_tgt, S_tgt) boolean, True for allowed (combined lookahead & padding).

def train_epoch(model, optimizer, criterion, train_dataloader):
    model.train()
    losses = 0
    progress_bar = tqdm(train_dataloader, desc="Training Epoch", leave=False)
    batch_count = 0
    for src, tgt in progress_bar: 
        # Skip filtered batches
        if src is None or tgt is None:
            continue
            
        src = src.to(DEVICE) # (B, S_src)
        tgt = tgt.to(DEVICE) # (B, S_tgt)

        tgt_input = tgt[:, :-1] # (B, S_tgt-1)
        
        # Create masks
        src_padding_mask = create_padding_mask(src, PAD_IDX, device=DEVICE) # (B, 1, S_src)
        
        # Target masks
        # look_ahead_mask: (S_tgt-1, S_tgt-1), True for allowed
        tgt_seq_len = tgt_input.size(1)
        bool_tgt_look_ahead_mask = generate_square_subsequent_mask(tgt_seq_len, device=DEVICE)
        # padding_mask for target_input: (B, 1, S_tgt-1), True for non-pad
        bool_tgt_padding_mask = create_padding_mask(tgt_input, PAD_IDX, device=DEVICE) 
        
        # Combine target masks for Transformer: (B, S_tgt-1, S_tgt-1)
        # (B, 1, S_tgt-1) -> transpose to (B, S_tgt-1, 1)
        # (B, S_tgt-1, 1) & (S_tgt-1, S_tgt-1) [broadcasts to (B, S_tgt-1, S_tgt-1)]
        combined_tgt_mask = bool_tgt_padding_mask.transpose(1,2) & bool_tgt_look_ahead_mask.unsqueeze(0)

        logits = model(src, tgt_input, src_padding_mask, combined_tgt_mask)
        # logits shape: (B, S_tgt-1, TGT_VOCAB_SIZE)
        
        optimizer.zero_grad()
        tgt_out = tgt[:, 1:] # (B, S_tgt-1)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        losses += loss.item()
        batch_count += 1
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    return losses / max(batch_count, 1) # Avoid division by zero

def evaluate(model, criterion, val_dataloader):
    model.eval()
    losses = 0
    batch_count = 0
    progress_bar = tqdm(val_dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for src, tgt in progress_bar:
            # Skip filtered batches
            if src is None or tgt is None:
                continue
                
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]

            src_padding_mask = create_padding_mask(src, PAD_IDX, device=DEVICE)
            
            tgt_seq_len = tgt_input.size(1)
            bool_tgt_look_ahead_mask = generate_square_subsequent_mask(tgt_seq_len, device=DEVICE)
            bool_tgt_padding_mask = create_padding_mask(tgt_input, PAD_IDX, device=DEVICE)
            combined_tgt_mask = bool_tgt_padding_mask.transpose(1,2) & bool_tgt_look_ahead_mask.unsqueeze(0)
            
            logits = model(src, tgt_input, src_padding_mask, combined_tgt_mask)
            
            tgt_out = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
            batch_count += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
    return losses / max(batch_count, 1) # Avoid division by zero

# --- Main Training Loop ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Verify that your mask functions in model.py are boolean and compatible:
    # generate_square_subsequent_mask(sz, device) -> (sz, sz) boolean, True for allowed positions.
    # create_padding_mask(seq, pad_idx, device) -> (B, 1, S) boolean, True for non-pad tokens.
    # If they are not, you MUST adjust them in model.py or the mask combination logic above.

    print("Preparing data loaders using WMT14 from Hugging Face...")
    # WMT14 'validation' split is often newstest2013. 'test' is often newstest2014.
    # Check dataset.column_names or dataset['train'].features for exact structure if issues arise.
    train_hf_split = wmt_dataset.get('train')
    val_hf_split = wmt_dataset.get('validation')

    if not train_hf_split:
        print("ERROR: 'train' split not found in the loaded WMT14 dataset.")
        exit(1)
    if not val_hf_split:
        print("ERROR: 'validation' split not found in the loaded WMT14 dataset. Using a subset of train for validation.")
        # Fallback: split train if validation is missing (not ideal for WMT14)
        # This is a placeholder, proper WMT validation set is preferred.
        full_train_len = len(train_hf_split)
        train_len = int(0.95 * full_train_len)
        val_len = full_train_len - train_len
        # Note: This simple split might not be directly supported by HF dataset object like this.
        # For simplicity, if 'validation' is missing, this script will error out unless you implement
        # a more robust splitting mechanism for HF datasets or ensure 'validation' split exists.
        # For now, we assume 'validation' split exists.
        print("Ensure your WMT14 dataset from Hugging Face has a 'validation' split.")


    # Limit dataset sizes for faster experimentation
    print("Limiting dataset sizes for faster training...")
    train_hf_split = limit_dataset_size(train_hf_split, MAX_SAMPLES_TRAIN)
    val_hf_split = limit_dataset_size(val_hf_split, MAX_SAMPLES_VAL)

    train_dataset = HFTranslationDataset(train_hf_split, SRC_LANGUAGE, TGT_LANGUAGE, text_transform_src, text_transform_tgt)
    val_dataset = HFTranslationDataset(val_hf_split, SRC_LANGUAGE, TGT_LANGUAGE, text_transform_src, text_transform_tgt)

    # Using num_workers=0 to avoid multiprocessing pickle errors with local functions
    # pin_memory is beneficial for CUDA and MPS transfers
    use_pin_memory = DEVICE.type in ['cuda', 'mps']
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0, pin_memory=use_pin_memory)
    
    print(f"Training data loader: {len(train_dataloader)} batches of size {BATCH_SIZE}")
    print(f"Validation data loader: {len(val_dataloader)} batches of size {BATCH_SIZE}")

    print("Starting training with WMT14 data...")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(transformer, optimizer, loss_fn, train_dataloader)
        end_time = time.time()
        val_loss = evaluate(transformer, loss_fn, val_dataloader)
        
        epoch_duration_mins, epoch_duration_secs = divmod(end_time - start_time, 60)
        
        print(f"Epoch: {epoch:02}/{NUM_EPOCHS} | Time: {epoch_duration_mins:.0f}m {epoch_duration_secs:.0f}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}")
        
        current_model_save_path = MODEL_SAVE_PATH.format(epoch=epoch)
        try:
            torch.save(transformer.state_dict(), current_model_save_path)
            print(f"Saved model checkpoint to {current_model_save_path}")
        except Exception as e:
            print(f"Error saving model checkpoint: {e}")

    print("Training complete.")
