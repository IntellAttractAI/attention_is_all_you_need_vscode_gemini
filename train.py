import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from model import Transformer, generate_square_subsequent_mask, create_padding_mask
import mlflow
import mlflow.pytorch
import time
import math

# --- Configuration ---
SRC_LANGUAGE = 'de' # Paper uses En-De and En-Fr. Multi30k has En-De.
TRG_LANGUAGE = 'en' # Let's try German to English for Multi30k
# For WMT14 En-Fr as per paper, you'd need a different dataset loader.

# Model Hyperparameters (matches paper's base model)
D_MODEL = 512
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
MAX_SEQ_LEN = 100 # Max sequence length for positional encoding & batching

# Training Hyperparameters
BATCH_SIZE = 128 # Paper uses ~25000 tokens per batch. This is sentence-based.
                  # For Multi30k, 128 sentences is a common starting point.
LEARNING_RATE = 0.0001 # Will be controlled by Adam optimizer with custom schedule
WARMUP_STEPS = 4000
EPOCHS = 20 # Paper trains for 100k steps (base) or 300k steps (big)
              # For Multi30k, 10-20 epochs is a common starting point.
PAD_IDX = 0 # Will be set by vocab
BOS_IDX = 1 # Will be set by vocab
EOS_IDX = 2 # Will be set by vocab
UNK_IDX = 3 # Will be set by vocab

MLFLOW_EXPERIMENT_NAME = "Transformer_AttentionIsAllYouNeed"

# Determine device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon (MPS) backend.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA backend.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU backend.")

# --- Tokenizers and Vocabulary ---
# Using basic tokenizers for simplicity. Paper uses byte-pair encoding (BPE) or word-piece.
# For production, use SentencePiece or a similar subword tokenizer.
token_transform = {}
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[SRC_LANGUAGE] = de_tokenizer
token_transform[TRG_LANGUAGE] = en_tokenizer


# Helper to yield list of tokens
def yield_tokens(data_iter, language):
    language_index = {SRC_LANGUAGE: 0, TRG_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Build Vocabulary
# Download and load Multi30k dataset
# train_iter, val_iter, test_iter = Multi30k(split=('train', 'valid', 'test'))
# Using new API for Multi30k
# from torchtext.datasets import multi30k # This line can be removed if Multi30k is imported at the top
# multi30k.URL["train"] = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.{}.gz"
# multi30k.URL["valid"] = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.{}.gz"
# multi30k.URL["test"] = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.{}.gz"
# multi30k.MD5["train"] = "20140d013d05dd9a72dfde464781a035ac7197697ea3f70e93d10201931a8df1"
# multi30k.MD5["valid"] = "aef2f505686586959983846ae89f76883973f8330855127519b837d563372049"
# multi30k.MD5["test"] = "713c33cf313a0acd9450039797595ec865003e9099b31750a47e660606524866"


# Load data iterators
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TRG_LANGUAGE))
val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TRG_LANGUAGE))
# test_iter = Multi30k(split='test', language_pair=(SRC_LANGUAGE, TRG_LANGUAGE)) # Test set for 2016 flickr

# --- Limit dataset size for 10k training items ---
# MAX_TRAIN_ITEMS = 10000
# # Multi30k has ~1000 validation items. Let\'s maintain a similar ratio for val set.
# # Original train: 29000, Original val: ~1014
# # Ratio val/train = 1014/29000 = ~0.035
# MAX_VAL_ITEMS = int(MAX_TRAIN_ITEMS * (1014 / 29000))
# if MAX_VAL_ITEMS == 0: MAX_VAL_ITEMS = 1 # Ensure at least 1 item for validation

full_train_list = list(train_iter)
full_val_list = list(val_iter)

# limited_train_list = full_train_list[:MAX_TRAIN_ITEMS]
# limited_val_list = full_val_list[:MAX_VAL_ITEMS]

# print(f"Using a subset of Multi30k: {len(limited_train_list)} training items, {len(limited_val_list)} validation items.")
print(f"Using full Multi30k dataset: {len(full_train_list)} training items, {len(full_val_list)} validation items.")
# --- End of dataset limiting ---


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

vocab_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    # Create training data iterator for language
    # train_iter_clone = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TRG_LANGUAGE))
    # Use the limited list for vocab building to be consistent if vocab is small
    # However, it's generally better to build vocab on more data if available.
    # For this specific request, let's build vocab on the limited set to ensure all tokens are seen.
    # If building vocab on full data: train_iter_clone = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TRG_LANGUAGE))
    # For now, to ensure consistency with the 10k items, build vocab from the limited_train_list
    # This requires yield_tokens to work with a list of tuples, not an iterator of tuples directly
    # Modifying yield_tokens slightly for this or preparing the input to yield_tokens:
    def yield_tokens_from_list(data_list, language):
        language_index = {SRC_LANGUAGE: 0, TRG_LANGUAGE: 1}
        for data_sample in data_list:
            yield token_transform[language](data_sample[language_index[language]])

    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens_from_list(full_train_list, ln), # Build vocab from the full training data
        min_freq=1, # Min frequency for a token to be in vocab
        specials=special_symbols,
        special_first=True # Important: <unk> should be at index 0 if not specified otherwise
    )
    vocab_transform[ln].set_default_index(UNK_IDX)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TRG_LANGUAGE])

print(f"Source ({SRC_LANGUAGE}) Vocabulary Size: {SRC_VOCAB_SIZE}")
print(f"Target ({TRG_LANGUAGE}) Vocabulary Size: {TGT_VOCAB_SIZE}")
print(f"PAD_IDX: {PAD_IDX}, BOS_IDX: {BOS_IDX}, EOS_IDX: {EOS_IDX}, UNK_IDX: {UNK_IDX}")

# --- Data Processing and DataLoader ---
# Collate function to process batch of raw text strings
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(vocab_transform[SRC_LANGUAGE](token_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))) , dtype=torch.long))
        tgt_batch.append(torch.tensor(vocab_transform[TRG_LANGUAGE](token_transform[TRG_LANGUAGE](tgt_sample.rstrip("\n"))) , dtype=torch.long))

    # Pad sequences
    src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch_padded, tgt_batch_padded


# train_dataloader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# val_dataloader = DataLoader(list(val_iter), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
train_dataloader = DataLoader(full_train_list, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(full_val_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
# test_dataloader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- Model, Optimizer, Loss ---
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    dropout=DROPOUT,
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE)

# Initialize weights (already done in model constructor, but can be explicit)
# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

# Optimizer with learning rate schedule from the paper
# lrate = d_model^−0.5 * min(step_num^−0.5, step_num * warmup_steps^−1.5)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

# Loss function - CrossEntropyLoss, ignoring padding
# Label smoothing is mentioned in the paper (epsilon_ls = 0.1)
# PyTorch CrossEntropyLoss has label_smoothing parameter (from v1.10.0)
# criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
# For older PyTorch or manual implementation:
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.padding_idx = padding_idx
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 2)) # -2 for pad and true class
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0 # Mask padding
            mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
            if mask.dim() > 0:
                 true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# criterion = LabelSmoothingLoss(classes=TGT_VOCAB_SIZE, padding_idx=PAD_IDX, smoothing=0.1)
# Using PyTorch's built-in for simplicity if available and correct version
if hasattr(nn, 'CrossEntropyLoss') and 'label_smoothing' in nn.CrossEntropyLoss.__init__.__annotations__:
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    print("Using PyTorch CrossEntropyLoss with label_smoothing.")
else:
    criterion = LabelSmoothingLoss(classes=TGT_VOCAB_SIZE, padding_idx=PAD_IDX, smoothing=0.1)
    print("Using custom LabelSmoothingLoss.")


# Learning rate scheduler function
step_num_ = 0
def lr_scheduler(step_num, d_model, warmup_steps):
    step_num +=1 # 1-based step number
    arg1 = step_num ** -0.5
    arg2 = step_num * (warmup_steps ** -1.5)
    return (d_model ** -0.5) * min(arg1, arg2)

# --- Training and Evaluation Loop ---
def train_epoch(model, dataloader, optimizer, criterion, current_epoch, total_epochs):
    global step_num_
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (src, tgt) in enumerate(dataloader):
        step_num_ += 1
        # Update learning rate based on the formula
        new_lr = lr_scheduler(step_num_, D_MODEL, WARMUP_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        src = src.to(DEVICE)  # (B, S_src)
        tgt = tgt.to(DEVICE)  # (B, S_tgt)

        # Prepare target for loss calculation (decoder output shifted right)
        # Input to decoder: <bos> w1 w2 ... wn
        # Target for loss:   w1 w2 ... wn <eos>
        tgt_input = tgt[:, :-1] # (B, S_tgt-1)
        tgt_output = tgt[:, 1:]  # (B, S_tgt-1)

        # Create masks
        src_padding_mask = create_padding_mask(src, PAD_IDX).to(DEVICE) # (B, 1, S_src)
        tgt_padding_mask = create_padding_mask(tgt_input, PAD_IDX).to(DEVICE) # (B, 1, S_tgt-1)
        tgt_look_ahead_mask = generate_square_subsequent_mask(tgt_input.size(1), device=DEVICE) # (S_tgt-1, S_tgt-1)
        
        # Combine target masks: (B, S_tgt-1, S_tgt-1)
        # Ensure masks are boolean for logical operations
        combined_tgt_mask = (tgt_padding_mask.bool() & tgt_look_ahead_mask.bool()).to(DEVICE)

        optimizer.zero_grad()
        preds = model(src, tgt_input, src_padding_mask, combined_tgt_mask) # (B, S_tgt-1, V_tgt)
        
        # Reshape for CrossEntropyLoss: (N, C) where N = B * (S_tgt-1), C = V_tgt
        loss = criterion(preds.reshape(-1, preds.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()

        if (i + 1) % 50 == 0: # Log every 50 batches
            avg_batch_loss = loss.item()
            print(f"Epoch [{current_epoch+1}/{total_epochs}], Batch [{i+1}/{len(dataloader)}], "
                  f"LR: {new_lr:.2e}, Batch Loss: {avg_batch_loss:.4f}")
            mlflow.log_metric("train_batch_loss", avg_batch_loss, step=step_num_)
            mlflow.log_metric("learning_rate", new_lr, step=step_num_)

    epoch_loss = total_loss / len(dataloader)
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch [{current_epoch+1}/{total_epochs}] completed. Train Loss: {epoch_loss:.4f}. Duration: {epoch_duration:.2f}s")
    return epoch_loss, epoch_duration

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_padding_mask = create_padding_mask(src, PAD_IDX).to(DEVICE)
            tgt_padding_mask = create_padding_mask(tgt_input, PAD_IDX).to(DEVICE)
            tgt_look_ahead_mask = generate_square_subsequent_mask(tgt_input.size(1), device=DEVICE)
            # Ensure masks are boolean for logical operations
            combined_tgt_mask = (tgt_padding_mask.bool() & tgt_look_ahead_mask.bool()).to(DEVICE)

            preds = model(src, tgt_input, src_padding_mask, combined_tgt_mask)
            loss = criterion(preds.reshape(-1, preds.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

# --- MLflow Setup ---
try:
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
except mlflow.exceptions.MlflowException as e:
    print(f"MLflow setup error: {e}. Ensure MLflow server is running or configured correctly.")
    # Fallback if MLflow is not available or experiment creation fails
    class DummyMLflow:
        def __init__(self):
            self.experiment_id = "local_run"
        def start_run(self, experiment_id=None, run_name=None): return self
        def __enter__(self): return self
        def __exit__(self, type, value, traceback): pass
        def log_param(self, key, value): print(f"MLflow (dummy): Param {key}={value}")
        def log_metric(self, key, value, step=None): print(f"MLflow (dummy): Metric {key}={value} at step {step}")
        def pytorch_log_model(self, model, artifact_path): print(f"MLflow (dummy): Model logged to {artifact_path}")
        def end_run(self): print("MLflow (dummy): Run ended")
    mlflow = DummyMLflow()
    experiment_id = mlflow.experiment_id

# --- Main Training Loop ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Source Language: {SRC_LANGUAGE}, Target Language: {TRG_LANGUAGE}")
    print(f"Model Hyperparameters: d_model={D_MODEL}, heads={NUM_HEADS}, enc_layers={NUM_ENCODER_LAYERS}, dec_layers={NUM_DECODER_LAYERS}, d_ff={D_FF}")
    print(f"Training Hyperparameters: epochs={EPOCHS}, batch_size={BATCH_SIZE}, warmup_steps={WARMUP_STEPS}")

    with mlflow.start_run(experiment_id=experiment_id, run_name="Transformer_TrainingRun") as run:
        mlflow.log_param("src_language", SRC_LANGUAGE)
        mlflow.log_param("tgt_language", TRG_LANGUAGE)
        mlflow.log_param("d_model", D_MODEL)
        mlflow.log_param("num_encoder_layers", NUM_ENCODER_LAYERS)
        mlflow.log_param("num_decoder_layers", NUM_DECODER_LAYERS)
        mlflow.log_param("num_heads", NUM_HEADS)
        mlflow.log_param("d_ff", D_FF)
        mlflow.log_param("dropout", DROPOUT)
        mlflow.log_param("max_seq_len", MAX_SEQ_LEN)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("initial_learning_rate_adam", LEARNING_RATE) # Adam LR is just initial
        mlflow.log_param("warmup_steps", WARMUP_STEPS)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("src_vocab_size", SRC_VOCAB_SIZE)
        mlflow.log_param("tgt_vocab_size", TGT_VOCAB_SIZE)
        mlflow.log_param("pad_idx", PAD_IDX)
        mlflow.log_param("bos_idx", BOS_IDX)
        mlflow.log_param("eos_idx", EOS_IDX)
        mlflow.log_param("unk_idx", UNK_IDX)

        best_val_loss = float('inf')
        total_training_time = 0

        for epoch in range(EPOCHS):
            print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
            train_loss, epoch_duration = train_epoch(model, train_dataloader, optimizer, criterion, epoch, EPOCHS)
            val_loss = evaluate(model, val_dataloader, criterion)
            total_training_time += epoch_duration

            print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Duration: {epoch_duration:.2f}s")
            mlflow.log_metric("train_epoch_loss", train_loss, step=epoch+1)
            mlflow.log_metric("val_epoch_loss", val_loss, step=epoch+1)
            mlflow.log_metric("epoch_duration_seconds", epoch_duration, step=epoch+1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the best model
                mlflow.pytorch.log_model(model, "best_transformer_model")
                torch.save(model.state_dict(), "transformer_best_model.pt")
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        mlflow.log_metric("total_training_time_seconds", total_training_time)
        mlflow.pytorch.log_model(model, "final_transformer_model") # Log final model
        torch.save(model.state_dict(), "transformer_final_model.pt")
        print("Training finished.")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Total training time: {total_training_time/3600:.2f} hours")

    # TODO: Add BLEU score calculation for benchmarking
    # TODO: Add example translation function
    # TODO: Add Streamlit app integration
