import streamlit as st
import torch
import heapq # For managing hypotheses in beam search
from huggingface_hub import hf_hub_download # Added
import json # Added
import os # Added

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(layout="wide")

# --- Configuration (Previously from train.py, now from HF or defaults) ---
# These will be loaded from config.json or set to defaults if not found
DEFAULT_D_MODEL = 512
DEFAULT_NUM_ENCODER_LAYERS = 6
DEFAULT_NUM_DECODER_LAYERS = 6
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FF = 2048
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_SEQ_LEN = 128 # Or a sensible default
DEFAULT_SRC_LANGUAGE = "de"
DEFAULT_TRG_LANGUAGE = "en"
DEFAULT_PAD_IDX = 1 # Common default, ensure consistency
DEFAULT_BOS_IDX = 2
DEFAULT_EOS_IDX = 3
DEFAULT_UNK_IDX = 0


# --- Hugging Face Repository Details ---
HF_REPO_ID = "posity/attention_is_all_you_need_de_en_wmt14"
MODEL_FILENAME = "model.py"
CONFIG_FILENAME = "config.json"
VOCAB_SRC_FILENAME = "vocab_src_wmt14.json"
VOCAB_TGT_FILENAME = "vocab_tgt_wmt14.json"

# --- Load Configuration from Hugging Face ---
@st.cache_resource
def load_hf_config():
    try:
        config_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CONFIG_FILENAME)
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Fill in defaults if any keys are missing (for robustness)
        config['d_model'] = config.get('d_model', DEFAULT_D_MODEL)
        config['num_encoder_layers'] = config.get('num_encoder_layers', DEFAULT_NUM_ENCODER_LAYERS)
        config['num_decoder_layers'] = config.get('num_decoder_layers', DEFAULT_NUM_DECODER_LAYERS)
        config['num_heads'] = config.get('num_heads', DEFAULT_NUM_HEADS)
        config['d_ff'] = config.get('d_ff', DEFAULT_D_FF)
        config['dropout'] = config.get('dropout', DEFAULT_DROPOUT)
        config['max_seq_len'] = config.get('max_seq_len', DEFAULT_MAX_SEQ_LEN)
        config['src_language'] = config.get('src_language', DEFAULT_SRC_LANGUAGE)
        config['tgt_language'] = config.get('tgt_language', DEFAULT_TRG_LANGUAGE)
        config['pad_idx'] = config.get('pad_idx', DEFAULT_PAD_IDX)
        config['bos_idx'] = config.get('bos_idx', DEFAULT_BOS_IDX)
        config['eos_idx'] = config.get('eos_idx', DEFAULT_EOS_IDX)
        config['unk_idx'] = config.get('unk_idx', DEFAULT_UNK_IDX)
        config['src_vocab_size'] = config.get('src_vocab_size') # Should be in config
        config['tgt_vocab_size'] = config.get('tgt_vocab_size') # Should be in config
        if config['src_vocab_size'] is None or config['tgt_vocab_size'] is None:
            st.error("src_vocab_size or tgt_vocab_size missing in config.json from Hugging Face.")
            return None
        print("Successfully loaded config.json from Hugging Face.")
        return config
    except Exception as e:
        st.error(f"Error loading config.json from Hugging Face: {e}")
        # Fallback to defaults if HF load fails
        return {
            'd_model': DEFAULT_D_MODEL, 'num_encoder_layers': DEFAULT_NUM_ENCODER_LAYERS,
            'num_decoder_layers': DEFAULT_NUM_DECODER_LAYERS, 'num_heads': DEFAULT_NUM_HEADS,
            'd_ff': DEFAULT_D_FF, 'dropout': DEFAULT_DROPOUT, 'max_seq_len': DEFAULT_MAX_SEQ_LEN,
            'src_language': DEFAULT_SRC_LANGUAGE, 'tgt_language': DEFAULT_TRG_LANGUAGE,
            'pad_idx': DEFAULT_PAD_IDX, 'bos_idx': DEFAULT_BOS_IDX, 'eos_idx': DEFAULT_EOS_IDX,
            'unk_idx': DEFAULT_UNK_IDX,
            # These would ideally be known or also have defaults, but model loading will fail
            'src_vocab_size': 50000, # Placeholder, will cause issues if not correct
            'tgt_vocab_size': 50000  # Placeholder
        }

hf_config = load_hf_config()

# Use loaded config values, falling back to defaults if hf_config is None (error case)
D_MODEL = hf_config.get('d_model') if hf_config else DEFAULT_D_MODEL
NUM_ENCODER_LAYERS = hf_config.get('num_encoder_layers') if hf_config else DEFAULT_NUM_ENCODER_LAYERS
NUM_DECODER_LAYERS = hf_config.get('num_decoder_layers') if hf_config else DEFAULT_NUM_DECODER_LAYERS
NUM_HEADS = hf_config.get('num_heads') if hf_config else DEFAULT_NUM_HEADS
D_FF = hf_config.get('d_ff') if hf_config else DEFAULT_D_FF
DROPOUT = hf_config.get('dropout') if hf_config else DEFAULT_DROPOUT
MAX_SEQ_LEN = hf_config.get('max_seq_len') if hf_config else DEFAULT_MAX_SEQ_LEN
SRC_LANGUAGE = hf_config.get('src_language') if hf_config else DEFAULT_SRC_LANGUAGE
TRG_LANGUAGE = hf_config.get('tgt_language') if hf_config else DEFAULT_TRG_LANGUAGE
PAD_IDX = hf_config.get('pad_idx') if hf_config else DEFAULT_PAD_IDX
BOS_IDX = hf_config.get('bos_idx') if hf_config else DEFAULT_BOS_IDX
EOS_IDX = hf_config.get('eos_idx') if hf_config else DEFAULT_EOS_IDX
UNK_IDX = hf_config.get('unk_idx') if hf_config else DEFAULT_UNK_IDX
SRC_VOCAB_SIZE = hf_config.get('src_vocab_size') if hf_config else None
TGT_VOCAB_SIZE = hf_config.get('tgt_vocab_size') if hf_config else None


# --- Load Vocabularies from Hugging Face ---
@st.cache_resource
def load_hf_vocabs():
    try:
        vocab_src_path = hf_hub_download(repo_id=HF_REPO_ID, filename=VOCAB_SRC_FILENAME)
        vocab_tgt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=VOCAB_TGT_FILENAME)
        
        # Load JSON vocabularies
        with open(vocab_src_path, 'r') as f:
            vocab_src_data = json.load(f)
        with open(vocab_tgt_path, 'r') as f:
            vocab_tgt_data = json.load(f)
        
        # Create simple vocabulary classes (like app_local.py)
        class SimpleVocab:
            def __init__(self, vocab_data):
                self.stoi = vocab_data['stoi']
                self.itos = vocab_data['itos']  # This is a list, not a dict
                
            def __getitem__(self, token):
                return self.stoi.get(token, self.stoi.get('<unk>', UNK_IDX))
            
            def __call__(self, tokens):
                return [self.__getitem__(token) for token in tokens]
            
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
            
            def get_default_index(self):
                return self.stoi.get('<unk>', UNK_IDX)
        
        vocab_src = SimpleVocab(vocab_src_data)
        vocab_tgt = SimpleVocab(vocab_tgt_data)
        
        # The loaded objects are custom vocab instances
        # We need to wrap them in a dictionary as expected by the rest of the code
        loaded_vocab_transform = {
            SRC_LANGUAGE: vocab_src,
            TRG_LANGUAGE: vocab_tgt
        }
        print("Successfully loaded vocabularies from Hugging Face.")
        return loaded_vocab_transform
    except Exception as e:
        st.error(f"Error loading vocabularies from Hugging Face: {e}")
        return None

vocab_transform = load_hf_vocabs()

# --- Tokenizer (Simple tokenization like app_local.py) ---
def simple_tokenize(text):
    """Simple tokenization by splitting on whitespace and punctuation"""
    import re
    # Split on whitespace and common punctuation, keep tokens
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens

# Create simple token transform
if vocab_transform: # Only create token_transform if vocabs loaded
    token_transform = {
        SRC_LANGUAGE: simple_tokenize,
        TRG_LANGUAGE: simple_tokenize
    }
else: # Fallback if vocab loading failed
    token_transform = None
    st.error("Tokenizers could not be initialized because vocabularies failed to load.")


from model import Transformer, generate_square_subsequent_mask, create_padding_mask
# Model architecture is loaded from local model.py (same as deployed to HF)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model from Hugging Face ---
@st.cache_resource 
def load_model_from_hf():
    if not hf_config or not SRC_VOCAB_SIZE or not TGT_VOCAB_SIZE:
        st.error("Cannot load model: Configuration from Hugging Face is missing or incomplete (vocab sizes).")
        return None

    # Download model.py from HuggingFace to get the model architecture
    try:
        model_py_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        print(f"Downloaded model.py from HuggingFace: {model_py_path}")
    except Exception as e:
        st.error(f"Error downloading model.py from Hugging Face: {e}")
        return None

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

    # Since we don't have a pre-trained model checkpoint, initialize with random weights
    # In a real scenario, you would load actual trained weights here
    model.eval()
    print(f"Model architecture loaded from Hugging Face: {HF_REPO_ID}")
    st.warning("Note: This model uses randomly initialized weights. For actual translation, you would need to load trained model weights.")
    return model

transformer_model = load_model_from_hf() # Changed from load_model()

# --- Translation Function (Simplified like app_local.py) ---
def translate_sentence(model, sentence: str, src_lang: str, tgt_lang: str, 
                       max_length: int = 50) -> str:
    if model is None:
        return "Model not loaded."
    if vocab_transform is None or token_transform is None:
        return "Vocabularies or tokenizers not loaded."
    
    model.eval()

    # Tokenize and numericalize source sentence
    if src_lang not in token_transform or src_lang not in vocab_transform:
        return f"Tokenizer or vocabulary not available for source language: {src_lang}"
    if tgt_lang not in vocab_transform:
         return f"Vocabulary not available for target language: {tgt_lang}"

    src_tokens = token_transform[src_lang](sentence.rstrip("\n"))
    
    # Convert to indices using vocabulary (like app_local.py)
    src_indices = [BOS_IDX]
    for token in src_tokens:
        try:
            idx = vocab_transform[src_lang][token]
        except KeyError:
            idx = UNK_IDX  # Use unknown token
        src_indices.append(idx)
    src_indices.append(EOS_IDX)

    # Create source tensor
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(DEVICE)  # (1, src_len)
    
    # Create source padding mask
    src_padding_mask = create_padding_mask(src_tensor, PAD_IDX).to(DEVICE)

    with torch.no_grad():
        # Encode source
        memory = model.encode(src_tensor, src_padding_mask)
        
        # Start with BOS token for target
        tgt_tokens = [BOS_IDX]
        
        for _ in range(max_length):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(DEVICE)
            tgt_mask = generate_square_subsequent_mask(len(tgt_tokens), device=DEVICE)
            
            # Decode
            decoder_output = model.decode(tgt_tensor, memory, tgt_mask, src_padding_mask)
            
            # Get next token prediction
            next_token_logits = model.generator(decoder_output[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1).item()
            
            tgt_tokens.append(next_token)
            
            # Stop if EOS token is generated
            if next_token == EOS_IDX:
                break
    
    # Convert indices back to tokens
    output_tokens = tgt_tokens[1:]  # Remove BOS
    if output_tokens and output_tokens[-1] == EOS_IDX:
        output_tokens = output_tokens[:-1]  # Remove EOS
        
    # Convert to text
    translated_tokens = vocab_transform[tgt_lang].lookup_tokens(output_tokens)
    final_translation = " ".join(translated_tokens)
    
    return final_translation

# --- Streamlit UI ---
st.title("Transformer Machine Translation (Attention Is All You Need)")
# Dynamically display languages from loaded config
st.write(f"Translate from **{SRC_LANGUAGE.upper()}** to **{TRG_LANGUAGE.upper()}**")
st.write("Based on the paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)")
st.markdown(f"Model loaded from Hugging Face Hub: [{HF_REPO_ID}](https://huggingface.co/{HF_REPO_ID})")


if transformer_model is None:
    st.warning(f"Model could not be loaded from Hugging Face repo {HF_REPO_ID}. Please check the logs for errors.")
else:
    st.sidebar.header("Model Details (from Hugging Face)")
    st.sidebar.markdown(f"**Source Language:** {SRC_LANGUAGE.upper()}")
    st.sidebar.markdown(f"**Target Language:** {TRG_LANGUAGE.upper()}")
    st.sidebar.markdown(f"**d_model:** {D_MODEL}")
    st.sidebar.markdown(f"**Heads:** {NUM_HEADS}")
    st.sidebar.markdown(f"**Encoder Layers:** {NUM_ENCODER_LAYERS}")
    st.sidebar.markdown(f"**Decoder Layers:** {NUM_DECODER_LAYERS}")
    st.sidebar.markdown(f"**d_ff:** {D_FF}")
    st.sidebar.markdown(f"**Max Sequence Length:** {MAX_SEQ_LEN}")
    st.sidebar.markdown(f"**Device:** {DEVICE}")

    st.header("Try it out!")
    default_text = "Eine Katze saß auf der Matte."
    source_text = st.text_area(f"Enter {SRC_LANGUAGE.upper()} text to translate:", default_text, height=100)

    if st.button("Translate"): 
        if source_text.strip():
            with st.spinner("Translating..."):
                translation = translate_sentence(transformer_model, source_text, SRC_LANGUAGE, TRG_LANGUAGE)
            st.subheader("Translation Result:")
            st.success(translation)
        else:
            st.warning("Please enter some text to translate.")

    st.markdown("---")
    st.subheader("Example Sentences (from Multi30k validation set)")
    example_sentences_de = [
        "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.",
        "Mehrere Männer mit Helmen bedienen ein riesiges Maschinensystem.",
        "Ein Mann in einem blauen Hemd schläft auf einer Couch, während ein anderer Mann eine Zeitung liest.",
        "Ein Junge springt über ein Hindernis auf einem Skateboard."
    ]

    for i, ex_de in enumerate(example_sentences_de):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Example {i+1} ({SRC_LANGUAGE.upper()}):**")
            st.write(ex_de)
        with col2:
            if st.button(f"Translate Example {i+1}", key=f"ex_btn_{i}"):
                with st.spinner("Translating example..."):
                    ex_translation = translate_sentence(transformer_model, ex_de, SRC_LANGUAGE, TRG_LANGUAGE)
                st.markdown(f"**Translation ({TRG_LANGUAGE.upper()}):**")
                st.info(ex_translation)

st.markdown("---")
st.write("Note: This model is trained on the Multi30k dataset, hosted on Hugging Face.")
st.write(f"Ensure the repository {HF_REPO_ID} contains the necessary model, config, and vocabulary files.")

# To run: streamlit run app_fixed.py
