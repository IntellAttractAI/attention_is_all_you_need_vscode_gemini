import streamlit as st
import torch
import json
import os
from model import Transformer, generate_square_subsequent_mask, create_padding_mask

# --- Page Configuration ---
st.set_page_config(layout="wide")

# --- Configuration ---
CONFIG_PATH = "config_local.json"
MODEL_PATH = "transformer_model_wmt14_epoch_3.pth"  # Use the most recent trained model
VOCAB_SRC_PATH = "vocab_src_wmt14.json"  # Use WMT14 vocabularies that match the recent model
VOCAB_TGT_PATH = "vocab_tgt_wmt14.json"

# Default configuration (matching the recent M4-optimized model)
DEFAULT_CONFIG = {
    "d_model": 256,           # Recent model configuration
    "num_encoder_layers": 3,  # Recent model configuration  
    "num_decoder_layers": 3,  # Recent model configuration
    "num_heads": 4,           # Recent model configuration
    "d_ff": 512,             # Recent model configuration
    "dropout": 0.1,
    "max_seq_len": 5000,     # Positional encoding size (from MAX_SEQ_LEN_CONFIG)
    "src_language": "de",
    "tgt_language": "en",
    "pad_idx": 1,
    "bos_idx": 2,
    "eos_idx": 3,
    "unk_idx": 0
}

# --- Load Configuration ---
@st.cache_resource
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return config
    else:
        # Create default config file
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG

config = load_config()

# --- Load Vocabularies ---
@st.cache_resource
def load_vocabularies():
    try:
        # Load JSON vocabularies
        with open(VOCAB_SRC_PATH, 'r') as f:
            vocab_src_data = json.load(f)
        with open(VOCAB_TGT_PATH, 'r') as f:
            vocab_tgt_data = json.load(f)
        
        # Create simple vocabulary classes
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
        
        # Get vocab sizes
        src_vocab_size = len(vocab_src)
        tgt_vocab_size = len(vocab_tgt)
        
        return vocab_transform, src_vocab_size, tgt_vocab_size
    except Exception as e:
        st.error(f"Error loading vocabularies: {e}")
        return None, None, None

vocab_transform, src_vocab_size, tgt_vocab_size = load_vocabularies()

# --- Simple Tokenizer ---
def simple_tokenize(text):
    """Simple tokenization by splitting on whitespace and punctuation"""
    import re
    # Split on whitespace and common punctuation, keep tokens
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens

# --- Device ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file {MODEL_PATH} not found!")
        return None
    
    if src_vocab_size is None or tgt_vocab_size is None:
        st.error("Cannot load model: vocabulary sizes not available")
        return None
    
    try:
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config['d_model'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len']
        ).to(DEVICE)
        
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        
        st.success(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

transformer_model = load_model()

# --- Translation Function ---
def translate_sentence(model, sentence: str, max_length: int = 50):
    if model is None or vocab_transform is None:
        return "Model or vocabularies not loaded."
    
    model.eval()
    src_lang = config['src_language']
    tgt_lang = config['tgt_language']
    
    # Tokenize source sentence
    src_tokens = simple_tokenize(sentence)
    
    # Convert to indices using vocabulary
    src_indices = [config['bos_idx']]
    for token in src_tokens:
        try:
            idx = vocab_transform[src_lang][token]
        except KeyError:
            idx = config['unk_idx']  # Use unknown token
        src_indices.append(idx)
    src_indices.append(config['eos_idx'])
    
    # Create source tensor
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(DEVICE)  # (1, src_len)
    
    # Create source padding mask
    src_padding_mask = create_padding_mask(src_tensor, config['pad_idx']).to(DEVICE)
    
    with torch.no_grad():
        # Encode source
        memory = model.encode(src_tensor, src_padding_mask)
        
        # Start with BOS token for target
        tgt_tokens = [config['bos_idx']]
        
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
            if next_token == config['eos_idx']:
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

# --- Streamlit UI ---
st.title("🔄 Transformer Translation Demo")
st.write(f"**{config['src_language'].upper()}** → **{config['tgt_language'].upper()}** Translation")
st.write("Based on: *Attention Is All You Need* (Vaswani et al., 2017)")

if transformer_model is None:
    st.error("❌ Model could not be loaded. Please check the model files.")
else:
    # Sidebar with model info
    st.sidebar.header("🔧 Model Configuration")
    st.sidebar.write(f"**d_model:** {config['d_model']}")
    st.sidebar.write(f"**Heads:** {config['num_heads']}")
    st.sidebar.write(f"**Encoder Layers:** {config['num_encoder_layers']}")
    st.sidebar.write(f"**Decoder Layers:** {config['num_decoder_layers']}")
    st.sidebar.write(f"**Vocab Size (src):** {src_vocab_size}")
    st.sidebar.write(f"**Vocab Size (tgt):** {tgt_vocab_size}")
    st.sidebar.write(f"**Device:** {DEVICE}")
    
    # Main interface
    st.header("✨ Try Translation")
    
    # Warning about model limitations
    st.warning("⚠️ **Model Limitation Notice**: This model has only been trained for 3 epochs and produces very limited output patterns. This is a demonstration of the transformer architecture rather than a production-ready translation system. For meaningful translations, the model would need significantly more training epochs.")
    
    # Default examples
    default_examples = {
        "de": [
            "Eine Katze sitzt auf dem Tisch.",
            "Der Mann geht zur Arbeit.",
            "Das Wetter ist heute schön.",
            "Ich liebe Machine Learning."
        ]
    }
    
    # Text input
    source_text = st.text_area(
        f"Enter {config['src_language'].upper()} text:",
        value=default_examples["de"][0],
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        translate_btn = st.button("🚀 Translate", type="primary")
    with col2:
        max_len = st.slider("Max output length:", 10, 100, 50)
    
    if translate_btn:
        if source_text.strip():
            with st.spinner("Translating..."):
                translation = translate_sentence(transformer_model, source_text, max_len)
            
            st.subheader("📝 Translation Result")
            st.success(translation)
            
            # Show tokens for debugging
            with st.expander("🔍 Debug Info"):
                src_tokens = simple_tokenize(source_text)
                st.write(f"**Source tokens:** {src_tokens}")
                st.write(f"**Translation:** {translation}")
        else:
            st.warning("⚠️ Please enter some text to translate.")
    
    # Example sentences
    st.markdown("---")
    st.subheader("📚 Example Sentences")
    
    for i, example in enumerate(default_examples["de"]):
        col1, col2, col3 = st.columns([3, 1, 3])
        
        with col1:
            st.write(f"**{config['src_language'].upper()}:** {example}")
        
        with col2:
            if st.button(f"Translate", key=f"ex_{i}"):
                with st.spinner("Translating..."):
                    ex_translation = translate_sentence(transformer_model, example)
                st.session_state[f"translation_{i}"] = ex_translation
        
        with col3:
            if f"translation_{i}" in st.session_state:
                st.write(f"**{config['tgt_language'].upper()}:** {st.session_state[f'translation_{i}']}")

st.markdown("---")
st.write("💡 **Note:** This model was trained on a subset of WMT14 data with reduced parameters for M4 GPU optimization.")
