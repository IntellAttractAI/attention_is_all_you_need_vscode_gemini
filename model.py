\
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding.
    As described in section 3.5 of "Attention Is All You Need".
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (d_model/2)
        
        pe = torch.zeros(max_len, 1, d_model) # (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Not a model parameter, but should be part of state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention.
    As described in section 3.2.1 of "Attention Is All You Need".
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    def __init__(self): # Removed dropout parameter
        super().__init__()
        # No self.dropout layer here

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # (batch, num_heads, seq_len_q, seq_len_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e9) # Fill with a very small number where mask is False
            
        p_attn = F.softmax(scores, dim=-1)
        # No dropout on p_attn directly
            
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    As described in section 3.2.2 of "Attention Is All You Need".
    Projects Q, K, V h times with different linear projections.
    Performs attention in parallel, concatenates, and applies a final linear layer.
    """
    def __init__(self, d_model: int, num_heads: int): # Removed dropout parameter
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention() # No dropout passed
        self.out_linear = nn.Linear(d_model, d_model)
        # Dropout is applied in the main Encoder/Decoder layer after adding residual

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 1) Linear projections
        query = self.query_linear(query) 
        key = self.key_linear(key)     
        value = self.value_linear(value) 
        
        # 2) Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3) Apply attention
        # Reshape mask for broadcasting with attention scores (batch_size, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            if mask.dim() == 2:  # Case: Look-ahead mask (seq_len_q, seq_len_k)
                # Expected shape: (1, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:  # Case: Padding mask (batch_size, 1, seq_len_k) or (batch_size, seq_len_q, seq_len_k)
                # Expected shape: (batch_size, 1, 1, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1) 
            # If mask.dim() == 4, it's assumed to be (batch_size, 1, sq, sk) or (batch_size, num_heads, sq, sk).
            # If it's (batch_size, 1, sq, sk), it will broadcast.
            # If it's (batch_size, num_heads, sq, sk), it matches directly.
            # The error implies a mask like (batch_size, 2, sq, sk) might be occurring if it's 4D.

        x, _ = self.attention(query, key, value, mask=mask) 
        
        # 4) Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k) # (batch_size, seq_len_q, d_model)
        return self.out_linear(x)


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    As described in section 3.3 of "Attention Is All You Need".
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single Encoder layer.
    Consists of a multi-head self-attention mechanism and a position-wise feed-forward network.
    Residual connections and layer normalization are applied.
    As described in section 3.1 of "Attention Is All You Need".
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads) # Removed dropout argument
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention sublayer
        src_attn = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(src_attn)) # Add & Norm: LayerNorm(x + Sublayer(x))
        
        # Feed-forward sublayer
        src_ff = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(src_ff)) # Add & Norm
        return src


class DecoderLayer(nn.Module):
    """
    Single Decoder layer.
    Consists of masked multi-head self-attention, multi-head encoder-decoder attention,
    and a position-wise feed-forward network.
    Residual connections and layer normalization are applied.
    As described in section 3.1 of "Attention Is All You Need".
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads) # Removed dropout argument
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads) # Removed dropout argument
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None) -> torch.Tensor:
        # Masked self-attention sublayer
        tgt_attn = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt_attn)) # Add & Norm
        
        # Encoder-decoder attention sublayer
        # Queries from previous decoder layer (tgt), keys and values from encoder output (memory)
        enc_dec_attn_out = self.enc_dec_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout2(enc_dec_attn_out)) # Add & Norm
        
        # Feed-forward sublayer
        tgt_ff = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(tgt_ff)) # Add & Norm
        return tgt


class Encoder(nn.Module):
    """
    Transformer Encoder: a stack of N identical EncoderLayers.
    """
    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.out_linear.out_features) # d_model

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


class Decoder(nn.Module):
    """
    Transformer Decoder: a stack of N identical DecoderLayers.
    """
    def __init__(self, layer: DecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.out_linear.out_features) # d_model

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Multiply by sqrt(d_model) as per section 3.4
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class Transformer(nn.Module):
    """
    Full Transformer model.
    Follows the encoder-decoder structure with stacked self-attention and FFNs.
    """
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 d_model: int = 512, 
                 num_encoder_layers: int = 6, 
                 num_decoder_layers: int = 6, 
                 num_heads: int = 8, 
                 d_ff: int = 2048, 
                 dropout: float = 0.1,
                 max_seq_len: int = 5000):
        super().__init__()
        
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)
        
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Weight sharing between target embedding and pre-softmax linear transformation (Section 3.4)
        # self.tgt_embedding.embedding.weight = self.generator.weight # This is a common practice
        # However, the paper says "share the same weight matrix between the two embedding layers and the pre-softmax linear transformation"
        # This implies src_embedding, tgt_embedding, and generator share weights if vocabularies are shared.
        # For separate vocabs, it's usually tgt_embedding and generator.
        # Let's assume shared weights between target embedding and generator for now.
        # This needs careful handling if d_model of embedding and generator output dim (vocab_size) are different.
        # The paper's statement "we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation"
        # is typically implemented for models with a shared source-target vocabulary.
        # If src_vocab_size == tgt_vocab_size, then self.src_embedding.embedding.weight can also be tied.
        # For now, I will tie target embedding and generator:
        if d_model == self.tgt_embedding.embedding.embedding_dim and tgt_vocab_size == self.generator.out_features:
             self.generator.weight = self.tgt_embedding.embedding.weight


        self._init_weights()

    def _init_weights(self):
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                src: torch.Tensor,    # (batch_size, src_seq_len)
                tgt: torch.Tensor,    # (batch_size, tgt_seq_len)
                src_mask: torch.Tensor, # (batch_size, 1, src_seq_len) or (batch_size, src_seq_len, src_seq_len)
                tgt_mask: torch.Tensor  # (batch_size, tgt_seq_len, tgt_seq_len) for look-ahead
               ) -> torch.Tensor:
        # src/tgt are token indices, need to be embedded and positionally encoded
        # PyTorch nn.Transformer convention is (seq_len, batch, feature)
        # My current implementation assumes (batch, seq_len, feature) for MHA, FFN
        # Let's adjust to (seq_len, batch, feature) for embeddings and Transformer overall
        # Or adjust the Transformer internal components to use (batch, seq_len, feature)
        # The PositionalEncoding is (seq_len, 1, d_model) and expects (seq_len, batch, d_model)
        # Let's stick to (batch_size, seq_len, d_model) for consistency in my layers and handle transpose at entry/exit if needed.
        # For now, assume input to embedding is (batch, seq_len)
        
        src_emb = self.positional_encoding(self.src_embedding(src).transpose(0,1)).transpose(0,1) # batch, seq, dim
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt).transpose(0,1)).transpose(0,1) # batch, seq, dim

        # Create memory_mask from src_mask.
        # src_mask is typically for padding. (batch_size, 1, src_seq_len)
        # memory_mask for enc_dec_attn should be (batch_size, tgt_seq_len, src_seq_len)
        # If src_mask is (B, 1, S_src), it can be broadcasted.
        # If src_mask is (B, S_src), it needs to be (B, 1, 1, S_src) for MHA.
        # Let's assume src_mask is (B, 1, S_src) for padding on keys.
        # And tgt_mask is (B, S_tgt, S_tgt) for look-ahead + padding.

        memory = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask, src_mask) # src_mask is used as memory_mask
        return self.generator(output) # (batch_size, tgt_seq_len, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.positional_encoding(self.src_embedding(src).transpose(0,1)).transpose(0,1)
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt).transpose(0,1)).transpose(0,1)
        return self.decoder(tgt_emb, memory, tgt_mask, memory_mask)


def generate_square_subsequent_mask(sz: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generates a square mask for the sequence.
    The masked positions are filled with False and unmasked positions are True.
    Useful for preventing attention to future positions in a sequence.
    """
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1) 
    return mask == 0 # True for non-masked, False for masked (upper triangle)

def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Creates a mask for padding tokens.
    Args:
        seq: Tensor of shape (batch_size, seq_len)
        pad_idx: Index of the padding token.
    Returns:
        mask: Tensor of shape (batch_size, 1, 1, seq_len) for MHA, or (batch_size, 1, seq_len)
              True for non-pad tokens, False for pad tokens.
    """
    # seq_pad_mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
    # My MHA expects (batch_size, 1, seq_len_q, seq_len_k) or similar.
    # For self-attention, src_mask is (B, 1, S) or (B, S, S).
    # For encoder-decoder attention, memory_mask is (B, 1, S_src)
    # Let's return (B, 1, S_len)
    mask = (seq != pad_idx).unsqueeze(1) # (batch_size, 1, seq_len)
    return mask


if __name__ == '__main__':
    # Example Usage (Illustrative)
    SRC_VOCAB_SIZE = 1000
    TGT_VOCAB_SIZE = 1200
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    D_FF = 2048
    DROPOUT = 0.1
    MAX_SEQ_LEN = 100
    PAD_IDX = 0 # Example padding index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer_model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    # Dummy input
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12

    src_tokens = torch.randint(1, SRC_VOCAB_SIZE, (batch_size, src_seq_len), device=device) # (B, S_src)
    tgt_tokens = torch.randint(1, TGT_VOCAB_SIZE, (batch_size, tgt_seq_len), device=device) # (B, S_tgt)
    
    # Create masks
    # Source padding mask: (B, 1, S_src) - True for non-pad, False for pad
    src_padding_mask = create_padding_mask(src_tokens, PAD_IDX).to(device) 

    # Target masks
    # Target padding mask: (B, 1, S_tgt)
    tgt_padding_mask = create_padding_mask(tgt_tokens, PAD_IDX).to(device) 
    # Target look-ahead mask: (S_tgt, S_tgt) - True for allowed, False for future
    tgt_look_ahead_mask = generate_square_subsequent_mask(tgt_seq_len, device=device) # (S_tgt, S_tgt)
    
    # Combine target masks: element-wise AND
    # tgt_padding_mask needs to be (B, S_tgt, S_tgt) or broadcastable with (S_tgt, S_tgt)
    # tgt_padding_mask is (B, 1, S_tgt). tgt_look_ahead_mask is (S_tgt, S_tgt)
    # Combined mask should be (B, S_tgt, S_tgt)
    # For MHA, it expects (B, H, S, S) or (B, 1, S, S) or (1, 1, S, S)
    # My MHA adds unsqueeze(1) for head dim. So (B, S, S) is fine.
    combined_tgt_mask = tgt_padding_mask.transpose(1,2) & tgt_look_ahead_mask # (B, S_tgt, 1) & (S_tgt, S_tgt) -> (B, S_tgt, S_tgt)
    
    print(f"src_tokens shape: {src_tokens.shape}")
    print(f"tgt_tokens shape: {tgt_tokens.shape}")
    print(f"src_padding_mask shape: {src_padding_mask.shape}") # (B, 1, S_src)
    print(f"combined_tgt_mask shape: {combined_tgt_mask.shape}") # (B, S_tgt, S_tgt)

    # Forward pass
    # src_mask for encoder self-attention and for encoder-decoder attention (memory_mask)
    # tgt_mask for decoder self-attention
    output = transformer_model(src_tokens, tgt_tokens, src_padding_mask, combined_tgt_mask)
    print(f"Output shape: {output.shape}") # Expected: (batch_size, tgt_seq_len, TGT_VOCAB_SIZE)

    # Test encode and decode methods separately (for inference)
    memory = transformer_model.encode(src_tokens, src_padding_mask)
    print(f"Encoded memory shape: {memory.shape}") # Expected: (batch_size, src_seq_len, d_model)

    # For decode, tgt_tokens would be generated one by one in a loop during inference
    # Here, just showing a single step with the full target sequence for illustration
    decoded_output = transformer_model.decode(tgt_tokens, memory, combined_tgt_mask, src_padding_mask)
    print(f"Decoded output shape: {decoded_output.shape}") # Expected: (batch_size, tgt_seq_len, d_model)
    
    # Check parameter count
    total_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}") # Base model has ~65M params
                                                            # My current calculation might be off due to no vocab.
                                                            # Paper: 65M for base.
                                                            # With vocab 37000 (shared), d_model 512:
                                                            # Embeddings: 37000 * 512 = 18,944,000 (x2 if not shared, or x1 if shared with output)
                                                            # Output layer: 512 * 37000 = 18,944,000
                                                            # If shared: 18.9M.
                                                            # My example SRC_VOCAB_SIZE=1000, TGT_VOCAB_SIZE=1200
                                                            # Src Emb: 1000*512 = 512000
                                                            # Tgt Emb: 1200*512 = 614400
                                                            # Generator: 512*1200 = 614400
                                                            # Total from these: ~1.74M
                                                            # The rest comes from attention and FFN layers.
                                                            # EncoderLayer:
                                                            #   MHA: 4 * (512*512) = 4 * 262144 = 1,048,576
                                                            #   FFN: (512*2048 + 2048) + (2048*512 + 512) = (1048576+2048) + (1048576+512) approx 2.1M
                                                            #   LayerNorms: 2 * (2*512) = 2048
                                                            #   Total per EncoderLayer: ~3.15M
                                                            # DecoderLayer:
                                                            #   MHA1: ~1.05M
                                                            #   MHA2: ~1.05M
                                                            #   FFN: ~2.1M
                                                            #   LayerNorms: 3 * (2*512) = 3072
                                                            #   Total per DecoderLayer: ~4.2M
                                                            # 6 EncoderLayers: 6 * 3.15M = 18.9M
                                                            # 6 DecoderLayers: 6 * 4.2M = 25.2M
                                                            # Total model (without embeddings): 18.9M + 25.2M = 44.1M
                                                            # With example embeddings: 44.1M + 1.74M = ~45.84M
                                                            # This is in the ballpark for a 65M model if vocabs are larger.
                                                            # For WMT'14 En-De (37k vocab):
                                                            # Shared Embeddings + Output Layer: 37000 * 512 = 18.944M
                                                            # Total: 44.1M + 18.944M = ~63.044M. This is close to 65M.
                                                            # The discrepancy might be due to biases or exact LayerNorm params.
