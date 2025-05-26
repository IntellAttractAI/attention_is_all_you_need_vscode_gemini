import unittest
import torch
from model import (
    PositionalEncoding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionwiseFeedForward,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
    TokenEmbedding,
    Transformer,
    generate_square_subsequent_mask,
    create_padding_mask
)

class TestTransformerComponents(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.dropout = 0.1
        self.max_seq_len = 100
        self.batch_size = 2
        self.seq_len = 10
        self.vocab_size = 1000
        self.pad_idx = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Common input tensor for many tests
        self.test_tensor = torch.rand(self.batch_size, self.seq_len, self.d_model).to(self.device)
        self.test_tokens = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)

    def test_positional_encoding(self):
        pe = PositionalEncoding(self.d_model, self.dropout, self.max_seq_len).to(self.device)
        # Input to PE is (seq_len, batch_size, d_model)
        x = torch.zeros(self.seq_len, self.batch_size, self.d_model).to(self.device)
        encoded_x = pe(x)
        self.assertEqual(encoded_x.shape, (self.seq_len, self.batch_size, self.d_model))
        # Check if something was added
        self.assertFalse(torch.allclose(encoded_x, torch.zeros_like(encoded_x)))

    def test_scaled_dot_product_attention(self):
        attention = ScaledDotProductAttention().to(self.device) # Removed dropout argument
        query = torch.rand(self.batch_size, self.num_heads, self.seq_len, self.d_model // self.num_heads).to(self.device)
        key = torch.rand(self.batch_size, self.num_heads, self.seq_len, self.d_model // self.num_heads).to(self.device)
        value = torch.rand(self.batch_size, self.num_heads, self.seq_len, self.d_model // self.num_heads).to(self.device)
        
        output, attn_weights = attention(query, key, value)
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        self.assertTrue(torch.all(attn_weights >= 0) and torch.all(attn_weights <= 1))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1))))

        # Test with mask
        mask = (torch.rand(self.batch_size, 1, self.seq_len, self.seq_len) > 0.5).to(self.device)
        output_masked, _ = attention(query, key, value, mask=mask)
        self.assertEqual(output_masked.shape, query.shape)

    def test_multi_head_attention(self):
        mha = MultiHeadAttention(self.d_model, self.num_heads).to(self.device) # Removed dropout argument
        output = mha(self.test_tensor, self.test_tensor, self.test_tensor)
        self.assertEqual(output.shape, self.test_tensor.shape)

        # Test with mask
        mask = create_padding_mask(self.test_tokens, self.pad_idx).to(self.device) # (B, 1, S)
        output_masked = mha(self.test_tensor, self.test_tensor, self.test_tensor, mask=mask)
        self.assertEqual(output_masked.shape, self.test_tensor.shape)

    def test_positionwise_feed_forward(self):
        ffn = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout).to(self.device)
        output = ffn(self.test_tensor)
        self.assertEqual(output.shape, self.test_tensor.shape)

    def test_encoder_layer(self):
        encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout).to(self.device)
        src_mask = create_padding_mask(self.test_tokens, self.pad_idx).to(self.device) # (B, 1, S)
        output = encoder_layer(self.test_tensor, src_mask=src_mask)
        self.assertEqual(output.shape, self.test_tensor.shape)

    def test_decoder_layer(self):
        decoder_layer = DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout).to(self.device)
        memory = torch.rand(self.batch_size, self.seq_len, self.d_model).to(self.device)
        
        tgt_tokens = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len - 2), device=self.device)
        tgt_tensor = torch.rand(self.batch_size, self.seq_len - 2, self.d_model).to(self.device)

        tgt_padding_mask = create_padding_mask(tgt_tokens, self.pad_idx).to(self.device) # (B, 1, S_tgt)
        tgt_look_ahead_mask = generate_square_subsequent_mask(tgt_tensor.size(1), device=self.device) # (S_tgt, S_tgt)
        # Ensure masks are boolean before bitwise AND and on the correct device
        combined_tgt_mask = (tgt_padding_mask.bool() & tgt_look_ahead_mask.bool()).to(self.device)

        memory_mask = create_padding_mask(self.test_tokens, self.pad_idx).to(self.device) # (B, 1, S_src)

        output = decoder_layer(tgt_tensor, memory, tgt_mask=combined_tgt_mask, memory_mask=memory_mask)
        self.assertEqual(output.shape, tgt_tensor.shape)

    def test_token_embedding(self):
        embedding = TokenEmbedding(self.vocab_size, self.d_model).to(self.device)
        embedded_tokens = embedding(self.test_tokens)
        self.assertEqual(embedded_tokens.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_encoder(self):
        num_layers = 6
        encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        encoder = Encoder(encoder_layer, num_layers).to(self.device)
        src_mask = create_padding_mask(self.test_tokens, self.pad_idx).to(self.device)
        output = encoder(self.test_tensor, src_mask=src_mask)
        self.assertEqual(output.shape, self.test_tensor.shape)

    def test_decoder(self):
        num_layers = 6
        decoder_layer = DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        decoder = Decoder(decoder_layer, num_layers).to(self.device)
        memory = torch.rand(self.batch_size, self.seq_len, self.d_model).to(self.device)
        
        tgt_tokens = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len - 2), device=self.device)
        tgt_tensor = torch.rand(self.batch_size, self.seq_len - 2, self.d_model).to(self.device)

        tgt_padding_mask = create_padding_mask(tgt_tokens, self.pad_idx).to(self.device)
        tgt_look_ahead_mask = generate_square_subsequent_mask(tgt_tensor.size(1), device=self.device)
        # Ensure masks are boolean before bitwise AND and on the correct device
        combined_tgt_mask = (tgt_padding_mask.bool() & tgt_look_ahead_mask.bool()).to(self.device)

        memory_mask = create_padding_mask(self.test_tokens, self.pad_idx).to(self.device)

        output = decoder(tgt_tensor, memory, tgt_mask=combined_tgt_mask, memory_mask=memory_mask)
        self.assertEqual(output.shape, tgt_tensor.shape)

    def test_transformer_model_forward(self):
        src_vocab_size = self.vocab_size
        tgt_vocab_size = self.vocab_size + 200 # Different target vocab
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=self.d_model,
            num_encoder_layers=2, # Using fewer layers for faster test
            num_decoder_layers=2,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len
        ).to(self.device)

        src_tokens = torch.randint(1, src_vocab_size, (self.batch_size, self.seq_len), device=self.device)
        tgt_tokens = torch.randint(1, tgt_vocab_size, (self.batch_size, self.seq_len -1), device=self.device) # Decoder input

        src_padding_mask = create_padding_mask(src_tokens, self.pad_idx).to(self.device)
        
        tgt_padding_mask = create_padding_mask(tgt_tokens, self.pad_idx).to(self.device)
        tgt_look_ahead_mask = generate_square_subsequent_mask(tgt_tokens.size(1), device=self.device)
        # Ensure masks are boolean before bitwise AND and on the correct device
        combined_tgt_mask = (tgt_padding_mask.bool() & tgt_look_ahead_mask.bool()).to(self.device)

        output = model(src_tokens, tgt_tokens, src_padding_mask, combined_tgt_mask)
        self.assertEqual(output.shape, (self.batch_size, tgt_tokens.size(1), tgt_vocab_size))

    def test_generate_square_subsequent_mask(self):
        sz = 5
        mask = generate_square_subsequent_mask(sz, device=self.device)
        self.assertEqual(mask.shape, (sz, sz))
        self.assertTrue(torch.all(torch.diag(mask) == True)) # Diagonal should be unmasked (True)
        self.assertTrue(torch.all(torch.triu(mask, diagonal=1) == False)) # Upper triangle should be masked (False)
        # For my ScaledDotProductAttention, mask==0 means masked_fill. So False means masked.
        # The generate_square_subsequent_mask returns True for keep, False for mask.
        # So this is correct for direct use with `masked_fill(mask == 0, -1e9)`
        # Let's verify the values for my ScaledDotProductAttention logic
        # My ScaledDotProductAttention: scores = scores.masked_fill(mask == 0, -1e9)
        # So, mask == 0 means it will be filled with -1e9 (masked out)
        # generate_square_subsequent_mask returns True for keep, False for mask.
        # So, if mask is False, it means mask == 0 is True, so it gets masked. Correct.
        # Example: for sz=3, mask should be:
        # [[ True, False, False],
        #  [ True,  True, False],
        #  [ True,  True,  True]]
        expected_mask = torch.tensor([
            [True, False, False, False, False],
            [True, True,  False, False, False],
            [True, True,  True,  False, False],
            [True, True,  True,  True,  False],
            [True, True,  True,  True,  True]
        ], device=self.device, dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected_mask))


    def test_create_padding_mask(self):
        seq = torch.tensor([[1, 2, self.pad_idx, 4, self.pad_idx]], device=self.device)
        mask = create_padding_mask(seq, self.pad_idx)
        # Expected: (batch_size, 1, seq_len)
        # True for non-pad, False for pad
        expected_mask = torch.tensor([[[True, True, False, True, False]]], device=self.device, dtype=torch.bool)
        self.assertEqual(mask.shape, (1, 1, seq.size(1)))
        self.assertTrue(torch.equal(mask, expected_mask))


if __name__ == '__main__':
    unittest.main()
