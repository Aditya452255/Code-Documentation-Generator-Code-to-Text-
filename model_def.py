import math
import os

try:
    import torch
    import torch.nn as nn
    _TORCH_IMPORT_ERROR = None
except Exception as exc:  # covers DLL load issues on Windows
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

# ---------- Positional Encoding ----------
if torch is not None:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

# ---------- Encoder Block ----------
    class EncoderBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            attn_out, _ = self.self_attn(x, x, x)
            x = self.norm1(x + attn_out)
            ffn_out = self.ffn(x)
            return self.norm2(x + ffn_out)

# ---------- Decoder Block ----------
    class DecoderBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

        def forward(self, x, enc_out, tgt_mask):
            attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
            x = self.norm1(x + attn_out)
            attn_out, _ = self.cross_attn(x, enc_out, enc_out)
            x = self.norm2(x + attn_out)
            return self.norm3(x + self.ffn(x))

# ---------- Transformer ----------
    class Transformer(nn.Module):
        def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, pad_id):
            super().__init__()
            self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
            self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
            self.positional_encoding = PositionalEncoding(d_model)
            self.encoder_layers = nn.ModuleList(
                [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
            )
            self.decoder_layers = nn.ModuleList(
                [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
            )
            self.output_layer = nn.Linear(d_model, vocab_size)

        def forward(self, src, tgt, tgt_mask):
            src = self.positional_encoding(self.src_embed(src))
            tgt = self.positional_encoding(self.tgt_embed(tgt))
            for layer in self.encoder_layers:
                src = layer(src)
            for layer in self.decoder_layers:
                tgt = layer(tgt, src, tgt_mask)
            return self.output_layer(tgt)

    def causal_mask(size):
        return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)

else:
    Transformer = None

    def causal_mask(size):
        raise RuntimeError("PyTorch is not available.")

# ---------- LOAD EVERYTHING ----------
def load_all():
    if torch is None:
        raise RuntimeError(
            "PyTorch failed to import. On Windows this is often missing "
            "Microsoft Visual C++ Redistributable or an incompatible torch build. "
            f"Original error: {_TORCH_IMPORT_ERROR}"
        )
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build absolute paths for tokenizer and model
    tokenizer_path = os.path.join(script_dir, "code_doc_tokenizer.json")
    model_path = os.path.join(script_dir, "final_model.pt")
    checkpoint_path = os.path.join(script_dir, "checkpoint.pt")
    
    # Use checkpoint if final_model doesn't exist
    if not os.path.exists(model_path) and os.path.exists(checkpoint_path):
        model_path = checkpoint_path
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = ByteLevel()

    model = Transformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        pad_id=tokenizer.token_to_id("<pad>")
    )
    
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return tokenizer, model

# ---------- GENERATE ----------
if torch is not None:
    @torch.no_grad()
    def generate(code, tokenizer, model, max_len=128):
        bos = tokenizer.token_to_id("<bos>")
        eos = tokenizer.token_to_id("<eos>")

        src = [bos] + tokenizer.encode(code).ids[:max_len-2] + [eos]
        src = torch.tensor(src).unsqueeze(0)

        out = [bos]
        for _ in range(max_len):
            tgt = torch.tensor(out).unsqueeze(0)
            mask = causal_mask(tgt.size(1))
            logits = model(src, tgt, mask)
            logits_step = logits[0, -1]
            
            values, indices = torch.topk(logits_step, k=20)
            probs = torch.softmax(values, dim=-1)
            
            next_id = indices[torch.multinomial(probs, 1)].item()
            if next_id == eos:
                break
            out.append(next_id)

        return tokenizer.decode(out[1:], skip_special_tokens=True)
else:
    def generate(*_args, **_kwargs):
        raise RuntimeError("PyTorch is not available.")
