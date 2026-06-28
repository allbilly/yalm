#!/usr/bin/env python3
"""Mistral-7B-Instruct-v0.2 in tinygrad (pip install tinygrad only).

Llama Transformer vendored from tinygrad extra/models/llama.py (MIT).
"""
import argparse
import json
import time
from pathlib import Path
from typing import Union

from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device, Context
from tinygrad.nn.state import load_state_dict, safe_load

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def mistral_weights_dir(model_id: str = DEFAULT_MODEL, local_dir: str | Path | None = None) -> Path:
  import os
  if local_dir is not None:
    return Path(local_dir).expanduser().resolve()
  if p := os.environ.get("MISTRAL_PATH"):
    return Path(p).expanduser().resolve()
  from huggingface_hub import snapshot_download
  try:
    return Path(snapshot_download(model_id, local_files_only=True)).resolve()
  except Exception:
    return Path(snapshot_download(model_id)).resolve()

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)

def complex_mult(A, c, d):
  a, b = A[..., 0:1], A[..., 1:2]
  return (a*c - b*d).cat(a*d + b*c, dim=-1)

def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
  return complex_mult(xq, c, d).flatten(3), complex_mult(xk, c, d).flatten(3)

def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  if n_rep == 1:
    return x
  bs, seqlen, n_kv_heads, head_dim = x.shape
  return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class Attention:
  def __init__(self, dim, n_heads, n_kv_heads, max_context, linear=nn.Linear):
    self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
    self.head_dim = dim // n_heads
    self.n_rep = n_heads // n_kv_heads
    self.max_context = max_context
    self.wq = linear(dim, n_heads * self.head_dim, bias=False)
    self.wk = linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.wv = linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.wo = linear(n_heads * self.head_dim, dim, bias=False)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Tensor | None) -> Tensor:
    xq, xk, xv = self.wq(x), self.wk(x.contiguous_backward()), self.wv(x)
    if x.dtype == dtypes.bfloat16:
      xq, xk = xq.contiguous_backward(), xk.contiguous_backward()
    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    bsz, seqlen, _, _ = xq.shape

    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
    self.cache_kv[:, :, start_pos:start_pos+seqlen, :, :].assign(Tensor.stack(xk, xv)).realize()
    keys = self.cache_kv[0, :, :start_pos+seqlen, :, :]
    values = self.cache_kv[1, :, :start_pos+seqlen, :, :]
    keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
    return self.wo(attn.reshape(bsz, seqlen, -1))

class FeedForward:
  def __init__(self, dim: int, hidden_dim: int, linear=nn.Linear):
    self.w1 = linear(dim, hidden_dim, bias=False)
    self.w2 = linear(hidden_dim, dim, bias=False)
    self.w3 = linear(dim, hidden_dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.w2(self.w1(x).silu() * self.w3(x.contiguous_backward()))

class TransformerBlock:
  def __init__(self, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context, linear=nn.Linear):
    self.attention = Attention(dim, n_heads, n_kv_heads, max_context, linear)
    self.feed_forward = FeedForward(dim, hidden_dim, linear)
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Tensor | None):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    return (h + self.feed_forward(self.ffn_norm(h))).contiguous().contiguous_backward()

def sample(logits: Tensor, temp: float) -> Tensor:
  if temp < 1e-6:
    return logits.argmax()
  logits = (logits != logits).where(-float("inf"), logits)
  return (logits / temp).softmax().multinomial()

class Transformer:
  def __init__(self, dim, hidden_dim, n_heads, n_layers, norm_eps, vocab_size, n_kv_heads, rope_theta, max_context):
    self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context) for _ in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().is_param_(False)
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens: Tensor, start_pos: Union[Variable, int], temperature: float):
    h = self.tok_embeddings(tokens).contiguous()
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, start_pos:start_pos+tokens.shape[1], :, :, :]
    mask = Tensor.full((1, 1, tokens.shape[1], start_pos+tokens.shape[1]), float("-inf"), dtype=h.dtype, device=h.device).triu(start_pos+1) \
      if tokens.shape[1] > 1 else None
    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis, mask)
    logits = self.output(self.norm(h).contiguous().contiguous_backward()).contiguous_backward()
    return sample(logits[:, -1, :].flatten(), temperature)

  def __call__(self, tokens: Tensor, start_pos: int, temperature: float = 0.0):
    if tokens.shape[0:2] == (1, 1) and start_pos:
      sp = Variable("start_pos", 1, self.max_context-1).bind(start_pos)
      return self.forward_jit(tokens, sp, temperature)
    return self.forward(tokens, start_pos, temperature)

def convert_from_huggingface(weights: dict[str, Tensor], n_layers: int, n_heads: int, n_kv_heads: int):
  def permute(v: Tensor, heads: int):
    return v.reshape(heads, 2, v.shape[0] // heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])

  keymap = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v", "o"] for l in range(n_layers)},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight"
       for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(n_layers)},
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
  }
  sd = {}
  for k, v in weights.items():
    if ".rotary_emb." in k:
      continue
    v = v.to(Device.DEFAULT)
    if "q_proj" in k:
      v = permute(v, n_heads)
    elif "k_proj" in k:
      v = permute(v, n_kv_heads)
    sd[keymap[k]] = v
  if "output.weight" not in sd:
    sd["output.weight"] = sd["tok_embeddings.weight"]
  return sd

def load_hf_weights(path: str) -> dict:
  if path.endswith(".index.json"):
    with open(path) as fp:
      weight_map = json.load(fp)["weight_map"]
    parts = {n: safe_load(str(Path(path).parent / n)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  return safe_load(path)


def main() -> None:
  p = argparse.ArgumentParser(description="Run Mistral-7B-Instruct-v0.2 in tinygrad")
  p.add_argument("--count", type=int, default=30)
  p.add_argument("--temperature", type=float, default=0.7)
  p.add_argument("--prompt", default="Q: What is the meaning of life?")
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--weights", default=None)
  args = p.parse_args()
  weights_dir = mistral_weights_dir(args.model, local_dir=args.weights)

  raw = load_hf_weights(str(weights_dir / "model.safetensors.index.json"))
  weights = {k: v.cast(dtypes.float16) if v.dtype == dtypes.bfloat16 else v for k, v in
             convert_from_huggingface(raw, 32, 32, 8).items()}
  model = Transformer(4096, 14336, 32, 32, 1e-5, 32000, 8, 1000000, 4096)
  with Context(BEAM=0):
    load_state_dict(model, weights, strict=False, consume=True)

  from sentencepiece import SentencePieceProcessor
  spp = SentencePieceProcessor(model_file=str(weights_dir / "tokenizer.model"))
  toks = spp.encode(args.prompt)
  start_pos = 0
  for tok in toks:
    model(Tensor([[tok]]), start_pos, args.temperature).realize()
    start_pos += 1

  t0 = time.time()
  for _ in range(args.count):
    tok = model(Tensor([[toks[-1]]]), start_pos, args.temperature).item()
    toks.append(tok)
    start_pos += 1
  gen_time = time.time() - t0

  print(spp.decode(toks))
  print(f"{args.count / gen_time:.2f} tok/s ({gen_time:.2f}s for {args.count} tokens)")


if __name__ == "__main__":
  main()
