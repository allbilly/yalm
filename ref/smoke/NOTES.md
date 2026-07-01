# ref/ smoke notes — Mistral coverage

Mistral is not natively supported by any of the 4 reference repos. Each smoke
test below exercises what the repo *can* do today, and labels the Mistral gap
explicitly rather than faking a run.

| repo | role | model(s) supported | Mistral? |
|---|---|---|---|
| `flashinfer-ai/flashinfer` | GPU kernel lib (attention/GEMM/MoE/sampling) | library, not a model server | N/A — no model needed; smoke = 1-token decode attention on synthetic Q/K/V |
| `openinfer-project/openinfer` | Rust+CUDA inference engine, OpenAI-compatible `/v1/completions` | Qwen3-4B/8B (default), Qwen3.5-4B, DeepSeek-V2-Lite, DeepSeek-V4-Flash, Kimi-K2-Instruct | no — would need a new `openinfer-mistral-7b` model crate |
| `frankkk96/FlashQwen` | C++/CUDA + Go engine, OpenAI-compatible server | Qwen3-8B only (`Qwen3ForCausalLM`) | no — engine hard-codes Qwen3 arch |
| `zml/zml` (the source of the `llmd` Docker image) | Zig + MLIR + Bazel inference stack | Llama 3.1/3.2, Qwen 3.5, LFM 2.5 | no — would need a Mistral module in `examples/llm` |

`zml/llmd` is a Docker image on Docker Hub (`zmlai/llmd`), not a git repo.
The git source is `zml/zml`; smoke runs against that.

## Why "1 token smoke" doesn't mean "1 token of Mistral"

For a model **server**, "1 token" = generate 1 token from a model. None of
these 4 servers can load Mistral without code changes. So:

- **flashinfer** is not a server — its "1 token" smoke is a single
  `single_decode_with_kv_cache` call on a synthetic query against a synthetic
  KV cache. That's the real "1 token decode" primitive.
- **openinfer / FlashQwen / zml** smoke = "the build pipeline that would host
  Mistral compiles and links". No model load, no 1-token decode (because the
  model loader would refuse the file).
