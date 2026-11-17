# Inference

Solo CUDA/C++ decode runtime for **Llama-3.2-1B-Instruct**.

## Model architecture (Llama-3.2-1B)

Dimensions from `weights/config.json`: `vocab=128256`, `d_model=2048`, `n_layers=16`, `n_heads=32`, `n_kv_heads=8`, `head_dim=64`, `intermediate=8192`. Embeddings and `lm_head` are tied.

```mermaid
flowchart TB
  subgraph In["Input"]
    Tok["token ids<br/>shape: [B, S]"]
  end

  Emb["embed_tokens<br/>W: [128256, 2048]<br/>out: [B, S, 2048]"]

  subgraph Blocks["×16 TransformerBlock"]
    direction TB
    AN["input_layernorm RMSNorm<br/>[2048]"]
    Q["q_proj Linear<br/>[2048 → 2048]<br/>reshape: [B, S, 32, 64]"]
    K["k_proj Linear<br/>[2048 → 512]<br/>reshape: [B, S, 8, 64]"]
    V["v_proj Linear<br/>[2048 → 512]<br/>reshape: [B, S, 8, 64]"]
    RoPE["RoPE on Q,K<br/>head_dim=64, θ=500000<br/>llama3 scaling"]
    Attn["GQA attention<br/>32 Q heads, 8 KV heads<br/>scores: [B, 32, S, S_kv]"]
    O["o_proj Linear<br/>[2048 → 2048]"]
    R1["residual +"]
    MN["post_attention_layernorm RMSNorm<br/>[2048]"]
    Gate["gate_proj Linear<br/>[2048 → 8192]"]
    Up["up_proj Linear<br/>[2048 → 8192]"]
    Act["SwiGLU: silu(gate) * up<br/>[B, S, 8192]"]
    Down["down_proj Linear<br/>[8192 → 2048]"]
    R2["residual +"]

    AN --> Q & K & V
    Q --> RoPE
    K --> RoPE
    RoPE --> Attn
    V --> Attn
    Attn --> O --> R1
    AN -.-> R1
    R1 --> MN
    MN --> Gate & Up
    Gate --> Act
    Up --> Act
    Act --> Down --> R2
    MN -.-> R2
  end

  FN["final RMSNorm<br/>[2048]"]
  Head["lm_head = tied embed<br/>[2048 → 128256]<br/>logits: [B, S, 128256]"]

  Tok --> Emb --> Blocks --> FN --> Head
```

**GQA note:** `k_proj`/`v_proj` emit `8 × 64 = 512` channels; each KV head is shared by `32/8 = 4` query heads.

## Engine workflow

```mermaid
flowchart TD
  Main["main.cpp<br/>CLI: weights, prompt, max_new_tokens"]
  Cfg["ModelConfig<br/>parse config.json"]
  WS["WeightStore<br/>load safetensors → GPU Tensors"]
  Eng["Engine<br/>owns model + cache + sampler"]

  Main --> Cfg --> WS --> Eng

  subgraph Prefill["Prefill once"]
    PR["KvCache.reset"]
    PF["LlamaModel.forward_prefill<br/>all prompt tokens"]
    PL["logits at last position"]
    PS["Sampler → first token"]
    PR --> PF --> PL --> PS
  end

  Eng --> Prefill

  subgraph Decode["Decode loop"]
    DF["LlamaModel.forward_decode<br/>single token + KvCache append"]
    DS["Sampler → next token"]
    Stop{"EOS or max_new_tokens?"}
    DF --> DS --> Stop
    Stop -->|no| DF
  end

  Prefill --> Decode
  Stop -->|yes| Out["detokenize + log tok/s"]

  subgraph Model["LlamaModel forward path"]
    direction LR
    M1["embed"] --> M2["TransformerBlock ×16"] --> M3["final norm"] --> M4["lm_head"]
  end

  PF -.-> Model
  DF -.-> Model
```

### Module roles

| Module | Responsibility |
|---|---|
| `ModelConfig` / `WeightStore` | Hyperparams + named GPU weights |
| `LlamaModel` / `TransformerBlock` | Graph wiring |
| `Linear` / `RMSNorm` / `RoPE` / `Attention` / `MLP` | Ops (`kernels/*.cu` + `CublasGemm`) |
| `KvCache` / `Sampler` / `Engine` | State + prefill/decode control |
| `python/scripts/serve_transformers.py` | Reference generate for verification |
