---
layout: post
title: "Continuous Batching for LLM Inference: A PyTorch Implementation"
date: 2025-01-17 12:00:00 +0000
categories: [Deep Learning, LLM]
tags: [pytorch, inference, optimization, transformers]
toc: true
math: true
comments: true
description: "A deep dive into continuous batching - the technique that powers efficient LLM inference."
---

# Introduction

One day I was reading a lecture about LLM inference frameworks and what optimizations make them fast: dynamic batching, efficient memory management (memory reuse), efficient kernels/fused operations, various model parallellisms (tensor parallel/pipeline parallel inference), quantization, speculative decoding, **KV-cache** and **continuous batching**. After I described continuous batching one of the students asked if there was a simple implementation in pytorch. I knew that digging in the source code of such frameworks as [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang) would be too hard, so I tried looking for some open source implementations. This was before [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), [mini-SGLang](https://github.com/sgl-project/mini-sglang) or support of [continuous batching in transformers](https://huggingface.co/docs/transformers/main/continuous_batching) so I've searched the internet and found a link to a [reference implementation in pytorch](https://inspiringlab.com.np/implementing-continuous-batching-from-scratch-with-pytorch/) which looked __good enough__. I however quickly found that this was not the case: the code did not work. In fact it did not constitute a program: there were no imports, the code referenced classes and functions that were never described anywhere and no matter how you permuted the provided code snippets you could never compose anything that would launch. I guess it could be considered almost a pseudo-code implementation, but this was not something that I was looking for, so I took it upon myself to write a simple implementation in pytorch.

**In this post I will cover:**
- [Background: Autoregressive Generation](#background-autoregressive-generation) — how decoder-only transformers generate text
- [KV-Cache](#kv-cache-avoiding-redundant-computation) — the optimization that makes generation 1000x times faster
- [Prefill vs Decode](#prefill-vs-decode-two-phases-of-generation) — two distinct phases of generation
- [The Problem: Naive Batching](#the-problem-naive-batching-wastes-compute) — why simple batching wastes compute and how continuous batching solves this
- [The Continuous Batching Algorithm](#the-continuous-batching-algorithm) — a PyTorch implementation walkthrough
- [Benchmark Results](#benchmark-results) — ~40% speedup over synchronous batching and comparison to [continuous batching in transformers](https://huggingface.co/docs/transformers/main/continuous_batching)
- [Advanced Topics](#advanced-topics) — preparing you to dive right into chunked prefill and paged attention

The full implementation is available at [github.com/hawkeoni/continuous_batching_pytorch](https://github.com/hawkeoni/continuous_batching_pytorch).

---

# Background: Autoregressive Generation

Before diving into optimizations, let's understand how decoder-only transformers (like GPT, LLaMA, Qwen) generate text.

Unlike models that produce output in one shot, these models generate **autoregressively**—one token at a time, where each new token is conditioned on all previous tokens. The generation loop looks like this:

```
Input: "The capital of France is"

Step 1: Model sees "The capital of France is" → predicts "Paris"
Step 2: Model sees "The capital of France is Paris" → predicts ","
Step 3: Model sees "The capital of France is Paris," → predicts "which"
...and so on until we hit a stopping condition
```

Each step requires a full forward pass through the model. The key insight is that we're repeatedly processing the same prefix tokens over and over—"The capital of France is" gets processed in step 1, then again (along with "Paris") in step 2, and so on.

We can actually avoid recomputing the same thing repeatedly, and that's where KV-cache comes in.

---

# KV-Cache: Avoiding Redundant Computation

To understand the following material you need to have a basic understanding of transformer architecture. I recommend the original [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html), or something more modern like [An even more annotated Transformer](https://pi-tau.github.io/posts/transformer/).

## Why KV-Cache Works

Two key observations make KV-caching possible:

**1. Attention is the only layer where tokens interact.**

All other layers—FFN/MLP, layer norms, embeddings, the final linear layer—operate on each token independently. Only the attention layer requires the full token sequence:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**2. Causal masking means past tokens don't change.**

Because of the causal mask, token representations at position $i$ only depend on tokens at positions $0, 1, ..., i$. Adding a new token at position $i+1$ doesn't change the representations at earlier positions.

This means we can **cache** the Key and Value projections from previous tokens and reuse them when generating new tokens.

![Attention mask](/assets/img/lower_triangle_attention_mask.png){: .normal }
_This is the caption text_
```
[DIAGRAM PLACEHOLDER: Causal Attention Matrix]

Show a lower-triangular attention matrix where:
- Rows = query positions (which token is "asking")
- Columns = key positions (which tokens can be "attended to")
- Highlight that row i only has non-zero values in columns 0..i
- Show that adding a new row doesn't change previous rows
```

## Generation Without KV-Cache (Naive)

Let's walk through a [minimal implementation](https://gist.github.com/hawkeoni/2920d1a2f59840eb673455b40137c73c). First, a simplified transformer that only has the components relevant to KV-caching:

```python
class SimpleCausalAttentionLLM(nn.Module):
    """
    A minimal single-layer causal attention model for educational purposes.

    Real LLMs have multiple layers, multi-head attention, layer norms,
    feed-forward networks, and positional encodings. We omit these because
    they operate on tokens independently and don't affect the KV-cache logic.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
```

The forward pass computes full attention over the sequence:

```python
def forward(self, input_ids: torch.Tensor) -> CausalAttentionOutput:
    # Embed tokens: [batch, seq_len] -> [batch, seq_len, d_model]
    hidden_states = self.embedding(input_ids)

    # Compute Q, K, V projections
    queries = self.W_Q(hidden_states)
    keys = self.W_K(hidden_states)
    values = self.W_V(hidden_states)

    # Attention scores: [batch, seq_len, seq_len]
    attention_scores = torch.matmul(queries, keys.transpose(-2, -1))

    # Apply causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones_like(attention_scores))
    attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))

    # Softmax and apply to values
    attention_weights = torch.softmax(attention_scores, dim=-1)
    context = torch.matmul(attention_weights, values)

    # Project to vocabulary
    logits = self.output_projection(context)

    return CausalAttentionOutput(
        logits=logits,
        k_cache=keys,
        v_cache=values,
    )
```

Generation without cache reprocesses the entire sequence each step:

```python
def generate_without_cache(model, input_ids, num_new_tokens):
    current_ids = input_ids.clone()

    for _ in range(num_new_tokens):
        # Recompute attention over the FULL sequence every time
        outputs = model(current_ids)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_token], dim=1)

    return current_ids
```

**Complexity:** Each step processes a sequence of length $n$, with attention costing $O(n^2)$. Over $N$ generation steps: $O(1^2 + 2^2 + ... + N^2) = O(N^3)$.

## Generation With KV-Cache (Optimized)

With KV-cache, we only compute Q, K, V for the **new token** and reuse cached K, V from previous tokens:

```python
def forward_with_kv_cache(self, input_ids, past_output):
    # Embed only the new token: [batch, 1] -> [batch, 1, d_model]
    hidden_states = self.embedding(input_ids)

    # Compute Q, K, V for the new token only
    new_query = self.W_Q(hidden_states)
    new_key = self.W_K(hidden_states)
    new_value = self.W_V(hidden_states)

    # Extend cache with new K, V
    updated_keys = torch.cat([past_output.k_cache, new_key], dim=1)
    updated_values = torch.cat([past_output.v_cache, new_value], dim=1)

    # New token attends to ALL tokens (no mask needed—it's the last position)
    attention_scores = torch.matmul(new_query, updated_keys.transpose(-2, -1))
    attention_weights = torch.softmax(attention_scores, dim=-1)
    context = torch.matmul(attention_weights, updated_values)

    logits = self.output_projection(context)

    return CausalAttentionOutput(
        logits=logits,
        k_cache=updated_keys,
        v_cache=updated_values,
    )
```

Generation with cache only processes one token per step:

```python
def generate_with_cache(model, input_ids, num_new_tokens):
    generated_ids = input_ids.clone()

    # PREFILL: Process the initial prompt, build KV cache
    outputs = model(input_ids)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # DECODE: Generate tokens one at a time using cached K, V
    for _ in range(num_new_tokens - 1):
        outputs = model.forward_with_kv_cache(next_token, outputs)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    return generated_ids
```

**Complexity:** Prefill is $O(P^2)$ for prompt length $P$. Each decode step is $O(n)$ where $n$ is the current sequence length. Total: $O(P^2 + N \cdot P + N^2) \approx O(N^2)$ for $N >> P$.

This is a **massive improvement**—from cubic to quadratic. For 1000-token generation, that's roughly 1000x faster attention computation.

You can find the complete code [here](https://gist.github.com/hawkeoni/2920d1a2f59840eb673455b40137c73c).

---

# Prefill vs Decode: Two Phases of Generation

The KV-cache optimization naturally divides generation into two distinct phases:

**Prefill Phase:**
- Processes the entire input prompt in one forward pass
- Computes K, V for all prompt tokens simultaneously
- Builds the initial KV cache
- Compute-bound: lots of matrix multiplications
- $O(P^2)$ complexity for prompt length $P$

**Decode Phase:**
- Generates tokens one at a time
- Only computes K, V for the new token
- Extends the KV cache by one position each step
- Memory-bound: small computation, but needs to read entire cache
- $O(n)$ per step

```
[DIAGRAM PLACEHOLDER: Prefill vs Decode Phases]

Show a timeline:
1. Prefill: [=======] Process entire prompt, build KV cache
2. Decode:  [.][.][.][.][.] Generate tokens one by one, extend cache

Visualize KV cache growth:
- After prefill: [████████] (prompt tokens)
- After decode step 1: [████████▓]
- After decode step 2: [████████▓▓]
- ...

█ = prefilled K,V    ▓ = generated K,V
```

This distinction matters for continuous batching because prefill and decode have very different computational characteristics, and we need to manage them carefully.

---

# The Problem: Naive Batching Wastes Compute

Now that we understand KV-cache, let's look at another problem: batching sequences of different lengths.

Modern inference frameworks batch user requests together—serving them one by one would underutilize the GPU. But different requests need different output lengths. Consider batching these two requests:

**Request 1:** "Is Python dynamically typed? Answer yes or no."
**Request 2:** "Explain the difference between TCP and UDP protocols."

Here's how generation proceeds with naive batching:

```
Turn    | Request 1 (short)     | Request 2 (long)
--------|-----------------------|------------------------
  1     | "Yes"                 | "TCP"
  2     | ","                   | "and"
  3     | "Python"              | "UDP"
  4     | "is"                  | "are"
  5     | <EOS> ✓               | "both"
  6     | <padding>             | "transport"
  7     | <padding>             | "layer"
  8     | <padding>             | "protocols"
  ...   | <padding>             | ...
 30     | <padding>             | <EOS> ✓
```

**The problem:** Request 1 finishes at turn 5, but we can't return it until Request 2 completes at turn 30. The GPU slot for Request 1 sits idle, wasting compute on padding.

**The solution:** Swap Request 1 out and bring in a new Request 3:

```
Turn    | Slot A                | Slot B
--------|-----------------------|------------------------
  1     | Req1: "Yes"           | Req2: "TCP"
  2     | Req1: ","             | Req2: "and"
  3     | Req1: "Python"        | Req2: "UDP"
  4     | Req1: "is"            | Req2: "are"
  5     | Req1: <EOS> ✓ DONE    | Req2: "both"
  6     | Req3: "The"    ← NEW  | Req2: "transport"
  7     | Req3: "capital"       | Req2: "layer"
  8     | Req3: "is"            | Req2: "protocols"
  9     | Req3: "Paris"         | Req2: "."
 10     | Req3: <EOS> ✓ DONE    | Req2: "TCP"
 11     | Req4: "..." ← NEW     | Req2: "provides"
 ...    |                       | ...
```

This is **continuous batching**: instead of waiting for the entire batch to complete, we continuously swap finished requests out and new requests in. Request 1's response returns immediately at turn 5, and we maximize GPU utilization by keeping all slots busy.

The catch: when we swap Request 3 in, we need to **prefill** it first (build its KV cache), then align its cache with the existing sequences before continuing decode.

---

# The Continuous Batching Algorithm

Here's the core generation loop from my [PyTorch implementation](https://github.com/hawkeoni/continuous_batching_pytorch):

```python
def _run_generation_loop(self, texts, batch, results, pbar):
    next_text_idx = self.config.batch_size

    while self._should_continue_generation(batch, next_text_idx, len(texts)):
        # Add waiting texts to batch if threshold met
        if self._should_prefill(batch):
            self._prefill_waiting_texts(batch)

        # Generate one token for all active samples
        self._generate_one_step(batch)

        # Check for completed samples and swap in new ones
        finished_text_ids, finished_texts = self._collect_finished_samples(batch)

        if finished_texts:
            self._save_results(results, finished_text_ids, finished_texts)
            next_text_idx = self._add_new_waiting_texts(
                batch, texts, next_text_idx, len(finished_texts)
            )
```

Notice that `batch_size` is a configuration parameter limiting simultaneous sequences. In production frameworks like vLLM or TensorRT-LLM, the limit isn't a fixed sample count but rather a **cumulative token budget** (total tokens across all sequences). This allows dynamic allocation: many short sequences or fewer long ones. For simplicity, we use a fixed batch size.

Let's break down each component.

---

## Prefill Decision: When to Add New Sequences

```python
def _should_prefill(self, batch: _Batch) -> bool:
    has_capacity = len(batch.texts_decoding) < self.config.batch_size
    meets_threshold = (
        len(batch.texts_waiting) >=
        len(batch.texts_decoding) * self.config.fraction
    )
    return has_capacity and meets_threshold
```

We prefill when:
1. **Capacity**: The batch isn't full
2. **Threshold**: Waiting texts meet a fraction threshold relative to active texts

The `fraction` parameter controls batching aggressiveness. A fraction of 1.0 means "wait until we have as many waiting requests as active ones." A fraction of 0.0 prefills immediately when there's capacity.

**Note:** Production frameworks use more sophisticated rules considering memory pressure, estimated completion times, request priorities, and whether prefill would cause reallocation. Our threshold-based approach is a simplification.

---

## Prefill Stage: Building the KV Cache

```python
def _prefill_waiting_texts(self, batch: _Batch) -> None:
    # Tokenize waiting texts
    inputs = self._tokenize_waiting_texts(batch.texts_waiting)

    # Run prefill forward pass
    prefill_outputs = self.model(**inputs, use_cache=True)

    # Initialize or expand the batch
    if self._is_first_prefill(batch):
        self._initialize_batch_from_prefill(batch, prefill_outputs, inputs)
    else:
        self._expand_batch_with_prefill(batch, prefill_outputs, inputs)
```

The prefill stage:
1. **Tokenizes** waiting texts into padded tensors
2. **Forward pass** with `use_cache=True` returns logits and the computed KV cache
3. **Stores** the KV cache for future decode steps

When expanding an existing batch, we face an alignment problem—existing sequences have longer KV caches than newly prefilled ones:

```
[DIAGRAM PLACEHOLDER: KV Cache Alignment During Prefill Expansion]

Existing sequences (already generating):
┌─────────────────────────────────────────────────────────┐
│ KV Cache: [prompt tokens] [generated tokens...]         │
│ Length: 50 tokens                                       │
└─────────────────────────────────────────────────────────┘

New sequences (just prefilled):
┌─────────────────────────────────────┐
│ KV Cache: [prompt tokens]           │
│ Length: 20 tokens                   │
└─────────────────────────────────────┘

After padding and concatenation:
┌─────────────────────────────────────────────────────────┐
│ Existing: [prompt tokens      ] [generated tokens...]   │
│ New:      [padding: 0 0 0 ... ] [prompt tokens     ]    │
│           ↑ 30 zeros padded    ↑ attention mask = 0     │
└─────────────────────────────────────────────────────────┘
```

The code handles this with explicit padding:

```python
def _expand_kv_cache(self, batch, prefill_outputs):
    existing_seqlen = batch.past_key_values.layers[0].keys.size(2)
    new_seqlen = prefill_outputs.past_key_values.layers[0].keys.size(2)
    padding_seqlen = existing_seqlen - new_seqlen

    # Create zero padding and concatenate
    padding_template = torch.zeros(...)
    for layer_idx in range(self.model.config.num_hidden_layers):
        # Keys: [padding | new_keys] then concat with existing
        padded_keys = torch.cat((padding_template, new_layer_cache.keys), dim=2)
        layer_cache.keys = torch.cat((layer_cache.keys, padded_keys), dim=0)
```

---

## Generate One Step: The Decode Phase

```python
def _generate_one_step(self, batch: _Batch) -> None:
    step_outputs = self.model(
        input_ids=batch.input_ids,           # [batch_size, 1] - just the last token
        attention_mask=batch.attention_mask, # [batch_size, seq_len] - full history
        position_ids=batch.position_ids,     # [batch_size, 1] - current position
        past_key_values=batch.past_key_values,
        use_cache=True,
    )

    # Get next tokens from logits
    batch.input_ids = step_outputs.logits[:, 0].argmax(dim=1, keepdim=True)

    # Extend attention mask for next step
    batch.attention_mask = torch.cat(
        (batch.attention_mask, torch.ones_like(batch.attention_mask[:, 0:1])),
        dim=1
    )

    batch.generated_tokens_counter += 1
```

Each decode step processes all active sequences in parallel. We only pass the **last generated token** as input, while the KV cache contains the full history.

```
[DIAGRAM PLACEHOLDER: KV Cache Growth During Decode Steps]

Step 0 (after prefill):
Seq 1: [████████████████████] len=20
Seq 2: [████████████████████] len=20
Seq 3: [████████████████████] len=20

Step 5:
Seq 1: [████████████████████▓▓▓▓▓] len=25
Seq 2: [████████████████████▓▓▓▓▓] len=25
Seq 3: [████████████████████▓▓▓▓▓] len=25

Step 10:
Seq 1: [████████████████████▓▓▓▓▓▓▓▓▓▓] len=30
Seq 2: [████████████████████▓▓▓▓▓▓▓▓▓▓] len=30
Seq 3: [████████████████████▓▓▓▓▓▓▓▓▓▓] len=30

█ = prefilled tokens    ▓ = generated tokens
```

---

## Collecting Finished Samples: Stopping Criteria and Cache Surgery

```python
def _find_finished_indices(self, batch: _Batch) -> List[int]:
    is_eos = batch.input_ids == self.tokenizer.eos_token_id
    is_max_length = (
        batch.generated_tokens_counter.unsqueeze(1) >=
        self.config.max_new_tokens
    )

    finished_mask = (is_eos | is_max_length).view(-1).long()
    finished_indices = finished_mask.nonzero().view(-1).tolist()
    return finished_indices
```

A sequence finishes when:
1. **EOS token**: The model generated end-of-sequence
2. **Max length**: Reached `max_new_tokens` limit

When sequences finish, we surgically remove them from all batch tensors:

```python
def _remove_samples_from_batch(self, batch, keep_indices):
    batch.input_ids = batch.input_ids.index_select(0, keep_indices)
    batch.position_ids = batch.position_ids.index_select(0, keep_indices)
    batch.attention_mask = batch.attention_mask.index_select(0, keep_indices)
    batch.generated_tokens_counter = batch.generated_tokens_counter.index_select(0, keep_indices)

    # Remove from KV cache - this is expensive!
    for layer_idx in range(self.model.config.num_hidden_layers):
        layer_cache = batch.past_key_values.layers[layer_idx]
        layer_cache.keys = layer_cache.keys.index_select(0, keep_indices)
        layer_cache.values = layer_cache.values.index_select(0, keep_indices)
```

**This is expensive.** The `index_select` operation allocates new memory and copies all data for remaining sequences. For a 32-layer model, that's 64 tensor copies per removal. Production frameworks solve this with:

- **PagedAttention** (vLLM): Treats KV cache as virtual memory pages, allowing efficient "freeing"
- **Pre-allocated pools**: Reserve max memory upfront, manage slots with indices
- **In-place compaction**: Move data within the same buffer

Our naive implementation accepts the performance hit for simplicity.

**Connection to prefill:** When sequences finish, `len(batch.texts_decoding)` decreases, creating capacity for `_should_prefill` to trigger. Finished sequences create "slots" for new ones—this is the heart of continuous batching.

---

# Benchmark Results

Testing with **Qwen3-8B** on 100 samples:

| Metric | Synchronous | Continuous | Improvement |
|--------|-------------|------------|-------------|
| Total Runtime | 107.8s | 64.6s | **40% faster** |
| Generation Speed | 28.9 tok/s | 49.1 tok/s | **70% faster** |
| Per-sample Latency | 3.19s | 1.94s | **39% lower** |
| Correctness | - | 99% match | ✓ |

---

# Conclusion

Continuous batching is a fundamental technique for efficient LLM serving. By dynamically managing the batch as sequences complete, we achieve significantly better GPU utilization and lower latency.

The key ideas:
1. **KV-cache** avoids redundant computation by caching Key and Value projections
2. **Prefill** builds the initial cache for new sequences; **decode** extends it one token at a time
3. **Continuous batching** swaps finished sequences out and new ones in, keeping GPU slots busy

The full implementation is available at [github.com/hawkeoni/continuous_batching_pytorch](https://github.com/hawkeoni/continuous_batching_pytorch).

---

# Advanced Topics

## Chunked Prefill

Long prompts create a problem: prefilling an 8000-token sequence monopolizes the GPU while decode requests wait. **Chunked prefill** splits the prefill into smaller pieces that can be interleaved with decode steps.

For an 8000-token prompt with 1000-token chunks:

| Iteration | Action | KV Cache State |
|-----------|--------|----------------|
| 1 | Prefill tokens 0–999 | Cache positions 0–999 |
| 2 | Prefill tokens 1000–1999 | Cache positions 0–1999 |
| 3 | Prefill tokens 2000–2999 | Cache positions 0–2999 |
| ... | ... | ... |
| 8 | Prefill tokens 7000–7999 | Cache complete (0–7999) |
| 9 | Begin decode phase | Generate first token |

The key insight: each chunk computes new KV entries but attends to **all** previous KV entries from the cache.

**Why this matters for continuous batching:** Between prefill iterations, we can batch decode steps from other requests:

```
Iteration 3:
- Prefill chunk (tokens 2000–2999) for Request A
- Decode step (1 token) for Request B
- Decode step (1 token) for Request C
All in one forward pass!
```

Further reading:
- [Sarathi paper (arxiv)](https://arxiv.org/pdf/2308.16369)
- [HuggingFace blog on prefill/decode concurrency](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests)

---

## Paged Attention

Paged attention is a memory management technique for KV caches, introduced in the [vLLM paper](https://arxiv.org/pdf/2309.06180).

**The problem:** Standard implementations pre-allocate contiguous memory for each sequence's KV cache based on maximum possible length. If max length is 8k but a sequence only uses 500 tokens, you've wasted memory for 7500 tokens. This limits batch sizes.

**The solution:** Borrow from OS virtual memory—divide KV cache into fixed-size **pages** (e.g., 16 or 32 tokens each). Each sequence gets a **block table** mapping logical positions to physical memory blocks.

```
Sequence A needs 50 tokens:
  Logical block 0 → Physical block 7
  Logical block 1 → Physical block 3
  Logical block 2 → Physical block 12

Blocks don't need to be contiguous in GPU memory.
```

**Benefits:**
- **Near-zero waste**: Only allocate what you use
- **No fragmentation**: Uniform block sizes, any free block works
- **Higher batch sizes**: Memory savings fit more sequences
- **Easy sharing**: Beam search/parallel sampling can share blocks for common prefixes

**Tradeoff:** The attention kernel is slightly slower due to indirection and non-contiguous memory access. But the memory efficiency gains—allowing 3x more sequences in a batch—lead to higher overall throughput.

The vLLM paper showed 2-4x throughput improvements over HuggingFace's naive implementation, almost entirely from fitting more sequences in each batch.

---

*Questions or feedback? Open an issue on the [GitHub repo](https://github.com/hawkeoni/continuous_batching_pytorch) or [contact me directly](/about/).*
