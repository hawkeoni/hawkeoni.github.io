---
layout: post
title: "Continuous Batching for LLM Inference: A PyTorch Implementation"
date: 2025-01-15 12:00:00 +0000
categories: [Deep Learning, LLM]
tags: [pytorch, inference, optimization, transformers]
toc: true
comments: true
description: "A deep dive into continuous batching - the technique that powers efficient LLM inference."
---

# Introduction

One day I was reading a lecture about LLM inference frameworks and what optimizations make them fast: dynamic batching, efficient memory management (memory reuse), efficient kernels/fused operations, various model parallellisms (tensor parallel/pipeline parallel inference), quantization, speculative decoding, **KV-cache** and **continous batching**. After I described continous batching one of the students asked if there was a simple implementation. I knew that digging in the source code of such frameworks as [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang) would be too hard, so I tried looking for some open source implementations. This was before [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), [mini-SGLang](https://github.com/sgl-project/mini-sglang) or support of [contionous batching in transformers](https://huggingface.co/docs/transformers/main/continuous_batching) so I've searched the internet and found a link to a [reference implementation in pytorch](https://inspiringlab.com.np/implementing-continuous-batching-from-scratch-with-pytorch/) which looked __good enough__. I however quickly found that this was not the case: the code did not work. In fact it did not constitute a program: there were no imports, the code referenced classes and functions that were never described anywhere and not matter how you permuted the provided code snippets you could never compose anything that would launch. I guess it could be considered almost a pseudo-code implementation, but this was not something that I was looking for, so I took it upon myself to write a simple implementation in pytorch.

TODO: remake this as CONTENTS so it is a nice table

In this blogpost I will describe the following:
* A small intro to KV cache in transformers and why it works
* Drawbacks of naive inference algorithms and how continous batching solves them
* Walkthrough of naive continous batching in pytorch
* Analysis of the algorithm and comparison to [native transformers continous batching](https://huggingface.co/docs/transformers/main/continuous_batching)
* Additional reading materials

# KV-Cache and token generation and prefill and decode stages
One of the optimizations that is **an absolute must** in any inference of a transformer-decoder (which is the most common LLM architecture as of 2025) is KV-cache. To understand the following material you need to have a great understanding of basic transformer architecture and I recommend the original [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html), because I believe it still holds pretty well, but you may also go for something more modern such as [An even more annotated Transformer](https://pi-tau.github.io/posts/transformer/) or any other explaination that you like.

I'd also recommend you give a read to official [huggingface post about continous batching](https://huggingface.co/blog/continuous_batching) because it describes the process of autoregressive token generation in great detail, but I'll give the gist of it.


TODO: Add diagram of causal attention and KV cache

First of all: **because of the causal mask attention is the only transformer layer where tokens interact with each other, that means that this is the only layer which works on the whole sequence.** All other layers such as FFN (MLP), positional embeddings (absolute or RoPE), layernorms and final linear layer **work on each token independently and do not require the whole sequence**. Only the famous attention layer 
$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ requires full matrices Q, K, V of shape \[sequence_length, hidden_dim\] - all the other layers can correctly work on each token vector of size \[hidden_dim\].

Second: only last token is used for prediction of the next token and because of the causal mask future tokens do not change representation of previous tokens. That means that we can cache outputs of previous tokens before calculating attention.

Here we will have a walkhtrough of a small language model that generates tokens autoregressively with and without KV cache and discuss speedup because of KV-cache. You can read whole in one place code [here](https://gist.github.com/hawkeoni/2920d1a2f59840eb673455b40137c73c)

First of all let's define a simplified transformer language model which does not have layernorms, residual connections or FFN layers - as stated above those layers work on tokens independently so this would not change the result and for simplicity we take them out.

```python
class SimpleCausalAttentionLLM(nn.Module):
    """
    A minimal single-layer causal attention model for educational purposes.
    This model implements:
    1. Token embedding
    2. Single-head self-attention with causal masking
    3. Output projection to vocabulary
    Note: Real LLMs have multiple layers, multi-head attention, layer norms,
    feed-forward networks, and positional encodings. This is simplified to
    focus on the KV-cache mechanism: those layers can be easily inserted into the model
    because the operate on tokens independently and do not require the full sequence of tokens.
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


Let's define a simple forward that does the following: embeds tokens from input_ids, calculates causal attention and predicts the next token. 

```python

def forward(self, input_ids: torch.Tensor) -> CausalAttentionOutput:
        """
        Standard forward pass - computes attention over the full sequence.
        Used for initial "prefill" phase or when not using KV-cache.
        """
        # Step 1: Embed tokens
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        hidden_states = self.embedding(input_ids)

        # Step 2: Compute Query, Key, Value projections
        # Each: [batch_size, seq_len, d_model]
        queries = self.W_Q(hidden_states)
        keys = self.W_K(hidden_states)
        values = self.W_V(hidden_states)

        # Step 3: Compute attention scores
        # Q @ K^T -> [batch_size, seq_len, seq_len]
        # Each position attends to all other positions
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))

        # Step 4: Apply causal mask (lower triangular)
        # This prevents tokens from attending to future tokens
        # Essential for autoregressive generation
        causal_mask = torch.tril(torch.ones_like(attention_scores))
        attention_scores = attention_scores.masked_fill(
            causal_mask == 0,
            float('-inf')
        )

        # Step 5: Softmax to get attention weights
        # [batch_size, seq_len, seq_len] - each row sums to 1
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Step 6: Apply attention weights to values
        # [batch_size, seq_len, seq_len] @ [batch_size, seq_len, d_model]
        # -> [batch_size, seq_len, d_model]
        context = torch.matmul(attention_weights, values)

        # Step 7: Project to vocabulary
        logits = self.output_projection(context)

        return CausalAttentionOutput(
            logits=logits,
            k_cache=keys,
            v_cache=values,
            attn_weights=attention_weights
        )
```

As you can see full attention is calculated here and if we wanted to generate new tokens our generation function would look like this:


```python
def generate_without_cache(
    model: SimpleCausalAttentionLLM,
    input_ids: torch.Tensor,
    num_new_tokens: int
) -> torch.Tensor:
    """
    Generate tokens WITHOUT KV-cache (naive approach).
    For each new token, we reprocess the ENTIRE sequence from scratch.
    
    Complexity analysis:
    - Step 1: process 1 token  → O(1)
    - Step 2: process 2 tokens → O(4)
    - Step N: process N tokens → O(N^2)
    - Total: O(1^2 + 2^2 + ... + N^2) = O(N^3)
    
    This cubic complexity makes generation slow for long sequences.
    """
    current_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(num_new_tokens):
            # Recompute attention over the FULL sequence every time
            outputs = model(current_ids)

            # Get prediction for the last position
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Append new token to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)

    return current_ids
```

The problem here is that:
* On each generation step the complexity of full attention calculation is quadratic, so for N steps the generation algorithm is actually cubic!
* We constantly recalculate states for each token, however only the last token state is required to generate the next token

Let's see how we can solve it with KV-cache - the trick is simple - we only calculate attention of last token by all previous tokens and previous tokens do not get recalculated because they are independent of future tokens because of causal mask

```python
def forward_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        past_output: CausalAttentionOutput
    ) -> CausalAttentionOutput:
        """
        Optimized forward pass using KV-cache.
        Only processes the NEW token, reusing cached K and V from previous tokens.
        Instead of recomputing K and V for all previous tokens, we:
        1. Compute Q, K, V only for the new token
        2. Concatenate new K, V with cached K, V
        3. Compute attention using full K, V but only new Q
        """
        # Embed only the new token
        # [batch_size, 1] -> [batch_size, 1, d_model]
        hidden_states = self.embedding(input_ids)

        # Compute Q, K, V for the new token only
        # Each: [batch_size, 1, d_model]
        new_query = self.W_Q(hidden_states)
        new_key = self.W_K(hidden_states)
        new_value = self.W_V(hidden_states)

        # Retrieve cached K and V from previous tokens
        # [batch_size, prev_seq_len, d_model]
        cached_keys = past_output.k_cache
        cached_values = past_output.v_cache

        # Extend cache with new K, V
        # [batch_size, prev_seq_len + 1, d_model]
        updated_keys = torch.cat([cached_keys, new_key], dim=1)
        updated_values = torch.cat([cached_values, new_value], dim=1)

        # Compute attention: new token attends to ALL tokens (including itself)
        # [batch_size, 1, d_model] @ [batch_size, d_model, seq_len+1]
        # -> [batch_size, 1, seq_len+1]
        attention_scores = torch.matmul(
            new_query,
            updated_keys.transpose(-2, -1)
        )

        # No causal mask needed here! The new token is the last position,
        # so it can attend to all previous tokens (and itself)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention to get context for the new token
        # [batch_size, 1, seq_len+1] @ [batch_size, seq_len+1, d_model]
        # -> [batch_size, 1, d_model]
        context = torch.matmul(attention_weights, updated_values)

        # Project to vocabulary
        logits = self.output_projection(context)

        # Update the full attention weight matrix for visualization
        # (This is optional and just for educational purposes)
        batch_size = hidden_states.size(0)
        prev_seq_len = cached_keys.size(1)

        # Previous attention weights: [batch, prev_seq_len, prev_seq_len]
        past_attn_weights = past_output.attn_weights

        # Add zero column: old tokens don't attend to new token (causal)
        zeros_column = torch.zeros(batch_size, prev_seq_len, 1)
        past_attn_weights = torch.cat([past_attn_weights, zeros_column], dim=2)

        # Add new token's attention row
        full_attn_weights = torch.cat([past_attn_weights, attention_weights], dim=1)

        # Note on memory efficiency: These concatenations (torch.cat) are costly for
        # large models and long sequences because they allocate new memory and copy
        # all existing data. Production implementations pre-allocate fixed-size buffers
        # and write to specific indices instead. We use concatenation here for clarity.

        return CausalAttentionOutput(
            logits=logits,
            k_cache=updated_keys,
            v_cache=updated_values,
            attn_weights=full_attn_weights
        )

```

Then generation would look like this
```python
def generate_without_cache(
    model: SimpleCausalAttentionLLM,
    input_ids: torch.Tensor,
    num_new_tokens: int
) -> torch.Tensor:
    """
    Generate tokens WITHOUT KV-cache (naive approach).
    For each new token, we reprocess the ENTIRE sequence from scratch.
    
    Complexity analysis:
    - Step 1: process 1 token  → O(1)
    - Step 2: process 2 tokens → O(4)
    - Step N: process N tokens → O(N^2)
    - Total: O(1^2 + 2^2 + ... + N^2) = O(N^3)
    
    This cubic complexity makes generation slow for long sequences.
    """
    current_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(num_new_tokens):
            # Recompute attention over the FULL sequence every time
            outputs = model(current_ids)

            # Get prediction for the last position
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Append new token to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)

    return current_ids
```

As you can see the complexity is square, because each step is linear.
So to summarize the ideas behind KV cache are:
* To use old computed kv caches, because a) all layers except for attention work on tokens independently and do not change when new tokens are added b) attention also does not change the state of previous tokens because the do not attend on future tokens
* To calculate only last token attention to previous tokens making each step not squared but linear


# Pitfalls of naive generation and Continous Batching algorithm
Now that we understand how KV-cache allows us to circumvent cubic generation time let's take a look at another problem associated with  transformer inference: generating on sequences of various lengths.
All modern frameworks batch user requests together - serving them one by one would underutilize the GPU and work slowly, so it makes sense to batch generation together. The problem here is that all sequence require answers of different lengths. For example let's say you've bathced 2 requests from different user, one simple yes or no question and the other question requires more tokens. For example:

Request 1: "Is Python dynamically typed? Answer yes or no."
Request 2: "Explain the difference between TCP and UDP protocols."

Here's how generation proceeds with naive batching after the prefill stage:


Turn    | Request 1 (short)     | Request 2 (long)
--------|-----------------------|------------------------
  1     | "Yes"                 | "TCP"
  2     | ","                   | "and"
  3     | "Python"              | "UDP"
  4     | "is"                  | "are"
  5     | "\<EOS>"              | "both"
  6     | \<padding>            | "transport"
  7     | \<padding>            | "layer"
  8     | \<padding>            | "protocols"
  9     | \<padding>            | "."
 10     | \<padding>            | "TCP"
 11     | \<padding>            | "provides"
 ...    | \<padding>            | ...
 30     | \<padding>            | "\<EOS>"

The problem is clear: Request 1 finishes at turn 5, but we can't return its response to the user until Request 2 completes at turn 30. Meanwhile, the GPU slot for Request 1 sits idle, wasting compute on generating tokens that the user will not see because generation was terminated with \<EOS>. What we'd like to do is switch request 1 for request 3 after it finishes so we'd never lose compute. 

For example if Request 3 is "What is the capital of Paris" the generations may look something like this:

Turn    | Slot A                | Slot B
--------|-----------------------|------------------------
  1     | Req1: "Yes"           | Req2: "TCP"
  2     | Req1: ","             | Req2: "and"
  3     | Req1: "Python"        | Req2: "UDP"
  4     | Req1: "is"            | Req2: "are"
  5     | Req1: "<EOS>" ✓ DONE  | Req2: "both"
  6     | Req3: "The"    ← NEW  | Req2: "transport"
  7     | Req3: "capital"       | Req2: "layer"
  8     | Req3: "is"            | Req2: "protocols"
  9     | Req3: "Paris"         | Req2: "."
 10     | Req3: "<EOS>" ✓ DONE  | Req2: "TCP"
 11     | Req4: "..." ← NEW     | Req2: "provides"
 ...    |                       | ...

This is the core idea behind continuous batching: instead of waiting for the entire batch to complete, we continuously swap finished requests out and new requests in, maximizing GPU utilization and minimizing user latency. Request 1's response is returned immediately at turn 5, rather than waiting 25 more turns for Request 2 to finish and request 3 to start.

The problem here is that generation consists of 2 stages: prefill and decode and before we start generating Request 3 we need to actually prefill it and swap KV cache of Req1 to KV cache of Req3!

TODO: write somewhere better that prefilling is basically KV-cache generation. In the next section we'll explore the naive continous batching algorithm in pytorch, how it manages memory and interleaves prefill stages with decode steps.

TODO: Add diagram showing prefill vs generation phases


When serving Large Language Models (LLMs) in production, efficient GPU utilization is critical. Traditional batch processing has a fundamental flaw: all sequences in a batch must wait for the longest one to finish. Enter **continuous batching** - the technique that enables systems like vLLM and HuggingFace TGI to achieve remarkable throughput improvements.

In this post, I'll walk through my [PyTorch implementation of continuous batching](https://github.com/hawkeoni/continuous_batching_pytorch) and explain the core algorithm that achieves **~40% faster inference** compared to synchronous batching.


## Core Algorithm

Here's the high-level algorithm:

```python
def continuous_batch(texts: List[str]) -> List[str]:
    results = [None] * len(texts)
    batch = initialize_batch(texts[:batch_size])
    next_idx = batch_size

    while has_work_remaining(batch, next_idx, texts):
        # Check if we should prefill waiting samples
        if should_prefill(batch):
            prefill_waiting_samples(batch)

        # Generate one token for all active samples
        generate_one_step(batch)

        # Check for completed samples
        finished = collect_finished_samples(batch)

        if finished:
            save_results(results, finished)
            add_new_samples_to_waiting(batch, texts, next_idx)

    return results
```

### The Prefill Decision

A critical question: **when should we prefill waiting samples?**

Too aggressive (prefill immediately) → frequent context switches, overhead
Too conservative (wait for batch to empty) → approaches synchronous batching

My implementation uses a `fraction` parameter:

```python
def should_prefill(batch) -> bool:
    return len(waiting_samples) >= len(generating_samples) * fraction
```

With `fraction=0.5` and 10 generating samples, we prefill when 5+ samples are waiting.
This balances out the long sequences.


## Key Implementation Challenges

### 1. KV Cache Management

The KV cache stores attention keys and values from previous tokens. When mixing sequences of different lengths, we must carefully align the cache:

```
Existing sequence: [====tokens====|generated|]
New sequence:      [pad|=tokens=|pad|generated|]
                   ↑ Padding to align with existing cache length
```

<!-- TODO: Add code snippet showing cache expansion -->

### 2. Attention Mask Handling

When sequences have different lengths, we need proper masking:

```python
# Convert attention mask to position IDs
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)
```

This ensures each token attends only to valid previous tokens, not padding.

### 3. Batch State Tracking

We track multiple pieces of state per sequence:

```python
@dataclass
class Batch:
    text_ids: List[int]           # Original indices for result ordering
    input_ids: torch.Tensor       # Current token being decoded
    attention_mask: torch.Tensor  # Valid token positions
    past_key_values: DynamicCache # KV cache per layer
    generated_tokens: List[List[int]]  # Output tokens per sequence
```

<!-- TODO: Expand on the _Batch dataclass implementation -->

## Benchmark Results

Testing with **Qwen3-8B** on 100 samples:

| Metric | Synchronous | Continuous | Improvement |
|--------|-------------|------------|-------------|
| Total Runtime | 107.8s | 64.6s | **40% faster** |
| Generation Speed | 28.9 tok/s | 49.1 tok/s | **70% faster** |
| Per-sample Latency | 3.19s | 1.94s | **39% lower** |
| Correctness | - | 99% match | - |


## Conclusion

Continuous batching is a fundamental technique for efficient LLM serving. By dynamically managing the batch as sequences complete, we achieve significantly better GPU utilization.

The full implementation is available at [github.com/hawkeoni/continuous_batching_pytorch](https://github.com/hawkeoni/continuous_batching_pytorch).

## Advanced topics
Chunked prefill is a technique where prefill stage is split into stpes.
For example we have a sequence of length 8000 and we want to prefill it in chunks of 1000 tokens, so the prefill would go as:


Iteration 1:

Take tokens 0–999 (first 1k chunk)
Run forward pass, compute attention over these 1k tokens
Store KV cache for positions 0–999
No token generated yet (still prefilling)
Iteration 2:

Take tokens 1000–1999
Run forward pass, attention can now attend to positions 0–1999 (using cached KV for 0–999, computing new for 1000–1999)
Append KV cache for positions 1000–1999
Still no token generated
Iteration 3:

Tokens 2000–2999
Attention over 0–2999
KV cache grows
...continues...

Iteration 8:

Tokens 7000–7999 (final chunk)
Attention over full 0–7999
KV cache now complete for entire prompt
Iteration 9:

Now decode phase begins
Generate first output token
Append its KV to cache
The key insight:

Each chunk only computes new KV entries, but attends to all previous KV entries from the cache. So chunk 5 computes KV for tokens 4000–4999 but attends to 0–4999.

Why this matters for continuous batching:

Between iterations 1–8, if another request's decode step is ready, you can batch them together. So iteration 3 might look like:

Prefill chunk (tokens 2000–2999) for request A
Decode step (1 token) for request B
Decode step (1 token) for request C
All in one forward pass, keeping everyone moving.


You can read more about chunked prefill here https://arxiv.org/pdf/2308.16369 and here https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests



Another trick is Paged attention it is a memory management technique for KV caches, introduced in the [vLLM paper](https://arxiv.org/pdf/2309.06180).

The problem it solves:

In standard attention, you pre-allocate a contiguous block of memory for each sequence's KV cache based on the maximum possible length. If your max length is 8k but a sequence only uses 500 tokens, you've wasted memory for 7500 tokens. And you can't reclaim it mid-inference.

This leads to massive memory fragmentation and waste, limiting how many sequences you can batch together.

How paged attention works:

Borrow the idea from OS virtual memory—instead of contiguous allocation, divide KV cache into fixed-size blocks (pages), like 16 or 32 tokens each.

Each sequence gets a block table that maps logical positions to physical memory blocks. Blocks are allocated on demand as the sequence grows.

Example:

Sequence A needs 50 tokens of KV cache:

Allocate block 0 → physical block 7
Allocate block 1 → physical block 3
Allocate block 2 → physical block 12
The blocks don't need to be contiguous in GPU memory. The block table tracks where everything lives.

Benefits:

Near-zero waste: Only allocate what you actually use
No fragmentation: Blocks are uniform size, any free block works
Higher batch sizes: Memory savings let you fit more concurrent sequences
Easy memory sharing: For beam search or parallel sampling, multiple sequences can point to the same physical blocks for shared prefixes
Tradeoff:

Slightly more complex attention kernel—instead of one contiguous read, you gather from scattered blocks. But the memory efficiency gains far outweigh this.

It's one of the core innovations that made vLLM so much more efficient than naive serving implementations.

so this is just for memory efficiency without speedup? the speed is probably worse because of the lack of contiguous blocks of memory?

Exactly right. Paged attention by itself is a memory optimization, not a speed optimization. The attention kernel is actually slightly slower due to the indirection and non-contiguous memory access.

But—and this is the key—the memory savings indirectly lead to higher throughput.

If you can fit 3x more sequences in a batch because you're not wasting memory, your GPU utilization goes up and you process more tokens per second overall, even if each individual attention operation is marginally slower.


The vLLM paper showed something like 2-4x throughput improvements over HuggingFace's naive implementation, almost entirely from being able to batch more sequences thanks to memory efficiency.

