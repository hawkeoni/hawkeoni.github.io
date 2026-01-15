from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CausalAttentionOutput:
    """
    Output container for the attention layer.

    Attributes:
        logits: Predicted token probabilities [batch_size, seq_len, vocab_size]
        k_cache: Cached Key projections [batch_size, seq_len, d_model]
        v_cache: Cached Value projections [batch_size, seq_len, d_model]
        attn_weights: Attention weight matrix [batch_size, seq_len, seq_len]
    """
    logits: torch.Tensor = None
    k_cache: torch.Tensor = None
    v_cache: torch.Tensor = None
    attn_weights: torch.Tensor = None


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


def generate_with_cache(
    model: SimpleCausalAttentionLLM,
    input_ids: torch.Tensor,
    num_new_tokens: int
) -> torch.Tensor:
    """
    Generate tokens WITH KV-cache (optimized approach).

    Two phases:
    1. Prefill: Process the initial prompt, building the KV cache. O(P²) for prompt length P.
    2. Decode: Generate tokens one at a time, each attending to the full cache. O(N) per token.

    Complexity analysis:
    - Prefill: O(P^2) for prompt of length P
    - Decode: O(P+1) + O(P+2) + ... + O(P+N) = O(N·P + N^2)
    - Total: O(P^2 + N·P + N^2) = O(N^2) for N >> P

    Compared to O(N^3) without cache, this is a significant improvement.
    For a 1000-token generation, that's ~1000x faster in the attention computation. 
    """
    generated_ids = input_ids.clone()

    with torch.no_grad():
        # PREFILL PHASE: Process the initial prompt
        # This builds our initial KV cache
        outputs = model(input_ids)

        # Get first generated token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # DECODE PHASE: Generate remaining tokens one at a time
        # Note: We only process the NEW token, not the full sequence!
        for _ in range(num_new_tokens - 1):
            # Only pass the latest token + cached K,V
            outputs = model.forward_with_kv_cache(next_token, outputs)

            # Get next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

    return generated_ids


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Model configuration
    BATCH_SIZE = 1
    INITIAL_SEQ_LEN = 3
    D_MODEL = 128
    VOCAB_SIZE = 7
    NUM_TOKENS_TO_GENERATE = 10

    # Create model
    model = SimpleCausalAttentionLLM(d_model=D_MODEL, vocab_size=VOCAB_SIZE)

    # Initialize weights (just for consistent demo)
    for param in model.parameters():
        nn.init.normal_(param)

    # Create random input sequence
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, INITIAL_SEQ_LEN))

    print("=" * 60)
    print("KV-Cache Demonstration")
    print("=" * 60)
    print(f"\nInitial sequence: {input_ids.tolist()}")
    print(f"Generating {NUM_TOKENS_TO_GENERATE} new tokens...\n")

    # Method 1: Without cache (naive)
    result_no_cache = generate_without_cache(
        model,
        input_ids.clone(),
        NUM_TOKENS_TO_GENERATE
    )
    print(f"Without KV-cache: {result_no_cache.tolist()}")

    # Method 2: With cache (optimized)
    result_with_cache = generate_with_cache(
        model,
        input_ids.clone(),
        NUM_TOKENS_TO_GENERATE
    )
    print(f"With KV-cache:    {result_with_cache.tolist()}")

    # Verify both methods produce identical results
    outputs_match = torch.equal(result_no_cache, result_with_cache)
    print(f"\nOutputs match: {outputs_match}")
