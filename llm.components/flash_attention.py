import math
import torch


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_q: int = 64,
    block_k: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """
    Educational FlashAttention-style forward pass.

    Args:
        q: [batch, heads, sequence_length, head_dim]
        k: [batch, heads, sequence_length, head_dim]
        v: [batch, heads, sequence_length, head_dim]
        block_q: number of query rows per tile
        block_k: number of key/value rows per tile
        causal: whether to apply causal attention

    Returns:
        output: same shape as q
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape [B, H, N, D]")

    batch, heads, query_length, head_dim = q.shape
    batch_k, heads_k, key_length, head_dim_k = k.shape

    if batch != batch_k or heads != heads_k:
        raise ValueError("q and k must have matching batch/head dimensions")

    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")

    if head_dim != head_dim_k:
        raise ValueError("q and k must have the same head dimension")

    if causal and query_length != key_length:
        raise ValueError(
            "This simple causal implementation assumes self-attention "
            "with query_length == key_length"
        )

    scale = 1.0 / math.sqrt(head_dim)
    output_blocks = []

    # Process one Q tile at a time.
    for q_start in range(0, query_length, block_q):
        q_end = min(q_start + block_q, query_length)

        # [B, H, Bq, D]
        q_tile = q[:, :, q_start:q_end, :].float()
        q_tile_length = q_end - q_start

        # Running maximum for every query row.
        # Shape: [B, H, Bq, 1]
        running_max = torch.full(
            (batch, heads, q_tile_length, 1),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )

        # Running softmax denominator.
        running_sum = torch.zeros(
            (batch, heads, q_tile_length, 1),
            device=q.device,
            dtype=torch.float32,
        )

        # Running unnormalized output numerator.
        output_accumulator = torch.zeros(
            (batch, heads, q_tile_length, head_dim),
            device=q.device,
            dtype=torch.float32,
        )

        # For causal attention, query q can only attend to keys <= q.
        # No key after q_end - 1 can be used by this entire Q tile.
        if causal:
            maximum_key_position = q_end
        else:
            maximum_key_position = key_length

        for k_start in range(0, maximum_key_position, block_k):
            k_end = min(k_start + block_k, key_length)

            # [B, H, Bk, D]
            k_tile = k[:, :, k_start:k_end, :].float()
            v_tile = v[:, :, k_start:k_end, :].float()

            # [B, H, Bq, Bk]
            scores = torch.matmul(
                q_tile,
                k_tile.transpose(-2, -1),
            )
            scores *= scale

            if causal:
                query_positions = torch.arange(
                    q_start,
                    q_end,
                    device=q.device,
                )

                key_positions = torch.arange(
                    k_start,
                    k_end,
                    device=q.device,
                )

                # [Bq, Bk]
                allowed = (
                    key_positions.unsqueeze(0)
                    <= query_positions.unsqueeze(1)
                )

                scores = scores.masked_fill(
                    ~allowed[None, None, :, :],
                    -float("inf"),
                )

            # Maximum score from the current K tile.
            # [B, H, Bq, 1]
            tile_max = scores.max(
                dim=-1,
                keepdim=True,
            ).values

            # New maximum over all K tiles seen so far.
            new_max = torch.maximum(
                running_max,
                tile_max,
            )

            # Correct the previous accumulator because it was
            # normalized relative to running_max.
            old_scale = torch.exp(
                running_max - new_max
            )

            # Exponentials for the current tile, normalized relative
            # to the new running maximum.
            probabilities = torch.exp(
                scores - new_max
            )

            # Update the softmax denominator.
            new_sum = (
                old_scale * running_sum
                + probabilities.sum(dim=-1, keepdim=True)
            )

            # Update the unnormalized output numerator.
            output_accumulator = (
                old_scale * output_accumulator
                + torch.matmul(probabilities, v_tile)
            )

            running_max = new_max
            running_sum = new_sum

        # Complete softmax normalization.
        output_tile = output_accumulator / running_sum

        output_blocks.append(output_tile)

    output = torch.cat(output_blocks, dim=-2)

    return output.to(q.dtype)
