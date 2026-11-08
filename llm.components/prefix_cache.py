##############################
# Prefix cache.

import hashlib
from dataclasses import dataclass


BLOCK_SIZE = 16


@dataclass
class KVBlock:
    block_id: int
    block_hash: bytes


def hash_block(
    parent_hash: bytes,
    token_ids: list[int],
    extra_keys: bytes = b"",
) -> bytes:
    h = hashlib.sha256()

    h.update(parent_hash)

    for token in token_ids:
        h.update(token.to_bytes(4, "little"))

    h.update(extra_keys)

    return h.digest()


class PrefixCache:
    def __init__(self):
        self.blocks: dict[bytes, KVBlock] = {}

    def insert(
        self,
        token_ids: list[int],
        block_ids: list[int],
    ):
        parent_hash = b""

        for i, block_id in enumerate(block_ids):
            start = i * BLOCK_SIZE
            end = start + BLOCK_SIZE

            tokens = token_ids[start:end]

            if len(tokens) < BLOCK_SIZE:
                break

            block_hash = hash_block(
                parent_hash, tokens
            )

            self.blocks[block_hash] = KVBlock(
                block_id=block_id,
                block_hash=block_hash,
            )

            parent_hash = block_hash

    def lookup(self, token_ids: list[int]):
        parent_hash = b""
        cached_blocks = []

        for start in range(
            0, len(token_ids), BLOCK_SIZE
        ):
            tokens = token_ids[
                start:start + BLOCK_SIZE
            ]

            # Only full blocks are reusable here.
            if len(tokens) < BLOCK_SIZE:
                break

            block_hash = hash_block(
                parent_hash, tokens
            )

            block = self.blocks.get(block_hash)

            if block is None:
                break

            cached_blocks.append(block)
            parent_hash = block_hash

        return cached_blocks


# Example: cache 3 blocks (48 tokens).
cache = PrefixCache()

original_prompt = list(range(48))

cache.insert(
    original_prompt,
    block_ids=[7, 21, 35],
)

# New request shares the first 32 tokens.
new_prompt = (
    list(range(32))
    + list(range(100, 132))
)

hits = cache.lookup(new_prompt)

print("Cached block IDs:", [b.block_id for b in hits])
print("Reusable tokens:", len(hits) * BLOCK_SIZE)

