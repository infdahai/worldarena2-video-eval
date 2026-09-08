# Wan Action Adapter Smoke Design

## Objective

Add the frozen eight-channel action raster to the official Wan2.2 TI2V-5B
transformer without forking or editing the upstream Wan repository. Prove shape
alignment, zero-init identity, timestep gating, gradient flow, and bounded
parameter count on a tiny backbone before downloading model weights.

## Upstream Contract

The pinned official Wan2.2 source is commit
`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`. TI2V-5B uses:

- 30 transformer blocks;
- hidden dimension 3072;
- patch size `(1, 2, 2)` after VAE latents;
- VAE stride `(4, 16, 16)` from RGB video.

For an 81-frame `480x832` video, the expected transformer grid is
`21x15x26`, or 8,190 tokens.

## Adapter

`ActionRasterEncoder` consumes `(B, 8, T, H, W)` and uses:

1. a `(5,16,16)` convolution with stride `(4,16,16)` and temporal padding 2;
2. normalization and SiLU;
3. a `(1,2,2)` convolution with stride `(1,2,2)`.

The result is flattened to `(B, L, adapter_dim)`. `adapter_dim` defaults to
256. Four independent zero-initialized linear projections map it to the Wan
hidden dimension at injection points `0, 8, 16, 24`.

A small MLP maps normalized diffusion timestep to four sigmoid gates. Each
injection is:

```text
x = x + gate_i(t) * zero_projection_i(action_tokens)
```

The zero projections guarantee exact output identity at initialization even
though gates are nonzero.

## Integration

`ActionConditionedWan` wraps an unmodified backbone. For one forward call it:

1. builds and pads action residuals to `seq_len`;
2. registers temporary forward-pre-hooks on blocks 0, 8, 16, and 24;
3. calls the original Wan forward with unchanged arguments;
4. removes all hooks in a `finally` block.

The smoke wrapper is single-process and does not claim sequence-parallel or
`torch.compile` compatibility. Those integrations follow only after real 5B
memory measurements.

## Gates

Before any Wan weights are downloaded:

- tiny encoder output shape matches the hand-derived token grid;
- invalid channel count or non-divisible spatial size raises `ValueError`;
- wrapped and unwrapped backbone outputs are bit-identical at zero init;
- making one projection nonzero changes the output;
- gradients reach the action raster, encoder, gate, and projection;
- trainable adapter parameter count is reported.

The official 34.2GB repository is not downloaded as a whole for this smoke.
When the gates pass, the next step may download only the exact transformer and
VAE files needed for a 5B forward-memory probe, after the disk guard approves a
two-times temporary-file budget.

