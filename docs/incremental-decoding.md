# Incremental decoding and KV storage

The main octree already caches layer keys and values. Its reference implementation
concatenates each new chunk to the complete prefix, copying the prefix repeatedly.
The optional `inference_config.cache_mode: buffer` uses geometric capacity growth
and writes new keys/values into unused positions. It exposes only the occupied
prefix to attention. When molecules finish, both occupied views and their backing
storage retain exactly the same surviving rows. Conditional and unconditional CFG
branches keep separate caches. The default remains `concat` during validation.

The inference buffer runs only without autograd. Model weights, dtype, positional
encoding, masks and attention computation are unchanged. Capacity growth avoids
per-token prefix copying; it does not remove the attention computation or the
other trajectory-building allocations. Peak memory and latency must be measured.

Tests compare full causal attention to cached chunks across growth and early row
completion. They exposed a separate compatibility bug in the SDPA wrapper: for
unequal query/key lengths, cached causal queries must align with the suffix of the
keys. A plain `is_causal=True` aligns the upper-left corner instead. Packed query
and key boundaries also need separate splits. These are now explicit. Existing
mainline generation uses chunks smaller than 128 and its manual attention branch;
the correction matters to long-chunk cache/prefill callers.

`scripts/validate_octree_cache.py` compares every layer, final hidden state, tree
and atom/XYZ logits on real held-out trajectories for chunk sizes 1, 2 and 129. It
then compares all generated tokens, coordinates and scores with CFG scales 0, 1
and 1.5, and alternates cache-mode timing order after warmup. It preserves numerical
comparison thresholds separately from exact token/coordinate equality. Shared-GPU
measurements are not dedicated-hardware throughput guarantees.

The first real-weight comparison exceeded its predeclared 2e-4 absolute logit
threshold: the maximum was 2.3651e-4 for the existing concatenation path and the
buffer path alike. The original failed run is retained. The revised diagnostic
keeps this threshold and its pass/fail field, while separately checking bitwise
buffer-versus-concat hidden states and atom/XYZ logits for each fixed chunk size.
Full-prefix numerical agreement and equivalence of two storage implementations
are different claims. A small final hidden-state difference can be amplified by
the readout; neither raw-logit nor trajectory equality is inferred from it.

## Real-weight result

On two held-out molecular trajectories, both storage modes produce bitwise equal
cached hidden states and atom/XYZ logits at chunk sizes 1, 2 and 129. For eight
generated molecules at each CFG scale 0/1/1.5, all 24 token sequences, coordinates
and sampling scores are exact matches. This validates the storage change on that
cohort; the separate full-prefix absolute threshold remains failed.

Three alternating shared-A800 timings give medians 11.8772 s for concatenation
and 11.8761 s for buffering. Peak PyTorch allocations are 639.4 and 689.5 MB,
respectively. This run shows no meaningful speed improvement and higher peak
allocation. The default remains `concat`; `buffer` is an explicit experimental
option, not a recommended acceleration. The complete measurements and the original
failed prefix comparison are archived as `experiments/kv_buffer_20261112.json`
and `experiments/prefix_absolute_threshold_failure_20261112.json`.
