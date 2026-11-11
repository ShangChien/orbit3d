"""Inference KV storage with explicit reference and amortized-buffer modes."""
import torch


def append_kv(cache, key, value):
    previous = cache.get('k')
    if cache.get('mode', 'concat') == 'concat':
        return ((torch.cat([previous, key], dim=2), torch.cat([cache['v'], value], dim=2))
                if previous is not None else (key, value))
    if cache.get('mode') != 'buffer':
        raise ValueError('KV cache mode must be concat or buffer')
    if torch.is_grad_enabled():
        raise RuntimeError('Buffered KV storage is inference-only; use torch.no_grad()')
    length = previous.shape[2] if previous is not None else 0
    needed = length + key.shape[2]
    capacity = cache['k_buffer'].shape[2] if 'k_buffer' in cache else 0
    if needed > capacity:
        capacity = max(64, 2*capacity, 1 << (needed-1).bit_length())
        shape = (*key.shape[:2], capacity, key.shape[3])
        k_buffer, v_buffer = key.new_empty(shape), value.new_empty(shape)
        if length:
            k_buffer[:, :, :length].copy_(previous)
            v_buffer[:, :, :length].copy_(cache['v'])
        cache['k_buffer'], cache['v_buffer'] = k_buffer, v_buffer
    cache['k_buffer'][:, :, length:needed].copy_(key)
    cache['v_buffer'][:, :, length:needed].copy_(value)
    return cache['k_buffer'][:, :, :needed], cache['v_buffer'][:, :, :needed]


def compact_kv(cache, keep):
    """Keep the same surviving molecule rows in views and allocated storage."""
    if 'k' not in cache:
        return
    key, value = cache['k'][keep], cache['v'][keep]
    if cache.get('mode', 'concat') == 'buffer':
        length, capacity = key.shape[2], cache['k_buffer'].shape[2]
        shape = (*key.shape[:2], capacity, key.shape[3])
        cache['k_buffer'], cache['v_buffer'] = key.new_empty(shape), value.new_empty(shape)
        cache['k_buffer'][:, :, :length].copy_(key)
        cache['v_buffer'][:, :, :length].copy_(value)
        key, value = cache['k_buffer'][:, :, :length], cache['v_buffer'][:, :, :length]
    cache['k'], cache['v'] = key, value
