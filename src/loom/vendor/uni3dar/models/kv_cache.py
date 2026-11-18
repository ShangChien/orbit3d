"""Inference KV storage with explicit reference, preallocated-buffer, and gather modes."""
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
    bsz = key.shape[0]
    length = previous.shape[2] if previous is not None else 0
    needed = length + key.shape[2]
    capacity = cache['k_buffer'].shape[2] if 'k_buffer' in cache else 0
    buf_bsz = cache['k_buffer'].shape[0] if 'k_buffer' in cache else 0
    if needed > capacity or bsz > buf_bsz:
        init_cap = int(cache.get('init_capacity', 64))
        capacity = max(init_cap, 2 * capacity, 1 << (needed - 1).bit_length())
        alloc_bsz = max(bsz, buf_bsz)
        shape = (alloc_bsz, key.shape[1], capacity, key.shape[3])
        k_buffer, v_buffer = key.new_empty(shape), value.new_empty(shape)
        if length and previous is not None:
            k_buffer[:bsz, :, :length].copy_(previous)
            v_buffer[:bsz, :, :length].copy_(cache['v'])
        cache['k_buffer'], cache['v_buffer'] = k_buffer, v_buffer
    cache['k_buffer'][:bsz, :, length:needed].copy_(key)
    cache['v_buffer'][:bsz, :, length:needed].copy_(value)
    return cache['k_buffer'][:bsz, :, :needed], cache['v_buffer'][:bsz, :, :needed]


def compact_kv(cache, keep):
    """Keep the same surviving molecule rows in views and preallocated storage without reallocating."""
    if 'k' not in cache:
        return
    key, value = cache['k'][keep], cache['v'][keep]
    if cache.get('mode', 'concat') == 'buffer' and 'k_buffer' in cache:
        n_keep, length = key.shape[0], key.shape[2]
        cache['k_buffer'][:n_keep, :, :length].copy_(key)
        cache['v_buffer'][:n_keep, :, :length].copy_(value)
        key = cache['k_buffer'][:n_keep, :, :length]
        value = cache['v_buffer'][:n_keep, :, :length]
    cache['k'], cache['v'] = key, value


def gather_kv(cache, row_indices, init_capacity=264):
    """Gather arbitrary batch rows `row_indices` into a fresh buffered KV cache for branch continuation."""
    if 'k' not in cache:
        return {'step': cache.get('step', 1), 'mode': 'buffer', 'init_capacity': int(init_capacity)}
    k_sel = cache['k'].index_select(0, row_indices)
    v_sel = cache['v'].index_select(0, row_indices)
    bsz, n_heads, length, head_dim = k_sel.shape
    cap = max(int(init_capacity), length + 64)
    k_buf = k_sel.new_empty((bsz, n_heads, cap, head_dim))
    v_buf = v_sel.new_empty((bsz, n_heads, cap, head_dim))
    k_buf[:, :, :length].copy_(k_sel)
    v_buf[:, :, :length].copy_(v_sel)
    return {
        'step': cache.get('step', 1),
        'mode': 'buffer',
        'init_capacity': cap,
        'k_buffer': k_buf,
        'v_buffer': v_buf,
        'k': k_buf[:, :, :length],
        'v': v_buf[:, :, :length],
    }
