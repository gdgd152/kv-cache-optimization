# kv_cache.py
import torch

def truncate_kv_cache(past_key_values, max_length=256):
    new_past = []
    for k, v in past_key_values:
        if k.size(2) <= max_length:
            new_past.append((k, v))
        else:
            new_past.append((
                k[:, :, -max_length:, :],
                v[:, :, -max_length:, :]
            ))
    return tuple(new_past)


def streaming_kv_cache(past_key_values, sink_size=4, window_size=256):
    new_past = []
    max_len = sink_size + window_size

    for k, v in past_key_values:
        seq_len = k.size(2)

        if seq_len <= max_len:
            new_past.append((k, v))
            continue

        k_new = torch.cat([
            k[:, :, :sink_size, :],
            k[:, :, -window_size:, :]
        ], dim=2)

        v_new = torch.cat([
            v[:, :, :sink_size, :],
            v[:, :, -window_size:, :]
        ], dim=2)

        new_past.append((k_new, v_new))

    return tuple(new_past)


def apply_kv_optimization(past_key_values, method=None, **kwargs):
    if method == "truncate":
        return truncate_kv_cache(past_key_values, **kwargs)
    elif method == "streaming":
        return streaming_kv_cache(past_key_values, **kwargs)
    return past_key_values