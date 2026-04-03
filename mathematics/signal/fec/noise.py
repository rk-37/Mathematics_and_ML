"""
Noise models for the FEC testbench.

Each model takes a bytearray and a target bit error rate (BER) and returns
a new bytearray with bit errors injected according to the model's statistics.
"""

import random
from typing import Optional

import numpy as np


def apply_noise(data: bytes, ber: float, model: str = "uniform", **kwargs) -> bytearray:
    """
    Inject bit errors into data according to the chosen noise model.

    Args:
        data:   Input bytes to corrupt.
        ber:    Target bit error rate in [0.0, 1.0].
        model:  "uniform" for i.i.d. errors, "burst" for burst-error channel.
        **kwargs: Model-specific parameters (see individual functions).

    Returns:
        A new bytearray with errors injected.
    """
    if not (0.0 <= ber <= 1.0):
        raise ValueError(f"ber must be in [0, 1], got {ber}")
    if model == "uniform":
        return _uniform_noise(data, ber)
    elif model == "burst":
        return _burst_noise(data, ber, **kwargs)
    else:
        raise ValueError(f"Unknown noise model '{model}'. Choose 'uniform' or 'burst'.")


def _uniform_noise(data: bytes, ber: float) -> bytearray:
    """
    Independently and identically distributed (i.i.d.) bit errors.

    Each bit is flipped independently with probability `ber`.
    Uses a binomial draw to pick the total error count, then samples
    unique bit positions — equivalent to i.i.d. flips for BER << 1.
    """
    result = bytearray(data)
    n_bits = len(data) * 8
    if n_bits == 0 or ber == 0.0:
        return result

    n_errors = int(np.random.binomial(n_bits, ber))
    if n_errors == 0:
        return result

    # Sample without replacement (fine for BER << 1; for very high BER the
    # distribution differs slightly from true i.i.d., which is acceptable).
    error_positions = random.sample(range(n_bits), min(n_errors, n_bits))
    for pos in error_positions:
        result[pos >> 3] ^= 1 << (pos & 7)

    return result


def _burst_noise(
    data: bytes,
    ber: float,
    p_good_to_bad: Optional[float] = None,
    p_bad_to_good: float = 0.1,
    ber_good_fraction: float = 0.01,
    ber_bad_fraction: float = 10.0,
) -> bytearray:
    """
    Burst noise via a two-state Gilbert-Elliott Markov channel.

    States:
      Good — low per-bit error probability.
      Bad  — high per-bit error probability (a "burst").

    The channel alternates between states with transition probabilities
    `p_good_to_bad` and `p_bad_to_good`. Parameters are tuned so that
    the stationary-state BER matches the requested `ber`.

    Args:
        p_good_to_bad:      P(Good→Bad) per bit. If None, auto-tuned so that
                            the long-run BER matches `ber`.
        p_bad_to_good:      P(Bad→Good) per bit. Controls average burst length
                            (~1/p_bad_to_good bits). Default: 0.1 (≈10-bit bursts).
        ber_good_fraction:  BER in Good state = ber * this. Default: 0.01.
        ber_bad_fraction:   BER in Bad state  = ber * this (capped at 0.5).
    """
    result = bytearray(data)
    n_bits = len(data) * 8
    if n_bits == 0 or ber == 0.0:
        return result

    ber_good = ber * ber_good_fraction
    ber_bad = min(ber * ber_bad_fraction, 0.5)

    if p_good_to_bad is None:
        # Solve for p_good_to_bad so that the stationary BER equals `ber`.
        # Stationary distribution: π_bad = p_g2b / (p_g2b + p_bad_to_good)
        # BER = π_good * ber_good + π_bad * ber_bad
        # Rearranging for p_g2b (guard against degenerate cases):
        if abs(ber_bad - ber_good) < 1e-12:
            p_good_to_bad = ber
        else:
            # π_bad = (ber - ber_good) / (ber_bad - ber_good)
            pi_bad = (ber - ber_good) / (ber_bad - ber_good)
            pi_bad = max(0.0, min(pi_bad, 0.999))
            p_good_to_bad = pi_bad * p_bad_to_good / max(1.0 - pi_bad, 1e-9)

    in_burst = False
    for pos in range(n_bits):
        # State transition
        if in_burst:
            if random.random() < p_bad_to_good:
                in_burst = False
        else:
            if random.random() < p_good_to_bad:
                in_burst = True

        err_prob = ber_bad if in_burst else ber_good
        if random.random() < err_prob:
            result[pos >> 3] ^= 1 << (pos & 7)

    return result
