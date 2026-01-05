import random
import math

# === Global shared parameters ===
MODULUS = 982451653        # Large prime
LINEAR_MULTIPLIER = 7919
LINEAR_OFFSET = 42


# === Math utilities ===

def extended_gcd(a, b):
    """Extended Euclidean Algorithm"""
    if b == 0:
        return (a, 1, 0)
    gcd, x1, y1 = extended_gcd(b, a % b)
    return (gcd, y1, x1 - (a // b) * y1)


def modular_inverse(value, modulus):
    """Find multiplicative inverse modulo modulus"""
    gcd, inverse, _ = extended_gcd(value % modulus, modulus)
    if gcd != 1:
        return None
    return inverse % modulus


# === Shared transformation (same role as sharedFunction) ===

def shared_linear_transform(x):
    """Public one-way linear transformation"""
    return (LINEAR_MULTIPLIER * x + LINEAR_OFFSET) % MODULUS


# === Optimized Yao's Millionaires Protocol ===
# Highest must be greater than Sender

def Yao_Millionaires_Protocol(
    comparison_threshold,
    sender_value,
    max_range,
    probe_value,
    random_mask=None
):
    """
    Determines whether sender_value >= comparison_threshold
    without revealing sender_value.
    """

    # Choose random blinding factor
    if random_mask is None:
        random_mask = random.randint(1, 500)

    r = random_mask
    threshold_index = round(comparison_threshold)

    # Encode an arithmetic progression (hidden list)
    base_encoding = (
        LINEAR_MULTIPLIER * (probe_value - r * sender_value) + LINEAR_OFFSET
    ) % MODULUS

    encoding_step = (LINEAR_MULTIPLIER * r) % MODULUS

    # Compute probe hash
    probe_hash = shared_linear_transform(probe_value)
    probe_hash_plus_one = probe_hash + 1

    # Prepare fast solver
    inverse_step = modular_inverse(encoding_step, MODULUS)

    # Fallback: linear scan (should never happen with valid params)
    if inverse_step is None:
        for i in range(max_range):
            encoded = (base_encoding + encoding_step * i) % MODULUS
            if i >= threshold_index:
                encoded += 1
            if encoded == probe_hash_plus_one:
                return True
        return False

    # Solve encoded(i) == target
    def solution_exists_in_range(target, start, end):
        if start >= end:
            return False
        index = ((target - base_encoding) % MODULUS) * inverse_step % MODULUS
        return start <= index < end

    # Region 1: before threshold (no +1 applied)
    if solution_exists_in_range(
        probe_hash_plus_one,
        0,
        min(threshold_index, max_range)
    ):
        return True

    # Region 2: after threshold (+1 applied)
    if solution_exists_in_range(
        probe_hash,
        max(threshold_index, 0),
        max_range
    ):
        return True

    return False
