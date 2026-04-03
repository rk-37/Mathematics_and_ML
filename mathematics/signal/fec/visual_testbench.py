#!/usr/bin/env python3
"""
FEC Visual Testbench
====================
Runs a single encode → corrupt → decode round-trip and prints a
byte-level visual breakdown of each stage.

Usage:
    python visual_testbench.py --impl xor_parity
    python visual_testbench.py --impl xor_parity --ber 0.05 --block-size 4
    python visual_testbench.py --impl reed_solomon --no-color
"""

import argparse
import importlib
import os
import random
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np

from noise import apply_noise

IMPL_ROOT = Path(__file__).parent / "implementations"

# ─── ANSI colour helpers ──────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"

_USE_COLOR = True

def c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if _USE_COLOR else text

# ─── Build helpers (shared with main testbench) ──────────────────────────

def compile_impl(impl_name: str) -> None:
    impl_path = IMPL_ROOT / impl_name
    if not impl_path.exists():
        impls = [d.name for d in sorted(IMPL_ROOT.iterdir()) if d.is_dir()]
        print(f"[error] '{impl_name}' not found. Available: {', '.join(impls)}", file=sys.stderr)
        sys.exit(1)

    candidates = [
        Path(sys.executable).parent / "maturin",
        Path(sysconfig.get_path("scripts", "posix_user")) / "maturin",
    ]
    maturin = next((p for p in candidates if p.exists()), None)
    if maturin is None:
        found = shutil.which("maturin")
        if found is None:
            print("[error] maturin not found. Install with: pip install maturin", file=sys.stderr)
            sys.exit(1)
        maturin = Path(found)

    cargo_bin = Path.home() / ".cargo" / "bin"
    env = os.environ.copy()
    env["PATH"] = str(cargo_bin) + os.pathsep + env.get("PATH", "")

    print(f"[build] Compiling '{impl_name}'...")
    build = subprocess.run(
        [str(maturin), "build", "--release", "--out", "dist"],
        cwd=impl_path, capture_output=True, text=True, env=env,
    )
    if build.returncode != 0:
        print(f"[error] Build failed:\n{build.stderr}", file=sys.stderr)
        sys.exit(1)

    wheels = list((impl_path / "dist").glob("*.whl"))
    if not wheels:
        print("[error] No wheel produced.", file=sys.stderr)
        sys.exit(1)

    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[-1])],
        capture_output=True, text=True,
    )
    if install.returncode != 0:
        print(f"[error] pip install failed:\n{install.stderr}", file=sys.stderr)
        sys.exit(1)
    print("[build] Done.\n")


def import_impl(impl_name: str):
    try:
        return importlib.import_module(impl_name)
    except ImportError as exc:
        print(f"[error] Could not import '{impl_name}': {exc}", file=sys.stderr)
        sys.exit(1)

# ─── Visual display ───────────────────────────────────────────────────────

def _chunks(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


def fmt_byte(byte: int, style: str = "") -> str:
    """Format a single byte as two hex digits, optionally styled."""
    s = f"{byte:02X}"
    return c(style, s) if style else s


def print_section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{c(BOLD + WHITE, title)}")
    print(c(DIM, bar))


def display_input(data: bytes, block_size: int) -> None:
    print_section(f"INPUT  ({len(data)} bytes, block size {block_size})")
    for block_idx, chunk in enumerate(_chunks(data, block_size)):
        byte_strs = [fmt_byte(b) for b in chunk]
        print(f"  block {block_idx:>2}: [ {' '.join(byte_strs)} ]")


def display_encoded(
    encoded: bytes,
    block_size: int,
    parity_bytes: int,
) -> None:
    enc_block = block_size + parity_bytes
    print_section(
        f"ENCODED  ({len(encoded)} bytes — "
        f"{block_size}B data + {c(YELLOW, f'{parity_bytes}B parity')} per block)"
    )
    for block_idx, chunk in enumerate(_chunks(encoded, enc_block)):
        data_part   = chunk[:block_size]
        parity_part = chunk[block_size:]
        data_strs   = [fmt_byte(b) for b in data_part]
        parity_strs = [fmt_byte(b, YELLOW) for b in parity_part]
        all_strs = data_strs + [c(DIM, "│")] + parity_strs
        print(f"  block {block_idx:>2}: [ {' '.join(all_strs)} ]")


def display_noisy(
    encoded: bytes,
    noisy: bytes,
    block_size: int,
    parity_bytes: int,
) -> None:
    enc_block = block_size + parity_bytes
    n_flips = sum(bin(a ^ b).count("1") for a, b in zip(encoded, noisy))
    n_corrupted_blocks = sum(
        1 for a, b in zip(_chunks(encoded, enc_block), _chunks(noisy, enc_block))
        if a != b
    )
    print_section(
        f"NOISY  ({n_flips} bit flip{'s' if n_flips != 1 else ''}, "
        f"{n_corrupted_blocks} block{'s' if n_corrupted_blocks != 1 else ''} affected)"
    )
    for block_idx, (orig_chunk, noisy_chunk) in enumerate(
        zip(_chunks(encoded, enc_block), _chunks(noisy, enc_block))
    ):
        data_part   = noisy_chunk[:block_size]
        parity_part = noisy_chunk[block_size:]
        data_strs = [
            fmt_byte(nb, RED if nb != ob else "")
            for ob, nb in zip(orig_chunk[:block_size], data_part)
        ]
        parity_strs = [
            fmt_byte(nb, RED + BOLD if nb != ob else YELLOW)
            for ob, nb in zip(orig_chunk[block_size:], parity_part)
        ]
        all_strs = data_strs + [c(DIM, "│")] + parity_strs
        changed = orig_chunk != noisy_chunk
        tag = c(RED, " ← noise") if changed else c(DIM, "")
        print(f"  block {block_idx:>2}: [ {' '.join(all_strs)} ]{tag}")


def display_decoded(
    original: bytes,
    decoded: bytes,
    error_blocks: list,
    block_size: int,
    parity_bytes: int,
) -> None:
    enc_block   = block_size + parity_bytes
    n_blocks    = (len(original) + block_size - 1) // block_size
    error_set   = set(error_blocks)
    correct_count = 0

    print_section(f"DECODED  ({len(decoded)} bytes)")
    for block_idx, (orig_chunk, dec_chunk) in enumerate(
        zip(_chunks(original, block_size), _chunks(decoded, block_size))
    ):
        detected  = block_idx in error_set
        recovered = orig_chunk == dec_chunk

        byte_strs = [
            fmt_byte(db, GREEN if db == ob else RED)
            for ob, db in zip(orig_chunk, dec_chunk)
        ]

        if detected and recovered:
            status = c(GREEN, "✓ corrected")
            correct_count += 1
        elif not detected:
            status = c(GREEN, "✓ clean")
            correct_count += 1
        else:
            status = c(RED, "✗ error")

        print(f"  block {block_idx:>2}: [ {' '.join(byte_strs)} ]  {status}")

    bar = "─" * 60
    print(c(DIM, bar))
    match_icon = c(GREEN, "✓") if decoded[:len(original)] == original else c(RED, "✗")
    print(f"  {match_icon}  {correct_count}/{n_blocks} blocks recovered correctly")


# ─── Entry point ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FEC Visual Testbench — single encode/corrupt/decode pass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--impl", default="xor_parity",
                        help="FEC implementation to use.")
    parser.add_argument("--ber", type=float, default=0.05,
                        help="Bit error rate for the noise channel.")
    parser.add_argument("--data-size", type=int, default=32,
                        help="Input payload size in bytes.")
    parser.add_argument("--block-size", type=int, default=8,
                        help="FEC block size (data bytes per block).")
    parser.add_argument("--noise-model", choices=["uniform", "burst"], default="uniform",
                        help="Noise model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip compilation (use cached build).")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colour output.")
    return parser.parse_args()


def main() -> None:
    global _USE_COLOR

    args = parse_args()
    _USE_COLOR = not args.no_color

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not args.no_compile:
        compile_impl(args.impl)

    fec = import_impl(args.impl)
    parity_bytes: int = fec.overhead(args.block_size)

    print(c(BOLD, f"\nFEC Visual Testbench — {args.impl}"))
    print(f"  BER: {args.ber}   noise: {args.noise_model}   "
          f"block: {args.block_size}B data + {parity_bytes}B parity")

    # Generate random input
    data = bytes(random.getrandbits(8) for _ in range(args.data_size))

    # Encode
    encoded = bytes(fec.encode(data, args.block_size))

    # Corrupt
    noisy = bytes(apply_noise(encoded, args.ber, model=args.noise_model))

    # Decode
    decoded_raw, error_blocks = fec.decode(noisy, args.block_size)
    decoded = bytes(decoded_raw)

    # Display
    display_input(data, args.block_size)
    display_encoded(encoded, args.block_size, parity_bytes)
    display_noisy(encoded, noisy, args.block_size, parity_bytes)
    display_decoded(data, decoded, error_blocks, args.block_size, parity_bytes)
    print()


if __name__ == "__main__":
    main()
