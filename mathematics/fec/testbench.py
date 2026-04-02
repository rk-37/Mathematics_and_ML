#!/usr/bin/env python3
"""
FEC Testbench
=============
Compiles and stress-tests Rust FEC implementations.

Usage examples:
    python testbench.py --impl xor_parity
    python testbench.py --impl xor_parity --ber 0.05 --trials 500 --noise-model burst
    python testbench.py --impl xor_parity --no-compile --seed 42
"""

import argparse
import importlib
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from noise import apply_noise

IMPL_ROOT = Path(__file__).parent / "implementations"


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def available_impls() -> list[str]:
    if not IMPL_ROOT.exists():
        return []
    return [d.name for d in sorted(IMPL_ROOT.iterdir()) if d.is_dir()]


def compile_impl(impl_name: str) -> None:
    impl_path = IMPL_ROOT / impl_name
    if not impl_path.exists():
        impls = available_impls()
        print(f"[error] Implementation '{impl_name}' not found at {impl_path}.", file=sys.stderr)
        if impls:
            print(f"        Available: {', '.join(impls)}", file=sys.stderr)
        sys.exit(1)

    # Locate maturin: check next to the Python interpreter, user scripts dir,
    # then fall back to PATH search.
    import sysconfig
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

    # Ensure cargo/rustc are visible (rustup installs to ~/.cargo/bin)
    cargo_bin = Path.home() / ".cargo" / "bin"
    env = os.environ.copy()
    env["PATH"] = str(cargo_bin) + os.pathsep + env.get("PATH", "")

    print(f"[build] Compiling '{impl_name}' with maturin...")
    build = subprocess.run(
        [str(maturin), "build", "--release", "--out", "dist"],
        cwd=impl_path,
        capture_output=True,
        text=True,
        env=env,
    )
    if build.returncode != 0:
        print(f"[error] Build failed:\n{build.stderr}", file=sys.stderr)
        sys.exit(1)

    # Install the freshly built wheel
    wheels = list((impl_path / "dist").glob("*.whl"))
    if not wheels:
        print("[error] No wheel produced by maturin build.", file=sys.stderr)
        sys.exit(1)
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[-1])],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        print(f"[error] pip install failed:\n{install.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"[build] Done.\n")


def import_impl(impl_name: str):
    try:
        return importlib.import_module(impl_name)
    except ImportError as exc:
        print(f"[error] Could not import '{impl_name}': {exc}", file=sys.stderr)
        print("        Run without --no-compile to rebuild.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def _iter_chunks(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


@dataclass
class TrialResult:
    """Outcome of a single encode → corrupt → decode round-trip."""
    original: bytes
    encoded: bytes
    noisy: bytes
    decoded: bytes
    error_blocks: list[int]            # Blocks flagged by decoder
    corrupted_block_indices: set[int]  # Blocks that were actually changed by noise
    total_blocks: int


def run_trial(
    fec,
    data: bytes,
    block_size: int,
    ber: float,
    noise_model: str,
) -> TrialResult:
    encoded: bytes = bytes(fec.encode(data, block_size))

    noisy_arr = apply_noise(encoded, ber, model=noise_model)
    noisy: bytes = bytes(noisy_arr)

    decoded_raw, error_blocks = fec.decode(noisy, block_size)
    decoded: bytes = bytes(decoded_raw)

    # Determine which encoded blocks were actually modified by the noise channel.
    enc_block_size = block_size + 1
    corrupted_block_indices: set[int] = set()
    for idx, (orig_chunk, noisy_chunk) in enumerate(
        zip(_iter_chunks(encoded, enc_block_size), _iter_chunks(noisy, enc_block_size))
    ):
        if orig_chunk != noisy_chunk:
            corrupted_block_indices.add(idx)

    total_blocks = (len(encoded) + enc_block_size - 1) // enc_block_size

    return TrialResult(
        original=data,
        encoded=encoded,
        noisy=noisy,
        decoded=decoded,
        error_blocks=list(error_blocks),
        corrupted_block_indices=corrupted_block_indices,
        total_blocks=total_blocks,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    trials: int = 0
    total_blocks: int = 0
    corrupted_blocks: int = 0
    detected_blocks: int = 0
    undetected_blocks: int = 0   # Corrupted but NOT flagged (false negative)
    false_positive_blocks: int = 0  # Flagged but NOT corrupted (shouldn't occur with XOR)
    perfect_decodes: int = 0         # decoded[:len(original)] == original

    # Bit-level stats
    input_bits: int = 0
    flipped_bits: int = 0

    overhead_ratio: float = 0.0  # encoded_size / original_size


def compute_metrics(results: list[TrialResult], block_size: int) -> Metrics:
    m = Metrics()
    m.trials = len(results)
    m.overhead_ratio = (block_size + 1) / block_size

    for r in results:
        m.total_blocks += r.total_blocks
        m.corrupted_blocks += len(r.corrupted_block_indices)

        detected_set = set(r.error_blocks)
        m.detected_blocks += len(detected_set & r.corrupted_block_indices)
        m.undetected_blocks += len(r.corrupted_block_indices - detected_set)
        m.false_positive_blocks += len(detected_set - r.corrupted_block_indices)

        if r.decoded[: len(r.original)] == r.original:
            m.perfect_decodes += 1

        m.input_bits += len(r.original) * 8
        m.flipped_bits += sum(
            bin(a ^ b).count("1")
            for a, b in zip(r.encoded, r.noisy)
        )

    return m


def print_report(m: Metrics, impl_name: str, ber: float, noise_model: str, block_size: int) -> None:
    actual_ber = m.flipped_bits / (m.input_bits * m.overhead_ratio) if m.input_bits else 0.0
    detection_rate = (
        m.detected_blocks / m.corrupted_blocks if m.corrupted_blocks > 0 else 1.0
    )
    perfect_rate = m.perfect_decodes / m.trials if m.trials > 0 else 0.0

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  FEC Testbench — {impl_name}")
    print(bar)
    print(f"  Noise model       : {noise_model}")
    print(f"  Target BER        : {ber:.4f}  ({ber*100:.2f}%)")
    print(f"  Actual BER        : {actual_ber:.4f}  ({actual_ber*100:.2f}%)")
    print(f"  Block size        : {block_size} data bytes + 1 parity byte")
    print(f"  Code overhead     : {(m.overhead_ratio - 1)*100:.1f}%  ({m.overhead_ratio:.4f}×)")
    print(bar)
    print(f"  Trials            : {m.trials}")
    print(f"  Perfect decodes   : {m.perfect_decodes}/{m.trials}  ({perfect_rate*100:.1f}%)")
    print(bar)
    print(f"  Total blocks      : {m.total_blocks}")
    print(f"  Corrupted blocks  : {m.corrupted_blocks}")
    print(f"  Detected errors   : {m.detected_blocks}  ({detection_rate*100:.2f}% of corrupted)")
    print(f"  Undetected errors : {m.undetected_blocks}  (false negatives — even-error blocks)")
    if m.false_positive_blocks:
        print(f"  False positives   : {m.false_positive_blocks}  (unexpected)")
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FEC Testbench — compile and test Rust FEC implementations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--impl",
        default="xor_parity",
        help="FEC implementation name (must match a folder under implementations/).",
    )
    parser.add_argument(
        "--ber",
        type=float,
        default=0.01,
        help="Bit error rate injected by the noise channel.",
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=256,
        help="Size of each random input payload in bytes.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="FEC block size in bytes (data bytes per parity symbol).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="Number of independent encode-corrupt-decode trials.",
    )
    parser.add_argument(
        "--noise-model",
        choices=["uniform", "burst"],
        default="uniform",
        help="Channel noise model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip the maturin compilation step (use cached build).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not args.no_compile:
        compile_impl(args.impl)

    fec = import_impl(args.impl)

    print(f"[run]   impl={args.impl}  ber={args.ber}  trials={args.trials}"
          f"  data-size={args.data_size}B  block-size={args.block_size}"
          f"  noise={args.noise_model}")

    results: list[TrialResult] = []
    for _ in range(args.trials):
        data = bytes(random.getrandbits(8) for _ in range(args.data_size))
        results.append(run_trial(fec, data, args.block_size, args.ber, args.noise_model))

    metrics = compute_metrics(results, args.block_size)
    print_report(metrics, args.impl, args.ber, args.noise_model, args.block_size)


if __name__ == "__main__":
    main()
