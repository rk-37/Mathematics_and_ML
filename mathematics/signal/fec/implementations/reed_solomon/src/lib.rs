///! Reed-Solomon error correction over GF(2^8).
///!
///! Parameters
///! ----------
///! ECC_SYMBOLS = 4 parity bytes per block.
///! Can detect up to 4 symbol errors per block, correct up to 2.
///! Maximum block_size = 251 (block_size + 4 ≤ 255).
///!
///! Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1  (0x11D)
///! Generator roots: α^0, α^1, α^2, α^3  where α = 2.

use pyo3::prelude::*;
use std::sync::OnceLock;

const ECC: usize = 4;
const GF_PRIM: u16 = 0x11D;

// ─── GF(2^8) arithmetic ───────────────────────────────────────────────────

static TABLES: OnceLock<([u8; 512], [u8; 256])> = OnceLock::new();

fn tables() -> &'static ([u8; 512], [u8; 256]) {
    TABLES.get_or_init(|| {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x = 1u16;
        for i in 0..255usize {
            exp[i] = x as u8;
            log[x as usize] = i as u8;
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= GF_PRIM;
            }
        }
        // Double the exp table to avoid modulo in hot paths.
        for i in 255..512 {
            exp[i] = exp[i - 255];
        }
        (exp, log)
    })
}

#[inline(always)]
fn gf_exp(i: usize) -> u8 {
    tables().0[i]
}

#[inline(always)]
fn gf_log(x: u8) -> usize {
    tables().1[x as usize] as usize
}

fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    gf_exp(gf_log(a) + gf_log(b))
}

fn gf_div(a: u8, b: u8) -> u8 {
    debug_assert!(b != 0, "GF division by zero");
    if a == 0 {
        return 0;
    }
    gf_exp((gf_log(a) + 255 - gf_log(b)) % 255)
}

fn gf_pow(x: u8, n: usize) -> u8 {
    if n == 0 {
        return 1;
    }
    if x == 0 {
        return 0;
    }
    gf_exp((gf_log(x) * n) % 255)
}

fn gf_inv(x: u8) -> u8 {
    debug_assert!(x != 0, "GF inverse of zero");
    gf_exp(255 - gf_log(x))
}

// ─── Polynomial operations (highest-degree-first storage) ────────────────
//
// Polynomial p stored as [p_n, p_{n-1}, ..., p_1, p_0] where p_n is the
// leading (highest-degree) coefficient.
// Evaluation by Horner: fold left → acc = acc*x XOR coeff.

fn poly_eval(poly: &[u8], x: u8) -> u8 {
    poly.iter().fold(0u8, |acc, &c| gf_mul(acc, x) ^ c)
}

fn poly_scale(poly: &[u8], s: u8) -> Vec<u8> {
    poly.iter().map(|&c| gf_mul(c, s)).collect()
}

fn poly_add(p: &[u8], q: &[u8]) -> Vec<u8> {
    let len = p.len().max(q.len());
    let mut r = vec![0u8; len];
    let p_off = len - p.len();
    let q_off = len - q.len();
    for (i, &c) in p.iter().enumerate() {
        r[p_off + i] ^= c;
    }
    for (i, &c) in q.iter().enumerate() {
        r[q_off + i] ^= c;
    }
    r
}

fn poly_mul(p: &[u8], q: &[u8]) -> Vec<u8> {
    let mut r = vec![0u8; p.len() + q.len() - 1];
    for (i, &a) in p.iter().enumerate() {
        for (j, &b) in q.iter().enumerate() {
            r[i + j] ^= gf_mul(a, b);
        }
    }
    r
}

// ─── Generator polynomial ─────────────────────────────────────────────────
//
// g(x) = ∏_{i=0}^{ECC-1} (x + α^i)
// Stored highest-degree-first; degree = ECC; leading coeff = 1 (monic).

fn gen_poly() -> Vec<u8> {
    tables(); // ensure initialised
    let mut g = vec![1u8];
    for i in 0..ECC {
        // (x + α^i) in highest-degree-first = [1, gf_pow(2, i)]
        g = poly_mul(&g, &[1, gf_pow(2, i)]);
    }
    g
}

// ─── Encoding ─────────────────────────────────────────────────────────────

/// Systematic RS encode of one block.
/// Codeword = [data | ECC parity bytes]  (length = data.len() + ECC)
///
/// Uses an LFSR (shift-register) to compute the remainder of
/// D(x)·x^ECC mod g(x) without touching the data bytes.
fn encode_block(data: &[u8]) -> Vec<u8> {
    let gen = gen_poly(); // monic, length ECC+1; gen[0]=1, gen[1..] are feedback taps
    let mut rem = vec![0u8; ECC];

    for &byte in data {
        // Feedback = incoming data byte XOR first (highest-degree) remainder cell.
        let feedback = byte ^ rem[0];
        // Shift the register left, clearing the last cell.
        rem.rotate_left(1);
        rem[ECC - 1] = 0;
        // XOR the feedback scaled by each tap of g (skipping the monic leading 1).
        if feedback != 0 {
            for j in 0..ECC {
                rem[j] ^= gf_mul(gen[j + 1], feedback);
            }
        }
    }

    // Systematic codeword: original data followed by the parity remainder.
    let mut codeword = data.to_vec();
    codeword.extend_from_slice(&rem);
    codeword
}

// ─── Decoding helpers ─────────────────────────────────────────────────────

fn calc_syndromes(msg: &[u8]) -> Vec<u8> {
    // S_i = msg(α^i),  i = 0 .. ECC-1
    (0..ECC).map(|i| poly_eval(msg, gf_pow(2, i))).collect()
}

/// Berlekamp-Massey: find the error-locator polynomial Λ(x).
/// Stored highest-degree-first; constant term = 1 (after stripping leading zeros).
/// Returns None if too many errors to correct (> ECC/2 errors).
fn berlekamp_massey(synd: &[u8]) -> Option<Vec<u8>> {
    let nsym = synd.len();
    let mut err_loc = vec![1u8];
    let mut old_loc = vec![1u8];

    for i in 0..nsym {
        // Discrepancy Δ = S_i + Σ_{j=1}^{L} Λ_j * S_{i-j}
        // In highest-degree-first, Λ_j = err_loc[len-1-j].
        let mut delta = synd[i];
        let el = err_loc.len();
        for j in 1..el {
            if i >= j {
                delta ^= gf_mul(err_loc[el - 1 - j], synd[i - j]);
            }
        }

        // Multiply old_loc by x (append 0 in highest-degree-first = shift right).
        old_loc.push(0);

        if delta != 0 {
            if old_loc.len() > err_loc.len() {
                let new_loc = poly_scale(&old_loc, delta);
                old_loc = poly_scale(&err_loc, gf_inv(delta));
                err_loc = new_loc;
            }
            err_loc = poly_add(&err_loc, &poly_scale(&old_loc, delta));
        }
    }

    // Strip leading zeros.
    let start = err_loc.iter().position(|&x| x != 0).unwrap_or(0);
    let err_loc = err_loc[start..].to_vec();

    let num_errors = err_loc.len().saturating_sub(1);
    if num_errors * 2 > nsym {
        return None;
    }
    Some(err_loc)
}

/// Chien search: find positions of errors by finding roots of the reversed Λ(x).
/// `err_loc_rev` is `err_loc` reversed (constant-first); evaluating highest-degree-first
/// at α^i gives zero iff α^i is a root, which maps to byte position nmess-1-i.
fn chien_search(err_loc_rev: &[u8], nmess: usize) -> Option<Vec<usize>> {
    let num_errors = err_loc_rev.len().saturating_sub(1);
    let mut err_pos = Vec::with_capacity(num_errors);
    for i in 0..nmess {
        if poly_eval(err_loc_rev, gf_pow(2, i)) == 0 {
            err_pos.push(nmess - 1 - i);
        }
    }
    if err_pos.len() == num_errors {
        Some(err_pos)
    } else {
        None // Chien search found the wrong number of roots — uncorrectable
    }
}

/// Build the errata locator from coefficient positions.
/// coef_pos[i] = n-1-err_pos[i], i.e., the exponent such that X_i = α^{coef_pos_i}.
fn errata_locator(coef_pos: &[usize]) -> Vec<u8> {
    let mut e_loc = vec![1u8];
    for &i in coef_pos {
        e_loc = poly_mul(&e_loc, &[1, gf_pow(2, i)]);
    }
    e_loc
}

/// Compute error evaluator polynomial Ω(x) = S(x)·Λ(x) mod x^{nsym+1}.
/// All inputs/outputs are highest-degree-first.
/// `synd_rev` is syndromes reversed so that the array represents S(x) in that convention.
fn error_evaluator(synd_rev: &[u8], err_loc: &[u8], nsym: usize) -> Vec<u8> {
    let product = poly_mul(synd_rev, err_loc);
    let len = product.len();
    // mod x^{nsym+1}: keep the last nsym+1 elements (low-degree terms in HDF).
    if len > nsym + 1 {
        product[len - nsym - 1..].to_vec()
    } else {
        product
    }
}

/// Forney algorithm: compute error magnitudes and apply corrections.
fn correct_errata(msg: &[u8], synd: &[u8], err_pos: &[usize]) -> Vec<u8> {
    let n = msg.len();
    // coef_pos[i] = exponent k such that X_i = α^k, where X_i is the error locator value.
    let coef_pos: Vec<usize> = err_pos.iter().map(|&p| n - 1 - p).collect();

    let e_loc = errata_locator(&coef_pos);

    // Build Ω(x) = S_reversed(x) · Λ(x)  mod  x^{e_loc.len()}
    // synd reversed = [S_{ECC-1}, ..., S_0] represents S(x) = S_0 + S_1*x + ...
    let synd_rev: Vec<u8> = synd.iter().rev().copied().collect();
    let omega = error_evaluator(&synd_rev, &e_loc, e_loc.len() - 1);

    let x_vals: Vec<u8> = coef_pos.iter().map(|&p| gf_pow(2, p)).collect();
    let mut e = vec![0u8; n];

    for (i, &xi) in x_vals.iter().enumerate() {
        let xi_inv = gf_inv(xi);

        // Λ'(Xi^{-1}) computed as product form: ∏_{j≠i} (1 + Xi^{-1}·Xj)
        let mut err_loc_prime = 1u8;
        for (j, &xj) in x_vals.iter().enumerate() {
            if j != i {
                err_loc_prime = gf_mul(err_loc_prime, 1u8 ^ gf_mul(xi_inv, xj));
            }
        }
        if err_loc_prime == 0 {
            continue; // Degenerate; skip
        }

        // Magnitude = Xi · Ω(Xi^{-1}) / Λ'(Xi^{-1})
        let omega_val = poly_eval(&omega, xi_inv);
        let magnitude = gf_div(gf_mul(xi, omega_val), err_loc_prime);
        e[err_pos[i]] = magnitude;
    }

    msg.iter().zip(e.iter()).map(|(&m, &ei)| m ^ ei).collect()
}

// ─── Top-level block decode ───────────────────────────────────────────────

/// Returns (decoded_data, error_detected, successfully_corrected).
fn decode_block(codeword: &[u8]) -> (Vec<u8>, bool, bool) {
    let k = codeword.len().saturating_sub(ECC);

    let synd = calc_syndromes(codeword);
    if synd.iter().all(|&s| s == 0) {
        return (codeword[..k].to_vec(), false, true);
    }

    // Find error locator polynomial
    let err_loc = match berlekamp_massey(&synd) {
        Some(loc) => loc,
        None => return (codeword[..k].to_vec(), true, false),
    };

    // Chien search needs the reversed polynomial
    let err_loc_rev: Vec<u8> = err_loc.iter().rev().copied().collect();
    let err_pos = match chien_search(&err_loc_rev, codeword.len()) {
        Some(pos) => pos,
        None => return (codeword[..k].to_vec(), true, false),
    };

    let corrected = correct_errata(codeword, &synd, &err_pos);

    // Verify: syndromes should all be zero after correction
    if calc_syndromes(&corrected).iter().all(|&s| s == 0) {
        (corrected[..k].to_vec(), true, true)
    } else {
        (codeword[..k].to_vec(), true, false)
    }
}

// ─── PyO3 interface ──────────────────────────────────────────────────────

/// Encode data using Reed-Solomon (RS) error correction.
///
/// Data is split into `block_size`-byte chunks. Each chunk is encoded
/// with 4 parity bytes appended.  Resulting encoded length:
///     ⌈len(data) / block_size⌉ × (block_size + 4) bytes.
///
/// Constraints: 1 ≤ block_size ≤ 251.
#[pyfunction]
#[pyo3(signature = (data, block_size))]
fn encode(data: &[u8], block_size: usize) -> PyResult<Vec<u8>> {
    if block_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("block_size must be > 0"));
    }
    if block_size > 251 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "block_size must be ≤ 251 (block_size + ECC ≤ 255)",
        ));
    }
    tables(); // ensure GF tables are ready

    let mut out =
        Vec::with_capacity(data.len() + (data.len() / block_size + 1) * ECC);
    for chunk in data.chunks(block_size) {
        out.extend_from_slice(&encode_block(chunk));
    }
    Ok(out)
}

/// Decode data encoded with Reed-Solomon.
///
/// Returns `(decoded_bytes, error_block_indices)`.
/// `error_block_indices` lists blocks where errors were detected (whether or
/// not they were successfully corrected).  Blocks with ≤ 2 symbol errors are
/// corrected; blocks with more errors are returned as-is.
#[pyfunction]
#[pyo3(signature = (data, block_size))]
fn decode(data: &[u8], block_size: usize) -> PyResult<(Vec<u8>, Vec<usize>)> {
    if block_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("block_size must be > 0"));
    }
    tables();

    let enc_block = block_size + ECC;
    let mut decoded = Vec::new();
    let mut error_blocks = Vec::new();

    for (idx, chunk) in data.chunks(enc_block).enumerate() {
        if chunk.len() <= ECC {
            // Too short to contain any data — pass through
            decoded.extend_from_slice(chunk);
            continue;
        }
        let (data_out, had_error, _corrected) = decode_block(chunk);
        if had_error {
            error_blocks.push(idx);
        }
        decoded.extend_from_slice(&data_out);
    }

    Ok((decoded, error_blocks))
}

/// Returns the number of parity bytes appended per data block.
/// For this RS implementation this is always 4, regardless of block_size.
#[pyfunction]
fn overhead(_block_size: usize) -> usize {
    ECC
}

#[pymodule]
fn reed_solomon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(overhead, m)?)?;
    Ok(())
}
