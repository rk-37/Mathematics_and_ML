use pyo3::prelude::*;

/// Encode data using XOR block parity.
///
/// Each `block_size`-byte chunk of data gets a single parity byte appended,
/// computed as the XOR of all bytes in that chunk.
///
/// Encoded layout: [d0, d1, ..., d_{n-1}, parity] repeated per block.
#[pyfunction]
#[pyo3(signature = (data, block_size))]
fn encode(data: &[u8], block_size: usize) -> PyResult<Vec<u8>> {
    if block_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "block_size must be greater than 0",
        ));
    }

    let num_blocks = data.len().div_ceil(block_size);
    let mut encoded = Vec::with_capacity(data.len() + num_blocks);

    for chunk in data.chunks(block_size) {
        encoded.extend_from_slice(chunk);
        let parity: u8 = chunk.iter().fold(0u8, |acc, &b| acc ^ b);
        encoded.push(parity);
    }

    Ok(encoded)
}

/// Decode data encoded with XOR block parity.
///
/// Returns `(decoded_bytes, error_block_indices)` where `error_block_indices`
/// is the list of block indices where parity failed (error detected).
/// Note: XOR parity detects an odd number of bit errors per block but cannot
/// correct them — the corrupted data bytes are still returned as-is.
#[pyfunction]
#[pyo3(signature = (data, block_size))]
fn decode(data: &[u8], block_size: usize) -> PyResult<(Vec<u8>, Vec<usize>)> {
    if block_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "block_size must be greater than 0",
        ));
    }

    let encoded_block_size = block_size + 1;
    let mut decoded = Vec::with_capacity(data.len());
    let mut error_blocks: Vec<usize> = Vec::new();

    for (block_idx, chunk) in data.chunks(encoded_block_size).enumerate() {
        if chunk.len() < 2 {
            // Malformed trailing bytes — pass through without parity check
            decoded.extend_from_slice(chunk);
            continue;
        }

        let (payload, parity_slice) = chunk.split_at(chunk.len() - 1);
        let computed_parity: u8 = payload.iter().fold(0u8, |acc, &b| acc ^ b);

        if computed_parity != parity_slice[0] {
            error_blocks.push(block_idx);
        }

        decoded.extend_from_slice(payload);
    }

    Ok((decoded, error_blocks))
}

#[pymodule]
fn xor_parity(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    Ok(())
}
