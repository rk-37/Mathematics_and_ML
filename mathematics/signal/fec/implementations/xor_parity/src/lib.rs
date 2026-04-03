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
        let mut parity = 0;
        for bit in 0..block_size {
            parity = parity ^ chunk[bit];
        }
        encoded.extend_from_slice(chunk);
        encoded.push(parity);
    }

    // //////////////////////////////////////////////////////
    // Write your encoding implementation here
    // unimplemented!("Implement XOR parity encoding");
    // //////////////////////////////////////////////////////

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

    decoded = data.to_vec();
    // //////////////////////////////////////////////////////
    // Write your decoding implementation here
    // unimplemented!("Implement XOR parity decoding");
    // //////////////////////////////////////////////////////

    Ok((decoded, error_blocks))
}

/// Returns the number of parity bytes appended per data block.
/// For XOR parity this is always 1, regardless of block_size.
#[pyfunction]
fn overhead(_block_size: usize) -> usize {
    1
}

#[pymodule]
fn xor_parity(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(overhead, m)?)?;
    Ok(())
}
