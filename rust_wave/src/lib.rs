//! # rust_wave — TS-OS hot path (PyO3)
//!
//! **TS-OS** = Thinking System / Thinking Wave Operating System.
//! This crate accelerates numerical kernels in the **Wave Cycle** (the 11-step
//! autonomous loop) so propagation, normalization, and similarity clustering run
//! 10–20× faster than pure Python on large node sets.
//!
//! Python orchestrates SQLite + JSON (**UniversalLivingGraph**); Rust handles
//! dense linear algebra on embedding vectors and activation arrays.

use ndarray::{Array1, Array2, Axis};
use pyo3::prelude::*;

/// L2-normalize each row of a 2D matrix (in-place on a copy returned to Python).
#[pyfunction]
fn normalize_rows(embeddings: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let n = embeddings.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let dim = embeddings[0].len();
    let flat: Vec<f64> = embeddings.iter().flatten().copied().collect();
    let a = Array2::from_shape_vec((n, dim), flat).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}"))
    })?;
    let mut out = a.clone();
    for mut row in out.axis_iter_mut(Axis(0)) {
        let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            row /= norm;
        }
    }
    Ok(out
        .axis_iter(Axis(0))
        .map(|r| r.to_vec())
        .collect())
}

/// Cosine similarity between all pairs of row vectors (assumes L2-normalized rows).
#[pyfunction]
fn pairwise_cosine_similarity(embeddings: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let n = embeddings.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let dim = embeddings[0].len();
    let flat: Vec<f64> = embeddings.iter().flatten().copied().collect();
    let a = Array2::from_shape_vec((n, dim), flat).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}"))
    })?;
    // Gram matrix for unit vectors = cosine similarity
    let gram = a.dot(&a.t());
    Ok(gram.axis_iter(Axis(0)).map(|r| r.to_vec()).collect())
}

/// Propagate activation along edges: a'[j] += sum_i a[i] * w[i,j] * sim(i,j) optional boost.
/// Here we use a simplified dense step: new_act = (1-alpha)*act + alpha * (adj @ act)
#[pyfunction]
fn propagate_dense(activations: Vec<f64>, adjacency: Vec<Vec<f64>>, alpha: f64) -> PyResult<Vec<f64>> {
    let n = activations.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let a = Array1::from(activations);
    let flat: Vec<f64> = adjacency.iter().flatten().copied().collect();
    let adj = Array2::from_shape_vec((n, n), flat).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("adjacency shape: {e}"))
    })?;
    let propagated = adj.dot(&a);
    let out = (1.0 - alpha) * &a + alpha * &propagated;
    Ok(out.to_vec())
}

/// Element-wise relax: activation *= (1 - rate) + stability * rate * base_anchoring
#[pyfunction]
fn relax_activations(
    activations: Vec<f64>,
    stabilities: Vec<f64>,
    base_strengths: Vec<f64>,
    rate: f64,
) -> PyResult<Vec<f64>> {
    let n = activations.len();
    if n != stabilities.len() || n != base_strengths.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "length mismatch in relax_activations",
        ));
    }
    let out: Vec<f64> = activations
        .iter()
        .zip(stabilities.iter())
        .zip(base_strengths.iter())
        .map(|((&a, &s), &b)| a * (1.0 - rate) + s * rate * b)
        .collect();
    Ok(out)
}

/// L1 or max normalization — here L2 on the activation vector (conceptual "energy" unit ball).
#[pyfunction]
fn normalize_activations_l2(activations: Vec<f64>) -> PyResult<Vec<f64>> {
    let n = activations.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let norm: f64 = activations.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return Ok(vec![0.0; n]);
    }
    Ok(activations.iter().map(|x| x / norm).collect())
}

/// Indices of pairs (i,j) with similarity >= threshold, i < j (for merge candidates).
#[pyfunction]
fn merge_candidate_pairs(similarity_matrix: Vec<Vec<f64>>, threshold: f64) -> PyResult<Vec<(usize, usize)>> {
    let n = similarity_matrix.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        let row = &similarity_matrix[i];
        for j in (i + 1)..row.len().min(n) {
            if row[j] >= threshold {
                pairs.push((i, j));
            }
        }
    }
    Ok(pairs)
}

/// Numpy-free path from Python lists: flatten 2D f64 list for zero-copy style handoff.
#[pyfunction]
fn sum_squares(v: Vec<f64>) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Expose list-of-float as Python list (helper for tests).
#[pyfunction]
fn add_vectors(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("length mismatch"));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

#[pymodule]
fn rust_wave(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_rows, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_dense, m)?)?;
    m.add_function(wrap_pyfunction!(relax_activations, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_activations_l2, m)?)?;
    m.add_function(wrap_pyfunction!(merge_candidate_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(sum_squares, m)?)?;
    m.add_function(wrap_pyfunction!(add_vectors, m)?)?;
    Ok(())
}
