// ============================================================
// src/window.rs
// Builds a Hann window of any given size.
// A window is just a list of numbers that taper from 0
// at the edges to 1 in the middle.
// ============================================================

use std::f32::consts::PI;

pub fn build_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| {
            let n = n as f32;
            let window_length = (size - 1) as f32;
            0.5 * (1.0 - (2.0 * PI * n / window_length).cos())
        })
        .collect()
}

// ── Tests ────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_zero() {
        let w = build_hann_window(8);
        assert!(w[0].abs() < 1e-6, "first sample should be ~0");
        assert!(w[7].abs() < 1e-6, "last sample should be ~0");
    }

    #[test]
    fn middle_is_one() {
        let w = build_hann_window(9);
        assert!((w[4] - 1.0).abs() < 1e-6, "middle should be ~1");
    }
}
