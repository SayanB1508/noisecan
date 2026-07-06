// ============================================================
// src/stft.rs
// Short-Time Fourier Transform module.
// Converts audio frames to frequency spectra and back.
// ============================================================

use rustfft::{FftPlanner, Fft, num_complex::Complex};
use std::sync::Arc;

// ── The struct that holds all STFT state ─────────────────────
pub struct Stft {
    pub fft_size:    usize,
    pub hop_size:    usize,
    pub num_bins:    usize,
    fft:             Arc<dyn Fft<f32>>,
    ifft:            Arc<dyn Fft<f32>>,
    overlap_buf:     Vec<f32>,
}

// ── Methods on the struct ────────────────────────────────────
impl Stft {

    // ── Constructor: create a new Stft ───────────────────────
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            fft_size,
            hop_size,
            num_bins:    fft_size / 2 + 1,
            fft:         planner.plan_fft_forward(fft_size),
            ifft:        planner.plan_fft_inverse(fft_size),
            overlap_buf: vec![0.0; fft_size],
        }
    }

    // ── Step 1: multiply frame by window ─────────────────────
    // Input:  raw audio samples  &[f32]
    //         hann window values &[f32]
    // Output: complex numbers ready for FFT
    pub fn apply_window(
        &self,
        frame:  &[f32],
        window: &[f32],
    ) -> Vec<Complex<f32>> {
        frame
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect()
    }

    // ── Step 2: forward FFT ──────────────────────────────────
    // Input:  windowed complex samples (time domain)
    // Output: frequency spectrum (frequency domain)
    pub fn forward(
        &self,
        mut buf: Vec<Complex<f32>>,
    ) -> Vec<Complex<f32>> {
        self.fft.process(&mut buf);
        buf
    }

    // ── Step 3: inverse FFT + overlap-add ────────────────────
    // Input:  modified frequency spectrum
    // Effect: accumulates time-domain samples into overlap_buf
    pub fn inverse(
        &mut self,
        mut spec: Vec<Complex<f32>>,
    ) {
        let n = self.fft_size;

        // Restore Hermitian symmetry in the upper half
        // (needed so IFFT gives real-valued output)
        for i in 1..(n / 2) {
            spec[n - i] = spec[i].conj();
        }

        // Run the inverse FFT in-place
        self.ifft.process(&mut spec);

        // Scale (rustfft does not normalise automatically)
        // then add into the overlap accumulation buffer
        let scale = 1.0 / n as f32;
        for (i, c) in spec.iter().enumerate() {
            self.overlap_buf[i] += c.re * scale;
        }
    }

    // ── Step 4: extract ready output samples ─────────────────
    // Takes the first hop_size samples out of overlap_buf,
    // shifts the buffer left, zeros the tail.
    // Returns the extracted samples → send these to speaker.
    pub fn drain_hop(&mut self) -> Vec<f32> {
        let h = self.hop_size;
        let n = self.fft_size;

        // Copy the ready samples
        let out = self.overlap_buf[..h].to_vec();

        // Shift remaining samples to the front
        self.overlap_buf.copy_within(h.., 0);

        // Zero the tail so next frame adds into clean zeros
        self.overlap_buf[n - h..].fill(0.0);

        out
    }

    // ── Helper: extract magnitudes from spectrum ─────────────
    // Returns the amplitude at each frequency bin.
    // Used by the noise estimator and spectral subtraction.
    pub fn magnitudes(
        spec: &[Complex<f32>],
        num_bins: usize,
    ) -> Vec<f32> {
        spec[..num_bins]
            .iter()
            .map(|c| c.norm())
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn flat_window(size: usize) -> Vec<f32> {
        vec![1.0; size]
    }

    fn hann(size: usize) -> Vec<f32> {
        (0..size)
            .map(|n| {
                let n = n as f32;
                let window_length = (size - 1) as f32;
                0.5 * (1.0 - (2.0 * PI * n / window_length).cos())
            })
            .collect()
    }

    #[test]
    fn window_zeroes_edges() {
        let s = Stft::new(8, 4);
        let w = hann(8);
        let frame = vec![1.0f32; 8];
        let result = s.apply_window(&frame, &w);
        assert!(result[0].re.abs() < 1e-6);
        assert!(result[7].re.abs() < 1e-6);
    }

    #[test]
    fn forward_gives_correct_length() {
        let s = Stft::new(8, 4);
        let frame = vec![1.0f32; 8];
        let w = flat_window(8);
        let windowed = s.apply_window(&frame, &w);
        let spec = s.forward(windowed);
        assert_eq!(spec.len(), 8);
    }

    #[test]
    fn round_trip_preserves_signal() {
        let mut s = Stft::new(8, 4);
        let frame: Vec<f32> =
            vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let w = flat_window(8);
        let windowed = s.apply_window(&frame, &w);
        let spec     = s.forward(windowed);
        s.inverse(spec);
        let out = s.drain_hop();
        // First hop_size samples should match original
        for (a, b) in out.iter().zip(frame.iter()) {
            assert!((a - b).abs() < 1e-5,
                "got {a}, expected {b}");
        }
    }
}
