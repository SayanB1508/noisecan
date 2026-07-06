// ============================================================
// src/spectral_subtraction.rs
// The core noise removal step.
// Subtracts the noise estimate from each frequency bin,
// then rebuilds the complex spectrum with cleaned magnitudes.
// ============================================================

use rustfft::num_complex::Complex;

// ── Parameters for spectral subtraction ──────────────────────
pub struct SpectralSubtractor {
    alpha: f32,  // subtraction strength  (try 1.0 – 2.0)
    beta:  f32,  // spectral floor factor (try 0.001 – 0.01)
}

impl SpectralSubtractor {

    // ── Constructor ──────────────────────────────────────────
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self { alpha, beta }
    }

    // ── Main method: subtract noise from spectrum ────────────
    //
    // spectrum     : the full complex FFT output (length N)
    // magnitude    : |spectrum[i]| for i in 0..num_bins
    // noise_est    : estimated noise magnitude per bin
    // num_bins     : N/2 + 1  (the useful bins)
    //
    // Returns: the modified spectrum (same length N)
    // with noise reduced in bins 0..num_bins.
    // Upper bins are left as-is — inverse() will mirror them.
    pub fn subtract(
        &self,
        mut spectrum:  Vec<Complex<f32>>,
        magnitude:    &[f32],
        noise_est:    &[f32],
        num_bins:      usize,
    ) -> Vec<Complex<f32>> {

        for i in 0..num_bins {
            let noisy_mag = magnitude[i];
            let noise     = noise_est[i];

            // Spectral subtraction formula:
            // clean = max( noisy - alpha*noise,
            //              beta*noise )
            //
            // The max() clamps to a spectral floor.
            // Without it, over-subtracted bins go negative
            // which creates metallic crackling ("musical noise").
            let clean_mag = (noisy_mag - self.alpha * noise)
                            .max(self.beta * noise);

            // Preserve the original phase of this bin.
            // Phase encodes TIMING of the signal —
            // changing it makes speech unintelligible.
            let phase = spectrum[i].arg();

            // Rebuild the complex number from
            // (new magnitude, original phase):
            // re = clean_mag * cos(phase)
            // im = clean_mag * sin(phase)
            spectrum[i] = Complex::from_polar(clean_mag, phase);
        }

        spectrum
    }
}

// ── Tests ────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_only_signal_goes_to_floor() {
        let sub = SpectralSubtractor::new(1.0, 0.002);
        // Build a fake spectrum where magnitude = noise_est
        // After subtraction: clean_mag = max(0, 0.002*noise)
        let noise_est = vec![1.0f32; 4];
        let magnitude = vec![1.0f32; 4]; // same as noise
        let spectrum: Vec<Complex<f32>> = magnitude
            .iter()
            .map(|&m| Complex::new(m, 0.0))
            .collect();
        // Pad to length 8 (as if fft_size=8, num_bins=4)
        let mut spec = spectrum.clone();
        spec.extend(vec![Complex::new(0.0, 0.0); 4]);

        let result = sub.subtract(spec, &magnitude, &noise_est, 4);
        // clean_mag should be beta * noise = 0.002
        assert!((result[0].norm() - 0.002).abs() < 1e-5);
    }

    #[test]
    fn strong_signal_survives() {
        let sub = SpectralSubtractor::new(1.0, 0.002);
        let noise_est = vec![0.1f32; 4];
        let magnitude = vec![1.0f32; 4];  // 10x louder than noise
        let mut spec: Vec<Complex<f32>> = magnitude
            .iter()
            .map(|&m| Complex::new(m, 0.0))
            .collect();
        spec.extend(vec![Complex::new(0.0, 0.0); 4]);

        let result = sub.subtract(spec, &magnitude, &noise_est, 4);
        // clean_mag = 1.0 - 0.1 = 0.9
        assert!((result[0].norm() - 0.9).abs() < 1e-5);
    }
}
