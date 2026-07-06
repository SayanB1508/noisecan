// ============================================================
// src/noise_estimator.rs
// Maintains a running estimate of the background noise spectrum.
// Only updates during silence frames.
// ============================================================

pub struct NoiseEstimator {
    estimate:      Vec<f32>,  // one value per FFT bin
    smooth_coeff:  f32,       // how slowly the estimate changes
}

impl NoiseEstimator {

    // ── Constructor ──────────────────────────────────────────
    // num_bins   = fft_size / 2 + 1
    // smooth     = 0.95 means estimate changes at 5% per frame
    // init_value = small non-zero start (avoids division by 0)
    pub fn new(
        num_bins:   usize,
        smooth:     f32,
        init_value: f32,
    ) -> Self {
        Self {
            estimate:     vec![init_value; num_bins],
            smooth_coeff: smooth,
        }
    }

    // ── Update estimate using current magnitude spectrum ─────
    // Called every frame.
    // If is_silence = true  → update the running average
    // If is_silence = false → freeze estimate (do not change)
    pub fn update(
        &mut self,
        magnitude: &[f32],
        is_silence: bool,
    ) {
        if !is_silence {
            return; // speech frame — do not update
        }
        let alpha = self.smooth_coeff;
        for (est, &mag) in self.estimate
                               .iter_mut()
                               .zip(magnitude.iter())
        {
            // Exponential moving average:
            // new_estimate = 0.95 * old_estimate
            //              + 0.05 * current_magnitude
            *est = alpha * (*est) + (1.0 - alpha) * mag;
        }
    }

    // ── Read-only access to the current estimate ─────────────
    pub fn get(&self) -> &[f32] {
        &self.estimate
    }
}

// ── Tests ────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_moves_toward_input_during_silence() {
        let mut ne = NoiseEstimator::new(4, 0.5, 0.0);
        let mag = vec![1.0f32; 4];
        ne.update(&mag, true);  // silence → update
        // after 1 update with alpha=0.5: 0.5*0 + 0.5*1 = 0.5
        assert!((ne.get()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn estimate_frozen_during_speech() {
        let mut ne = NoiseEstimator::new(4, 0.95, 0.1);
        let mag = vec![1.0f32; 4];
        ne.update(&mag, false); // speech → no update
        assert!((ne.get()[0] - 0.1).abs() < 1e-6);
    }
}
