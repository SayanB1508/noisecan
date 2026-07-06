// ============================================================
// src/vad.rs
// Voice Activity Detector.
// Decides whether a frame of audio is speech or silence.
// ============================================================

// ── The struct that holds VAD state ──────────────────────────
pub struct Vad {
    threshold: f32,   // energy level below which = silence
}

impl Vad {

    // ── Constructor ──────────────────────────────────────────
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    // ── Main function: is this frame silent? ─────────────────
    // Computes mean energy of the frame.
    // Returns true  = silence (use to update noise estimate)
    //         false = speech  (freeze noise estimate)
    pub fn is_silence(&self, frame: &[f32]) -> bool {
        let energy = Self::mean_energy(frame);
        energy < self.threshold
    }

    // ── Helper: compute mean squared energy of a frame ───────
    fn mean_energy(frame: &[f32]) -> f32 {
        let sum_sq: f32 = frame
            .iter()
            .map(|&s| s * s)        // square each sample
            .sum();                  // add them all up
        sum_sq / frame.len() as f32  // divide by count
    }

    // ── Tune helper: print energy for threshold calibration ──
    //pub fn print_energy(&self, frame: &[f32]) {
      //  let e = Self::mean_energy(frame);
        //println!("frame energy: {:.6}  threshold: {:.6}  -> {}",
          //  e, self.threshold,
            //if e < self.threshold { "SILENCE" } else { "SPEECH" });
    //}
}

// ── Tests ────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_detected() {
        let vad = Vad::new(0.01);
        let silent = vec![0.001f32; 512];
        assert!(vad.is_silence(&silent));
    }

    #[test]
    fn speech_detected() {
        let vad = Vad::new(0.01);
        let loud = vec![0.5f32; 512];
        assert!(!vad.is_silence(&loud));
    }
}
