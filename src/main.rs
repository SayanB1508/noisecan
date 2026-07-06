// ============================================================
// src/main.rs
// Entry point. Wires all modules together.
// Mic → STFT → VAD → NoiseEstimator → SpectralSubtractor
//   → InverseSTFT → DelayBuffer → Speaker
// ============================================================

// ── Declare all modules ──────────────────────────────────────
// This tells Rust "go find src/window.rs, src/stft.rs, etc."
mod window;
mod stft;
mod vad;
mod noise_estimator;
mod spectral_subtraction;

// ── Bring specific items into scope ──────────────────────────
use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{StreamConfig, BufferSize, SampleRate};
use std::sync::{Arc, Mutex};

use stft::Stft;
use vad::Vad;
use noise_estimator::NoiseEstimator;
use spectral_subtraction::SpectralSubtractor;

// ── Configuration constants ──────────────────────────────────
const SAMPLE_RATE:   u32   = 48_000;
const CHANNELS:      u16   = 1;        // mono
const FFT_SIZE:      usize = 1024;
const HOP_SIZE:      usize = 512;      // 50% overlap
const VAD_THRESHOLD: f32   = 0.01;
const ALPHA:         f32   = 1.5;      // subtraction strength
const BETA:          f32   = 0.002;    // spectral floor
const NOISE_SMOOTH:  f32   = 0.95;     // noise estimate smoothing

// ── Delay configuration ───────────────────────────────────────
// Set DELAY_MS to however many milliseconds of delay you want.
// Examples:
//   0.0    → no extra delay (original behaviour)
//   200.0  → barely noticeable echo
//   500.0  → obvious half-second delay
//   1000.0 → dramatic one-second delay
//
// How it works: delay_buf is pre-filled with (delay_len) zeros
// at startup. Clean samples are pushed into the back of delay_buf
// and pulled from the front. The initial zeros drain out first,
// introducing exactly DELAY_MS milliseconds of silence before
// your processed voice is heard.
const DELAY_MS: f32 = 1000.0;

// ── All processing state bundled into one struct ──────────────
struct Processor {
    stft:        Stft,
    window:      Vec<f32>,
    vad:         Vad,
    noise_est:   NoiseEstimator,
    subtractor:  SpectralSubtractor,
    input_buf:   Vec<f32>,   // accumulates mic samples
    output_buf:  Vec<f32>,   // holds samples ready for speaker
    delay_buf:   Vec<f32>,   // FIFO that introduces DELAY_MS of latency
    delay_len:   usize,      // sample count of delay = DELAY_MS/1000 * SAMPLE_RATE
}

impl Processor {
    fn new() -> Self {
        let stft     = Stft::new(FFT_SIZE, HOP_SIZE);
        let num_bins = stft.num_bins;

        // Convert DELAY_MS into a sample count.
        // e.g. 500ms → (500/1000) * 48000 = 24,000 samples
        // delay_buf is pre-filled with this many zeros so that
        // the zeros drain from the front before real audio arrives,
        // giving exactly DELAY_MS of silence before the voice is heard.
        let delay_len = ((DELAY_MS / 1000.0) * SAMPLE_RATE as f32) as usize;
        let delay_buf = vec![0.0f32; delay_len];

        Self {
            window:     window::build_hann_window(FFT_SIZE),
            vad:        Vad::new(VAD_THRESHOLD),
            noise_est:  NoiseEstimator::new(num_bins,
                                            NOISE_SMOOTH,
                                            0.001),
            subtractor: SpectralSubtractor::new(ALPHA, BETA),
            stft,
            input_buf:  Vec::new(),
            output_buf: Vec::new(),
            delay_buf,
            delay_len,
        }
    }

    // Called by the input callback with fresh mic samples.
    // Processes as many complete frames as possible.
    fn push_samples(&mut self, samples: &[f32]) {
        // Accumulate incoming samples
        self.input_buf.extend_from_slice(samples);

        // Process every complete frame of FFT_SIZE samples
        while self.input_buf.len() >= FFT_SIZE {

            // ── 1. Window the frame ──────────────────────────
            let windowed = self.stft.apply_window(
                &self.input_buf[..FFT_SIZE],
                &self.window,
            );

            // ── 2. Forward FFT → frequency domain ───────────
            let spectrum = self.stft.forward(windowed);

            // ── 3. Extract magnitudes for processing ─────────
            let magnitude = Stft::magnitudes(
                &spectrum,
                self.stft.num_bins,
            );

            // ── 4. VAD: is this frame speech or silence? ─────
            let silence = self.vad.is_silence(
                &self.input_buf[..FFT_SIZE]
            );

            // ── 5. Update noise estimate (only in silence) ───
            self.noise_est.update(&magnitude, silence);

            // ── 6. Spectral subtraction ──────────────────────
            let clean_spectrum = self.subtractor.subtract(
                spectrum,
                &magnitude,
                self.noise_est.get(),
                self.stft.num_bins,
            );

            // ── 7. Inverse FFT + overlap-add ─────────────────
            self.stft.inverse(clean_spectrum);

            // ── 8. Drain output samples from overlap buffer ──
            let output = self.stft.drain_hop();

            // ── 9. Route through the delay FIFO ──────────────
            //
            // delay_buf is a FIFO (First In, First Out) queue.
            //
            // At startup it contains (delay_len) zeros.
            // New clean samples are pushed onto the BACK.
            // The same number of old samples are pulled from the FRONT.
            //
            // Because the front starts full of zeros, those drain
            // first — giving DELAY_MS ms of silence before real audio
            // reaches the speaker.
            //
            // After the zeros are gone, delay_buf holds a constant
            // (delay_len) samples at all times. Samples enter the back
            // and leave the front at the same rate, so the buffer size
            // never grows or shrinks in steady state.
            //
            // In C terms this is a circular buffer / queue:
            //   enqueue(delay_buf, output)   -- push onto back
            //   dequeue(delay_buf, n) -> ready -- pull from front
            //
            // If DELAY_MS == 0.0 then delay_len == 0 and delay_buf
            // is always empty — the if branch skips and output goes
            // straight to output_buf, preserving the original behaviour.

            if self.delay_len == 0 {
                // No delay configured — pass straight through
                self.output_buf.extend_from_slice(&output);
            } else {
                // Step A: push new clean samples onto back of FIFO
                self.delay_buf.extend_from_slice(&output);

                // Step B: pull the same number of samples from the
                // front of the FIFO into output_buf.
                //
                // drain(..n) removes the first n elements from the
                // Vec and returns an iterator over them.
                // collect() materialises the iterator into a Vec<f32>
                // so we can borrow it as a slice for extend_from_slice.
                //
                // In C terms:
                //   memcpy(ready, delay_buf.ptr, n * 4);
                //   memmove(delay_buf.ptr,
                //           delay_buf.ptr + n,
                //           (delay_buf.len - n) * 4);
                //   delay_buf.len -= n;
                let n = output.len();
                let ready: Vec<f32> = self.delay_buf
                    .drain(..n)
                    .collect();
                self.output_buf.extend_from_slice(&ready);
            }

            // ── 10. Advance input buffer by one hop ──────────
            self.input_buf.drain(0..HOP_SIZE);
        }
    }

    // Called by the output callback. Fills the speaker buffer.
    fn pull_samples(&mut self, out: &mut [f32]) {
        let available = out.len().min(self.output_buf.len());
        // Copy available processed samples to speaker
        out[..available]
            .copy_from_slice(&self.output_buf[..available]);
        // Remove consumed samples
        self.output_buf.drain(0..available);
        // Fill any remaining space with silence
        for s in out[available..].iter_mut() {
            *s = 0.0;
        }
    }
}

// ── Main function ─────────────────────────────────────────────
fn main() -> Result<()> {
    println!("=== Classical Noise Canceller ===");
    println!("Starting up...\n");

    // ── Audio device setup ───────────────────────────────────
    let host = cpal::default_host();

    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow!(
            "No microphone found. \
             Is one plugged in and enabled?"))?;

    let output_device = host
        .default_output_device()
        .ok_or_else(|| anyhow!(
            "No speaker/headphone found."))?;

    println!("Microphone : {}", input_device.name()?);
    println!("Speaker    : {}", output_device.name()?);

    // Print delay info so the user knows what to expect
    if DELAY_MS > 0.0 {
        let delay_samples =
            ((DELAY_MS / 1000.0) * SAMPLE_RATE as f32) as usize;
        println!("Delay      : {:.0}ms ({} samples)",
            DELAY_MS, delay_samples);
    } else {
        println!("Delay      : none (pipeline latency only ~21ms)");
    }

    // ── Stream configuration ─────────────────────────────────
    let config = StreamConfig {
        channels:    CHANNELS,
        sample_rate: SampleRate(SAMPLE_RATE),
        buffer_size: BufferSize::Default,
    };

    // ── Shared processor (accessed by both callbacks) ─────────
    // Arc = shared ownership, Mutex = safe concurrent access
    let processor = Arc::new(Mutex::new(Processor::new()));

    let proc_for_input  = Arc::clone(&processor);
    let proc_for_output = Arc::clone(&processor);

    // ── Input stream: mic → processor ────────────────────────
    let input_stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            let mut p = proc_for_input.lock().unwrap();
            p.push_samples(data);
        },
        |err| eprintln!("Input error: {}", err),
        None,
    )?;

    // ── Output stream: processor → speaker ───────────────────
    let output_stream = output_device.build_output_stream(
        &config,
        move |data: &mut [f32], _| {
            let mut p = proc_for_output.lock().unwrap();
            p.pull_samples(data);
        },
        |err| eprintln!("Output error: {}", err),
        None,
    )?;

    // ── Start both streams ───────────────────────────────────
    input_stream.play()?;
    output_stream.play()?;

    println!("\nNoise canceller is running.");
    println!("Speak into your microphone.");
    println!("Background noise will be reduced after ~2 seconds");
    println!("(time needed to learn the noise profile).");
    if DELAY_MS > 0.0 {
        println!(
            "Your voice will be heard {:.0}ms after you speak.",
            DELAY_MS
        );
    }
    println!("\nPress Ctrl+C to stop.\n");

    // ── Keep main thread alive ───────────────────────────────
    // If main() returned, the streams would be dropped (stopped).
    // We sleep forever so the audio threads keep running.
    loop {
        std::thread::sleep(
            std::time::Duration::from_secs(1)
        );
    }
}
