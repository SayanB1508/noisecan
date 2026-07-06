# 🎙️ Classical Noise Canceller in Rust

A real-time background noise canceller built from scratch in Rust — no AI, no magic black boxes. Just pure **Digital Signal Processing**: FFT, spectral subtraction, voice activity detection, and overlap-add reconstruction.

This is **Stage 1** of a three-stage project:
- ✅ **Stage 1 (this repo):** Classical DSP noise canceller on Fedora Linux
- 🔜 **Stage 2:** Port to Raspberry Pi via cross-compilation
- 🔜 **Stage 3:** Replace the classical core with [nnnoiseless](https://github.com/nicowillis/nnnoiseless) (RNNoise neural network)

---

## 🔊 What It Does

Captures audio from your microphone, removes background noise (fan hum, AC, ambient noise) in real time, and plays the cleaned audio through your speaker — all with under 40ms of pipeline latency.

```
Mic → FFT → Noise Estimate → Spectral Subtraction → IFFT → Speaker
```

The system **learns your room's noise** over the first ~2 seconds of silence, then continuously subtracts it from incoming audio — leaving only your voice.

---

## 🧠 How It Works (High Level)

| Step | What happens |
|------|-------------|
| 1. Capture | `cpal` reads 512-sample chunks from the microphone at 48,000 Hz |
| 2. Window | Each 1024-sample frame is multiplied by a **Hann window** to prevent spectral leakage |
| 3. FFT | The windowed frame is transformed to the frequency domain (1024 complex bins) |
| 4. VAD | **Voice Activity Detection** — is this frame silence or speech? |
| 5. Noise Estimate | During silence: update a per-bin **Exponential Moving Average** of the noise spectrum |
| 6. Subtract | For each of the 513 frequency bins: `clean = max(noisy − α×noise,  β×noise)` |
| 7. IFFT | Transform back to time domain |
| 8. Overlap-Add | Accumulate overlapping IFFT frames with `+=` for seamless reconstruction |
| 9. Output | `cpal` plays the cleaned audio through the speaker |

---

## 📁 Project Structure

```
noise_canceller/
├── Cargo.toml
└── src/
    ├── main.rs                  # Entry point — wires everything together
    ├── window.rs                # Hann window generation
    ├── stft.rs                  # Short-Time Fourier Transform (FFT + IFFT + overlap-add)
    ├── vad.rs                   # Voice Activity Detector (energy-based)
    ├── noise_estimator.rs       # Exponential moving average noise profile (513 bins)
    └── spectral_subtraction.rs  # Spectral subtraction with phase preservation
```

---

## ⚙️ Configuration

All tunable parameters are constants at the top of `src/main.rs`:

| Constant | Default | Description |
|----------|---------|-------------|
| `SAMPLE_RATE` | `48_000` | Audio sample rate in Hz |
| `FFT_SIZE` | `1024` | Frame size — 21.33ms per frame |
| `HOP_SIZE` | `512` | Frame advance — 50% overlap (required for COLA) |
| `VAD_THRESHOLD` | `0.01` | Energy threshold between silence and speech |
| `ALPHA` | `1.5` | Subtraction strength (higher = more aggressive) |
| `BETA` | `0.002` | Spectral floor (prevents musical noise) |
| `NOISE_SMOOTH` | `0.95` | EMA smoothing coefficient (higher = slower adaptation) |
| `DELAY_MS` | `500.0` | Extra pipeline delay in ms (set to `0.0` to disable) |

**Tuning tip:** If noise is not being reduced, `VAD_THRESHOLD` may be too low for your environment. Temporarily add this line inside the `while` loop in `push_samples()` to observe actual energy levels:
```rust
self.vad.print_energy(&self.input_buf[..FFT_SIZE]);
```
Then set `VAD_THRESHOLD` between the silence and speech energy values you observe.

---

## 🚀 Getting Started

### Prerequisites

```bash
# Fedora / RHEL
sudo dnf install alsa-lib-devel gcc make

# Ubuntu / Debian
sudo apt install libasound2-dev gcc make
```

Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Build and Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/noise_canceller.git
cd noise_canceller

# Run tests
cargo test

# Run in release mode (important — debug builds may glitch)
cargo run --release
```

### Expected Output

```
=== Classical Noise Canceller ===
Starting up...

Microphone : Built-in Audio Analog Stereo
Speaker    : Built-in Audio Analog Stereo
Delay      : 500ms (24000 samples)

Noise canceller is running.
Speak into your microphone.
Background noise will be reduced after ~2 seconds
(time needed to learn the noise profile).
Your voice will be heard 500ms after you speak.

Press Ctrl+C to stop.
```

---

## 🧪 Running Tests

```bash
cargo test
```

Each module has its own unit tests:
- `window.rs` — verifies Hann window edges are 0 and middle is 1
- `vad.rs` — verifies silence and speech classification
- `noise_estimator.rs` — verifies EMA updates during silence and freezes during speech

---

## 📐 DSP Concepts Used

- **Nyquist Theorem** — 48 kHz sample rate → 24 kHz Nyquist frequency → 513 unique FFT bins
- **Hann Window** — eliminates spectral leakage at frame boundaries
- **COLA (Constant Overlap-Add)** — `w[n] + w[n + N/2] = 1.0` guarantees perfect reconstruction at 50% overlap
- **Hermitian Symmetry** — `X[N-k] = conj(X[k])` for real-valued signals — must be restored before IFFT
- **Exponential Moving Average** — per-bin noise tracking: `estimate[i] = 0.95 × estimate[i] + 0.05 × magnitude[i]`
- **Spectral Subtraction** — `clean_mag = max(noisy − α×noise, β×noise)` with phase preservation

---

## 🦀 Rust Concepts Used

- **Ownership and Move Semantics** — spectrum moves through the pipeline with zero allocation
- **Borrowing (`&T`, `&mut T`)** — magnitude is borrowed by both VAD and subtractor
- **`Arc<Mutex<T>>`** — shared Processor state between the cpal input and output threads
- **Iterators** — `iter_mut().zip()` for the per-bin EMA update loop
- **`Vec<T>` and slices** — ring buffer, overlap buffer, delay FIFO all as standard Rust collections

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| No microphone found | Run `pactl list short sources` to check PipeWire sees your mic |
| Audio glitches / dropouts | Always use `cargo run --release`. Debug builds are too slow. |
| No noise reduction | Use `print_energy()` to tune `VAD_THRESHOLD` for your environment |
| Feedback squeal | Use headphones, or lower speaker volume |
| Compile error about `asound` | Run `sudo dnf install alsa-lib-devel` |

---

## 🗺️ Roadmap

- [x] Stage 1 — Classical spectral subtraction on Fedora (Rust + cpal + rustfft)
- [ ] Stage 2 — Cross-compile for Raspberry Pi (`aarch64-unknown-linux-gnu`)
- [ ] Stage 3 — Replace `SpectralSubtractor` with [nnnoiseless](https://github.com/nicowillis/nnnoiseless) (RNNoise neural net)
- [ ] Wiener filter option (per-bin SNR-based gain instead of hard subtraction)
- [ ] OLED status display (noise level, VAD state, convergence progress)
- [ ] Real-time spectrogram in terminal using `ratatui`

---

## 📚 Technical Handbook

A complete technical handbook (90+ pages across two versions) covering:
- Full DSP theory from first principles
- Line-by-line Rust code explanations with C comparisons
- Architecture decisions and tradeoffs
- Tuning guide for different noise environments

Available in the [`/docs`](./docs) folder.

---

## 📦 Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| [`cpal`](https://crates.io/crates/cpal) | 0.15 | Cross-platform audio I/O (works on Linux, Windows, macOS) |
| [`rustfft`](https://crates.io/crates/rustfft) | 6 | High-performance FFT (Cooley-Tukey algorithm) |
| [`anyhow`](https://crates.io/crates/anyhow) | 1 | Ergonomic error handling |

---

## 🎓 Learning Context

This project was built during a summer internship project by a 1st year ECE student at IIIT. The goal was to learn Rust and real-time DSP simultaneously by building something that actually works — rather than toy examples.

If you want to understand every single line of code, the technical handbook in `/docs` has you covered.

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.
