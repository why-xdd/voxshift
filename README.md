# 🎙️ VoxShift

Real-time voice changer for Windows written in pure Python. Pitch-shifts your microphone with a phase vocoder and routes the result to any output device — pick **VB-Cable** as the output and the changed voice becomes a virtual microphone you can use in Discord, games or any voice chat.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![NumPy](https://img.shields.io/badge/DSP-NumPy-013243)
![GUI](https://img.shields.io/badge/GUI-CustomTkinter-2b6cb0)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Real-time processing** — ~35 ms of algorithmic latency (1024-sample phase-vocoder frame + 512-sample audio block at 44.1 kHz)
- **Pitch shifting from −12 to +12 semitones** — STFT phase vocoder, no chipmunk-tempo artifacts of naive resampling
- **5 presets**: Normal 🎤, Robot 🤖 (ring modulation), Chipmunk 🐿️, Deep 🎩, Demon 😈 (pitch-down + low-frequency ring mod)
- **Virtual microphone** — auto-detects VB-Cable and pre-selects it as output
- **VU meters** for input and output, volume control, device picker — dark-themed CustomTkinter GUI
- Single file, three dependencies

## Quick start

```bash
pip install -r requirements.txt
python voxshift.py
```

or just double-click `run.bat`.

### Using it as a microphone in Discord / games

1. Install the free [VB-Cable](https://vb-audio.com/Cable/) virtual audio device
2. In VoxShift select your real microphone as input and **CABLE Input** as output
3. In Discord (or any app) select **CABLE Output** as the microphone

## How it works

Naively resampling audio changes pitch *and* speed. VoxShift instead uses a **phase vocoder** (the smbPitchShift analysis/synthesis scheme, vectorized with NumPy):

1. The signal is windowed into overlapping 1024-sample frames (75% overlap) and transformed with an FFT
2. The *true* frequency of every bin is recovered from the phase difference between consecutive frames
3. Magnitudes and true frequencies are moved to their pitch-scaled positions
4. Phases are re-accumulated, the frame is inverse-transformed and overlap-added back into a stream

The Robot and Demon presets additionally multiply the signal with a sine carrier (ring modulation) — the classic Dalek effect.

## Roadmap

- [ ] Formant preservation (WORLD vocoder) so pitched-down voices sound less "giant"
- [ ] Noise gate
- [ ] Hotkeys for switching presets while in game
- [ ] Save custom presets

## License

MIT
