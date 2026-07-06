<div align="center">

# 🎙️ VoxShift

### Real-time voice changer for Windows — in Python

Turn your microphone into a **woman, kid, robot, demon, alien** or your own custom voice,
live, with ~35 ms latency. A full studio effect chain, 17 presets, global hotkeys and
recording — in two small files.

![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![DSP](https://img.shields.io/badge/DSP-NumPy%20%2F%20SciPy-013243?logo=scipy&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-CustomTkinter-2b6cb0)
![Platform](https://img.shields.io/badge/platform-Windows-0078D6?logo=windows&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## ✨ Features

| | |
|---|---|
| 🎚️ **Pitch shift −24…+24 st** | STFT phase vocoder with 8× overlap + rigid phase-locking — smooth, no chipmunk-tempo artifacts |
| 🧑‍🤝‍🧑 **Independent formant shift** | Change timbre / "gender" separately from pitch, tuned to real vocal-tract acoustics |
| 🎛️ **Full effect chain** | 3-band EQ · distortion · bitcrusher · ring mod · tremolo · vibrato · echo · reverb |
| 🔇 **Noise gate** | Cuts background hiss so it isn't amplified into the changed voice |
| 🎭 **17 presets** | Man, Woman, Kid, Baby, Squeaky, Helium, Robot, Cyborg, Demon, Alien, Ghost, Telephone, Radio, Cave… + save your own |
| ⌨️ **Global hotkeys** | Switch presets & mute without leaving your game or call |
| 🔴 **Record to WAV** | Capture the changed voice even without a virtual cable |
| ⚡ **Light** | The whole chain runs at ~10 % of the real-time CPU budget |

---

## 📸 The interface

```
┌────────────────────────────────────────────┐
│                 VoxShift                     │
│  Microphone  [ your mic            ▼]        │
│  Output      [ headphones / cable  ▼]        │
│  Preset  [👩 Woman ▼]   [name…] 💾 Save 🗑    │
│ ┌ Voice ─ Tone ─ Modulation ─ Space ───────┐ │
│ │ Pitch    ──●────────────  +7.0 st        │ │
│ │ Formant  ────●──────────  +3.0 st        │ │
│ │ ☑ Preserve formants                      │ │
│ │ Noise gate ●───────────   0.000          │ │
│ │ Volume   ──────●────────  100%           │ │
│ └──────────────────────────────────────────┘ │
│  In  ▮▮▮▮▮▮▯▯▯▯▯▯▯▯▯                          │
│  Out ▮▮▮▮▮▮▮▮▯▯▯▯▯▯▯                          │
│  [ ▶  Start ]                    [ ● Rec ]    │
└────────────────────────────────────────────┘
```

---

## 🚀 Installation

### 1. Install Python 3.10+

Download from [python.org](https://www.python.org/downloads/) and **tick "Add Python to PATH"** during setup.
Check it in a terminal (PowerShell):

```powershell
python --version
```

### 2. Get VoxShift

```powershell
git clone https://github.com/why-xdd/voxshift.git
cd voxshift
```

(or download the ZIP from GitHub and extract it)

### 3. Install the dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run it

```powershell
python voxshift.py
```

or just double-click **`run.bat`** (it installs the requirements and launches the app).

That's it — pick your microphone, choose your headphones as the output, hit **Start** and talk. 🎉

---

## 🎧 Using it as a microphone in Discord / games

> **The honest truth:** to make *any* program's audio show up as a **microphone** in Discord or a game, Windows needs a **virtual audio cable** (a kernel-level loopback driver). This isn't a VoxShift limitation — every "real" voice changer (Voicemod, Clownfish, MorphVOX…) installs its own such driver under the hood.

**Without a cable** you can still:
- 🎧 **Monitor** the changed voice through your headphones (pick them as the output)
- 🔴 **Record** it to a `.wav` with the `● Rec` button

**To route it into Discord / games:**

1. Install the free [**VB-Cable**](https://vb-audio.com/Cable/)
2. In VoxShift: **input** = your real mic, **output** = `CABLE Input`
3. In Discord (or the game): **microphone** = `CABLE Output`

> ### 😱 "Installing VB-Cable killed my microphone!"
> Common, and **not a real fault**. The installer sets `CABLE Output` as your *default* recording device, so apps that use the default mic now hear silence (nothing feeds the cable unless VoxShift is running). **Fix:** open **Settings → System → Sound**, set your **real microphone** back as the default input, and only pick `CABLE Output` *inside Discord*. Your mic then keeps working everywhere else.

VoxShift shows a green/orange line under the output picker telling you whether a virtual cable was detected.

---

## 🎭 Presets

| Preset | Pitch | Formant | Character |
|---|:---:|:---:|---|
| 🎤 Normal | 0 | 0 | passthrough |
| 👨 Man | −4 | −3 | deeper, chestier |
| 👩 Woman | +7 | +3 | female F0 + female vocal tract (1.2×) |
| 🧒 Kid | +10 | +5 | child tract (1.33×) |
| 👶 Baby | +12 | +8 | tiny tract (1.6×) |
| 🐭 Squeaky | +16 | +8 | very high & thin |
| 🎈 Helium | +4 | +12 | physically-accurate helium |
| 🎩 Deep | −5 | −2 | rich low voice |
| 🐿️ Chipmunk | +9 | — | classic (formants move too) |
| 🤖 Robot | 0 | 0 | ring mod + bitcrush |
| 🦾 Cyborg | −2 | 0 | ring mod + distortion |
| 😈 Demon | −7 | −4 | low + ring + distortion + reverb |
| 👽 Alien | +3 | +5 | ring mod + vibrato |
| 👻 Ghost | −3 | 0 | big reverb + echo + vibrato |
| ☎️ Telephone | 0 | 0 | band-limited + bitcrush |
| 📻 Radio | 0 | 0 | mid-forward + distortion |
| 🕳️ Cave | −2 | 0 | huge reverb + echo |

Dial in your own on the tabs and hit **💾 Save** to add it to the list.

---

## 🎯 Tuning a realistic voice

The ideal settings depend on **your own** voice, so treat presets as starting points and fine-tune on the **Voice** tab:

- **👩 Woman** — if it sounds like "a man on helium", use **more Pitch, less Formant**. If it's cartoonish, drop the Formant. Typical: pitch **+5…+9**, formant **+2…+4**.
- **🧒 Kid / 👶 Baby** — kids are mostly about **higher formants** (shorter vocal tract), not just pitch. Formant **+5…+9**, pitch **+9…+13**.
- **🐭 Squeaky** — crank pitch to **+14…+22**, add formant **+6…+10** to keep it thin.
- **🎈 Helium** — high formant (**+10…+12**) with only a little pitch (**+3…+5**) is the real helium physics.

Keep **Preserve formants** on for natural results; turn it off for the classic chipmunk/robot effect.

---

## ⌨️ Hotkeys (while running)

| Shortcut | Action |
|---|---|
| `Ctrl+Alt+1…9` | Switch to preset 1–9 (in list order) |
| `Ctrl+Alt+0` | Toggle mute |

Global hotkeys use the optional `keyboard` package (installed by `requirements.txt`); if it's missing the app still runs and just disables them.

---

## 🧠 How it works

The signal flows through a fixed chain, each stage a small block-based, NumPy/SciPy-vectorized effect:

```
mic → noise gate → pitch/formant → EQ → distortion → bitcrusher
    → ring mod → tremolo → vibrato → echo → reverb → volume → out
```

The core **pitch/formant shifter** is a phase vocoder:

1. The signal is windowed into overlapping 1024-sample frames (87.5 % overlap) and FFT'd
2. The **true** frequency of each bin is recovered from the phase difference between frames
3. The **spectral envelope** (formants) is separated from the excitation by cepstral liftering
4. Only the excitation is pitch-scaled; the envelope is re-applied — optionally **warped** by the formant slider — so pitch and timbre move independently
5. Phases are re-accumulated with **rigid peak phase-locking** (bins of one harmonic stay coherent, which kills the "phasey / robotic" artifact), inverse-FFT'd and overlap-added back into a stream

---

## 📁 Project layout

| File | What it is |
|---|---|
| `voxshift.py` | Audio engine (effect chain, presets, recording) + tabbed GUI |
| `dsp.py` | All the DSP effect blocks |
| `requirements.txt` | Dependencies |
| `run.bat` | One-click install + launch |

Recordings are saved to `~/VoxShift Recordings/`, custom presets to `~/.voxshift_presets.json`.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| **No sound / error on Start** | Make sure the input and output devices aren't the same, and that nothing else has the mic locked in exclusive mode. |
| **Robotic / metallic on extreme settings** | Keep **Preserve formants** on; very large pitch shifts (±16+) are inherently thinner. |
| **Latency too high** | It's ~35 ms by design; close other heavy audio apps if you hear more. |
| **Hotkeys don't work** | `pip install keyboard`, then restart the app. |
| **Discord doesn't hear me** | VoxShift must be **running with output = CABLE Input**, and Discord's mic set to **CABLE Output**. |

---

## 🗺️ Roadmap

- [x] Independent formant shift, rigid phase-locking
- [x] Full effect chain (EQ, distortion, bitcrusher, ring/tremolo/vibrato, echo, reverb)
- [x] Record to WAV, custom presets, global hotkeys
- [ ] Autotune / hard pitch-quantize
- [ ] Per-preset hotkey assignment in the UI
- [ ] Bundled virtual-cable setup helper

---

## 📜 License

[MIT](LICENSE) — do whatever you like, no warranty.
