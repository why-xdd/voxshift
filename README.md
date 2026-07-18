<div align="center">

# 🎙️ VoxShift

### Real-time voice changer for Windows — in Python

Turn your microphone into a **woman, kid, robot, demon, alien, Minion** or your own
custom voice — live, with ~35 ms latency. A full studio effect chain, **46 character
presets** in categories, global hotkeys, recording, and one-click virtual-mic setup.

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
| 🎭 **46 character presets** | Human · Cartoon · Sci-Fi · Horror · Famous · FX — browse by category, one click to switch |
| 🎚️ **Pitch −24…+24 st** | STFT phase vocoder, 8× overlap + rigid phase-locking for a smoother, less "robotic" sound |
| 🧑‍🤝‍🧑 **Independent formant shift** | Change timbre / "gender" separately from pitch, tuned to real vocal-tract acoustics |
| 🎛️ **Full effect chain** | 3-band EQ · distortion · bitcrusher · ring mod · tremolo · vibrato · echo · reverb · noise gate |
| 🔌 **One-click virtual mic** | Installs the VB-Cable driver for you so the voice works as a mic in Discord/games out of the box |
| ⌨️ **Global hotkeys** | Switch presets & mute without leaving your game or call |
| 🎧 **Hear yourself** | A monitor toggle plays the result to your headphones so you can test voices — even while the main output feeds Discord |
| 🔴 **Record to WAV** | Capture the changed voice, with or without a cable |
| 🧠 **AI path (planned)** | Scaffold in place for neural voice conversion (RVC) — the truly natural / undetectable route |

---

## 🚀 Installation

### 1. Install Python 3.10+
Download from [python.org](https://www.python.org/downloads/) and **tick "Add Python to PATH"**.

### 2. Get VoxShift
```powershell
git clone https://github.com/why-xdd/voxshift.git
cd voxshift
```
(or download the ZIP from GitHub and extract it)

### 3. One-click setup
Double-click **`setup.bat`** — it installs the dependencies **and** sets up the virtual
microphone driver (VB-Cable). Accept the UAC prompt, click *Install Driver*, then reboot.

<sub>Prefer manual? `pip install -r requirements.txt`, then optionally set up the mic driver from inside the app.</sub>

### 4. Run
Double-click **`run.bat`** (or `python voxshift.py`). Pick your mic, choose an output, hit **Start**. 🎉

---

## 🎧 Using it as a microphone in Discord / games

To make *any* program's audio appear as a **microphone**, Windows needs a **virtual audio cable** (a kernel-level loopback driver). VoxShift can install one for you:

- **Automatic:** run `setup.bat`, **or** click **🎚 Set up virtual mic (VB-Cable)** inside the app. It downloads and launches the official installer — accept UAC, click *Install Driver*, reboot.
- Then in VoxShift set **output = `CABLE Input`**, and in Discord/the game set the **microphone = `CABLE Output`**.

Without a cable you can still **monitor** through your headphones and **record** to a `.wav`.

### 🎧 Hearing yourself (testing voices)

Turn on **🎧 Hear myself (monitor)** and set its device to the headphones/speakers you actually
listen on. VoxShift then plays the processed voice to that device — so you can test presets live,
even when the main **Output** is routed to a virtual cable for Discord. (It skips monitoring if
the monitor device is the same as the main output, to avoid doubled sound.)

> ### 😱 "Installing the cable killed my microphone!"
> Common, and **not a fault**. The installer sets `CABLE Output` as your *default* recording device, so apps that use the default mic hear silence. **Fix:** *Settings → System → Sound* → set your **real microphone** back as the default input, and only pick `CABLE Output` *inside Discord*. Your mic keeps working everywhere else.

VoxShift shows a green/orange line under the output picker telling you whether a cable was detected.

---

## 🎭 Presets

Browse them by category in the app — 46 in total:

| Category | Voices |
|---|---|
| 👤 **Human** | Man, Woman, Kid, Baby, Deep, Old Man, Announcer |
| 🐭 **Cartoon** | Chipmunk, **Minion**, Mouse, Helium, Duck, Parrot, Blue Elf, Sponge, Troll |
| 🤖 **Sci-Fi** | Robot, Cyborg, AI Core, Alien, Dalek, Drone, Glitch, Comms |
| 👻 **Horror** | Demon, Ghost, Giant, Goblin, Vampire, Zombie, Monster, Skeleton |
| 🎬 **Famous** | Dark Lord, Green Ogre, Wizard, Pirate, Clown, Santa, Cowboy |
| 🎚️ **FX** | Telephone, Radio, Cave, Hall, Underwater, Megaphone |
| ⭐ **Custom** | Anything you dial in and **💾 Save** |

---

## 🎯 Tuning a realistic voice

The ideal settings depend on **your own** voice, so tweak on the **Voice** tab:

- **👩 Woman** — if it sounds like "a man on helium", use **more Pitch, less Formant**. Typical: pitch **+5…+9**, formant **+2…+4**.
- **🧒 Kid / 👶 Baby** — kids are mostly about **higher formants** (shorter vocal tract): formant **+5…+9**, pitch **+9…+13**.
- **🐭 High/squeaky** — pitch **+14…+22**, formant **+6…+10**.

Keep **Preserve formants** on for natural results; off gives the classic chipmunk effect.

> **Why does DSP still sound a bit processed?** Classic pitch/formant DSP always leaves some signature at large shifts. The truly *natural, hard-to-detect* route is **AI voice conversion** — planned as an opt-in engine (see roadmap); the integration point already exists in `ai_engine.py`.

---

## ⌨️ Hotkeys (while running)

| Shortcut | Action |
|---|---|
| `Ctrl+Alt+1…9` | Switch to preset 1–9 |
| `Ctrl+Alt+0` | Toggle mute |

Global hotkeys use the optional `keyboard` package; if missing, the app still runs.

---

## 🧠 How it works

```
mic → noise gate → [AI convert*] → pitch/formant → EQ → distortion → bitcrusher
    → ring mod → tremolo → vibrato → echo → reverb → volume → out          (*planned)
```

Each stage is a block-based, NumPy/SciPy-vectorized effect (≈12 % of the real-time CPU budget with everything on). The pitch/formant core is a phase vocoder that separates the formant envelope (cepstral liftering) from the excitation, shifts only the excitation, re-applies the envelope — optionally warped by the formant slider — and uses **rigid peak phase-locking** to keep harmonics coherent and kill most of the phasey/robotic artifact.

---

## 📁 Project layout

| File | What it is |
|---|---|
| `voxshift.py` | Audio engine, presets, recording + tabbed GUI |
| `dsp.py` | All DSP effect blocks (pitch/formant, EQ, distortion, bitcrusher, ring/tremolo/vibrato, echo, reverb) |
| `driver.py` | One-click VB-Cable virtual-mic installer |
| `ai_engine.py` | Interface + stub for the future AI voice-conversion engine |
| `setup.bat` / `run.bat` | One-click setup / launch |
| `requirements.txt` · `requirements-ai.txt` | Core deps · optional AI stack |

Recordings → `~/VoxShift Recordings/` · custom presets → `~/.voxshift_presets.json`.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| **No sound / error on Start** | Input and output must be different devices; close apps holding the mic in exclusive mode. |
| **Still sounds a bit synthetic** | Keep **Preserve formants** on; extreme shifts (±16+) are inherently thinner. AI engine (roadmap) is the natural route. |
| **Hotkeys don't work** | `pip install keyboard`, restart the app. |
| **Discord can't hear me** | VoxShift must be **running** with output = `CABLE Input`, and Discord's mic = `CABLE Output`. |
| **Driver install did nothing** | Accept the UAC prompt and click *Install Driver* in the VB-Cable window, then reboot. |

---

## 🗺️ Roadmap

- [x] 46 character presets in categories, custom presets, hotkeys, recording
- [x] Independent formant shift, rigid phase-locking, full effect chain
- [x] One-click virtual-mic (VB-Cable) installer
- [ ] **AI voice conversion (RVC)** — natural, hard-to-detect voices on your GPU
- [ ] Trained character models (per-voice) for the AI engine
- [ ] Per-preset hotkey assignment in the UI

---

## 📜 License

[MIT](LICENSE). VB-Cable is a separate product by VB-Audio, installed from its official source, under its own licence.
