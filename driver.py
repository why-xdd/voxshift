"""Virtual audio cable setup for VoxShift.

To make the changed voice show up as a *microphone* in Discord / games,
Windows needs a virtual audio device (a loopback driver). This module
installs the free official **VB-CABLE** (VB-Audio) — it downloads the
package (or reuses one already in your Downloads folder), then launches
the vendor's installer elevated so you just accept the UAC prompt and
click *Install Driver*.

We deliberately do not bundle or redistribute the driver — VB-CABLE's
licence doesn't allow that — we fetch it from the official server.
"""

import ctypes
import glob
import os
import tempfile
import urllib.request
import zipfile

VBCABLE_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"


def is_cable_installed():
    """True if a VB-CABLE (or compatible) virtual device is present."""
    try:
        import sounddevice as sd
        return any("CABLE" in d["name"] for d in sd.query_devices())
    except Exception:
        return False


def _find_local_zip():
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    hits = sorted(glob.glob(os.path.join(downloads, "VBCABLE_Driver_Pack*.zip")))
    return hits[-1] if hits else None


def install_cable(log=print):
    """Download (or reuse) and launch the VB-CABLE installer, elevated.

    Returns "already", or "launched" once the vendor installer is running.
    Raises on failure. The vendor installer still needs one click on
    *Install Driver* and a reboot — that's how VB-CABLE ships.
    """
    if is_cable_installed():
        log("✓ Virtual cable already installed.")
        return "already"

    tmp = tempfile.mkdtemp(prefix="voxshift_vbcable_")
    zip_path = _find_local_zip()
    if zip_path:
        log(f"Using package from Downloads: {os.path.basename(zip_path)}")
    else:
        zip_path = os.path.join(tmp, "vbcable.zip")
        log("Downloading VB-CABLE from vb-audio.com …")
        urllib.request.urlretrieve(VBCABLE_URL, zip_path)

    log("Extracting …")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)

    exe = None
    for name in ("VBCABLE_Setup_x64.exe", "VBCABLE_Setup.exe"):
        cand = os.path.join(tmp, name)
        if os.path.exists(cand):
            exe = cand
            break
    if exe is None:
        found = glob.glob(os.path.join(tmp, "**", "VBCABLE_Setup*.exe"), recursive=True)
        exe = found[0] if found else None
    if exe is None:
        raise FileNotFoundError("VBCABLE_Setup executable not found in the package.")

    log("Launching installer — accept the UAC prompt, click 'Install Driver', then reboot.")
    # runas triggers the UAC elevation the driver install requires
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", exe, None, os.path.dirname(exe), 1)
    if rc <= 32:
        raise RuntimeError(
            f"Could not launch the installer (code {rc}). "
            f"Open {exe} manually and choose 'Run as administrator'.")
    return "launched"


if __name__ == "__main__":
    try:
        result = install_cable()
        if result == "launched":
            print("\nFinish the VB-CABLE installer window, then reboot.")
            print("After that, in VoxShift set output = 'CABLE Input' and in Discord")
            print("set the microphone = 'CABLE Output'.")
    except Exception as exc:
        print("Install failed:", exc)
