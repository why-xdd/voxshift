@echo off
echo ============================================
echo    VoxShift setup
echo ============================================
echo.
echo [1/2] Installing Python dependencies...
pip install -r requirements.txt
echo.
echo [2/2] Setting up the virtual microphone driver (VB-Cable)...
echo       Accept the UAC prompt, click "Install Driver", then reboot.
python -c "import driver; driver.install_cable()"
echo.
echo Setup finished. Launch VoxShift with run.bat
pause
