@echo off
rem Run this on the WINDOWS PC (not on the Pi): double-click it, or run it in a
rem PowerShell / cmd window on the PC. It opens an SSH tunnel so the Pi's web
rem stream (127.0.0.1:5000 on the Pi) shows at http://127.0.0.1:5050 on the PC.
rem Keep this window open while you watch; close it to stop the tunnel.
echo Tunnel to the Pi - open http://127.0.0.1:5050 in your browser.
echo Keep this window open. Close it (or press Ctrl+C) to stop.
ssh -N -L 5050:127.0.0.1:5000 skinderholm-pi5@192.168.1.187
pause
