@echo off
setlocal
cd /d "%~dp0"

rem === configuration ===
set "POOL=ws://fxl.baikalmine.com:2030"
set "WALLET=0xEe73Ed81501Fa503FC708265A43B07dCf86A8763"
set "MINER=target\release\vulkminer.exe"

rem Build the release binary if it is missing.
if not exist "%MINER%" (
    echo Release binary not found, building...
    cargo build --release || (echo Build failed. & pause & exit /b 1)
)

echo Starting FusionHash miner
echo   pool   : %POOL%
echo   wallet : %WALLET%
echo.

rem Any extra args passed to this script are forwarded (e.g. --intensity 1.5 -d 1).
"%MINER%" --pool "%POOL%" --user "%WALLET%" --pass x %*

echo.
echo Miner exited.
pause
