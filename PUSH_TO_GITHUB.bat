@echo off
title Push SPAZE to GitHub
echo =========================================================
echo   Pushing SPAZE to https://github.com/ombansal1224-max/SPAZE.git
echo =========================================================
echo.
cd /d "%~dp0"
git push -u origin main
echo.
echo =========================================================
echo Done! Check your repo at:
echo https://github.com/ombansal1224-max/SPAZE
echo =========================================================
pause
