@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ---- Proxy fallback ----
rem Some networks (e.g. mobile hotspot) cannot reach github.com directly.
rem If your proxy uses a different port, change PROXY_PORT below.
set "PROXY_PORT=65532"
curl -s -o nul --max-time 6 https://github.com >nul 2>&1
if errorlevel 1 (
    set "HTTPS_PROXY=http://127.0.0.1:%PROXY_PORT%"
    set "HTTP_PROXY=http://127.0.0.1:%PROXY_PORT%"
    echo [INFO] Direct GitHub unreachable. Using proxy 127.0.0.1:%PROXY_PORT%
    echo.
)

echo ============================================
echo   Upload study records to GitHub
echo ============================================
echo.

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not a git repository in this folder.
    echo         Run "git init" first.
    pause
    exit /b 1
)

git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No remote "origin" configured.
    echo         Run this once:
    echo         git remote add origin https://github.com/YOUR_NAME/YOUR_REPO.git
    pause
    exit /b 1
)

git add -A

git diff --cached --quiet
if not errorlevel 1 (
    echo No changes to commit. Nothing to do.
    pause
    exit /b 0
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm"') do set STAMP=%%i

echo.
echo Changes to be committed:
git diff --cached --stat
echo.

git commit -m "study: %STAMP%"

if errorlevel 1 (
    echo [ERROR] Commit failed.
    pause
    exit /b 1
)

git push

if errorlevel 1 (
    echo.
    echo [ERROR] Push failed. Check your network or GitHub login.
    pause
    exit /b 1
)

echo.
echo Done. Uploaded successfully.
pause