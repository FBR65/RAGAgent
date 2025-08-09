@echo off
REM RAGAgent System Launcher - Windows Batch Version
REM Startet API und Frontend Services

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo    RAGAgent System Launcher
echo ============================================================
echo.

REM Prüfe ob uv verfügbar ist
uv --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] UV Package Manager nicht gefunden!
    echo         Bitte installieren Sie UV: https://github.com/astral-sh/uv
    pause
    exit /b 1
)

REM Prüfe Projektverzeichnis
if not exist "ragagent" (
    echo [ERROR] Nicht im RAGAgent Projektverzeichnis!
    echo         Wechseln Sie ins RAGAgent Verzeichnis und starten Sie erneut.
    pause
    exit /b 1
)

echo [OK] UV Package Manager gefunden
echo [OK] RAGAgent Projektverzeichnis erkannt
echo.

REM Menü anzeigen
echo Wählen Sie einen Service:
echo.
echo [1] Nur API starten (Port 8000)
echo [2] Nur Frontend starten (Port 7860)  
echo [3] Beide Services starten (Empfohlen)
echo [4] Beenden
echo.

:menu
set /p choice="Ihre Wahl (1-4): "

if "%choice%"=="1" goto start_api
if "%choice%"=="2" goto start_frontend  
if "%choice%"=="3" goto start_both
if "%choice%"=="4" goto end
goto invalid_choice

:start_api
echo.
echo [API] Starte API Server...
start "RAGAgent API" cmd /k "uv run python start_api.py"
echo [OK] API Server gestartet auf http://localhost:8000
echo [DOC] API Dokumentation: http://localhost:8000/docs
goto end

:start_frontend
echo.
echo [UI] Starte Frontend...
start "RAGAgent Frontend" cmd /k "uv run python start_frontend.py"
echo [OK] Frontend gestartet auf http://localhost:7860
goto end

:start_both
echo.
echo [API] Starte API Server...
start "RAGAgent API" cmd /k "uv run python start_api.py"
echo [WAIT] Warte 3 Sekunden...
timeout /t 3 /nobreak >nul
echo [UI] Starte Frontend...
start "RAGAgent Frontend" cmd /k "uv run python start_frontend.py"
echo.
echo [OK] Beide Services gestartet!
echo.
echo Verfügbare Services:
echo   API Server: http://localhost:8000
echo   Frontend: http://localhost:7860
echo   API Docs: http://localhost:8000/docs
echo.
echo [INFO] Schließen Sie die CMD-Fenster zum Beenden
goto end

:invalid_choice
echo.
echo [ERROR] Ungültige Auswahl! Bitte wählen Sie 1-4.
echo.
goto menu

:end
echo.
echo [SUCCESS] RAGAgent Launcher beendet.
pause
