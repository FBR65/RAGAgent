#!/usr/bin/env python3
"""
Hauptstartscript für das RAGAgent System
Startet sowohl API als auch Frontend
"""

import sys
import os
import time
import threading
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ragagent.utils.logging import get_logger

logger = get_logger("ragagent_launcher")


def start_api():
    """Startet den API Server in einem separaten Prozess"""
    try:
        logger.info("🚀 Starte API Server...")
        api_script = project_root / "start_api.py"
        subprocess.run([sys.executable, str(api_script)], check=True)
    except Exception as e:
        logger.error(f"❌ Fehler beim Starten der API: {e}")


def start_frontend():
    """Startet das Frontend in einem separaten Prozess"""
    try:
        logger.info("🎨 Starte Frontend...")
        # Warte 3 Sekunden, damit die API Zeit hat zu starten
        time.sleep(3)
        frontend_script = project_root / "start_frontend.py"
        subprocess.run([sys.executable, str(frontend_script)], check=True)
    except Exception as e:
        logger.error(f"❌ Fehler beim Starten des Frontends: {e}")


def main():
    """Hauptfunktion zum Starten beider Services"""

    logger.info("=" * 60)
    logger.info("🤖 RAGAgent System Launcher")
    logger.info("=" * 60)
    logger.info("📦 Starte API und Frontend Services...")

    try:
        # Threads für parallele Ausführung
        api_thread = threading.Thread(target=start_api, daemon=True)
        frontend_thread = threading.Thread(target=start_frontend, daemon=True)

        # API zuerst starten
        api_thread.start()

        # Frontend nach kurzer Pause starten
        frontend_thread.start()

        logger.info("✅ Services gestartet!")
        logger.info("📍 API: http://localhost:8000")
        logger.info("🔗 API Docs: http://localhost:8000/docs")
        logger.info("🎨 Frontend: http://localhost:7860")
        logger.info("⏹️  Drücke Ctrl+C zum Beenden")

        # Warten auf beide Threads
        api_thread.join()
        frontend_thread.join()

    except KeyboardInterrupt:
        logger.info("🛑 System wird heruntergefahren...")
    except Exception as e:
        logger.error(f"❌ Systemfehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
