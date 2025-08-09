#!/usr/bin/env python3
"""
Entwickler-Startscript für das RAGAgent System
Startet Services mit Hot-Reload und erweiterten Debug-Features
"""

import sys
import os
import time
import threading
import subprocess
import signal
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ragagent.utils.logging import get_logger

logger = get_logger("dev_launcher")


class CodeChangeHandler(FileSystemEventHandler):
    """Handler für Code-Änderungen"""

    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_restart = 0

    def on_modified(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith((".py", ".toml", ".yaml", ".yml")):
            current_time = time.time()
            if current_time - self.last_restart > 2:  # Debounce: 2 Sekunden
                self.last_restart = current_time
                logger.info(f"🔄 Code-Änderung erkannt: {event.src_path}")
                self.restart_callback()


class DevLauncher:
    """Entwicklungs-Launcher mit Hot-Reload"""

    def __init__(self):
        self.api_process = None
        self.frontend_process = None
        self.observer = None
        self.running = True

    def start_api(self):
        """Startet den API Server"""
        if self.api_process:
            self.stop_api()

        logger.info("🚀 Starte API Server (Dev Mode)...")
        self.api_process = subprocess.Popen(
            [sys.executable, "start_api.py"], cwd=project_root
        )

    def start_frontend(self):
        """Startet das Frontend"""
        if self.frontend_process:
            self.stop_frontend()

        logger.info("🎨 Starte Frontend (Dev Mode)...")
        self.frontend_process = subprocess.Popen(
            [sys.executable, "start_frontend.py"], cwd=project_root
        )

    def stop_api(self):
        """Stoppt den API Server"""
        if self.api_process:
            logger.info("⏹️  Stoppe API Server...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
            self.api_process = None

    def stop_frontend(self):
        """Stoppt das Frontend"""
        if self.frontend_process:
            logger.info("⏹️  Stoppe Frontend...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            self.frontend_process = None

    def restart_services(self):
        """Startet alle Services neu"""
        logger.info("🔄 Starte Services neu...")
        self.stop_api()
        self.stop_frontend()
        time.sleep(1)
        self.start_api()
        time.sleep(3)
        self.start_frontend()

    def setup_file_watcher(self):
        """Richtet File-Watcher für Hot-Reload ein"""
        logger.info("👀 Richte File-Watcher ein...")

        handler = CodeChangeHandler(self.restart_services)
        self.observer = Observer()

        # Überwache ragagent Verzeichnis
        self.observer.schedule(handler, str(project_root / "ragagent"), recursive=True)

        # Überwache Konfigurationsdateien
        for config_file in ["pyproject.toml", "uv.lock"]:
            if (project_root / config_file).exists():
                self.observer.schedule(
                    handler, str(project_root / config_file), recursive=False
                )

        self.observer.start()

    def cleanup(self):
        """Cleanup beim Beenden"""
        logger.info("🧹 Cleanup...")
        self.running = False

        if self.observer:
            self.observer.stop()
            self.observer.join()

        self.stop_api()
        self.stop_frontend()

    def run(self):
        """Hauptausführung"""
        logger.info("=" * 60)
        logger.info("🛠️  RAGAgent Development Launcher")
        logger.info("=" * 60)
        logger.info("🔥 Hot-Reload aktiviert")
        logger.info("📂 Überwacht: ragagent/")
        logger.info("🔄 Auto-Restart bei Code-Änderungen")
        logger.info("")

        try:
            # File-Watcher starten
            self.setup_file_watcher()

            # Services starten
            self.start_api()
            time.sleep(3)
            self.start_frontend()

            logger.info("✅ Development Environment gestartet!")
            logger.info("📍 API: http://localhost:8000")
            logger.info("🔗 API Docs: http://localhost:8000/docs")
            logger.info("🎨 Frontend: http://localhost:7860")
            logger.info("⏹️  Drücke Ctrl+C zum Beenden")
            logger.info("")

            # Hauptschleife
            while self.running:
                time.sleep(1)

                # Prüfe ob Prozesse noch laufen
                if self.api_process and self.api_process.poll() is not None:
                    logger.warning("⚠️  API Server beendet - starte neu...")
                    self.start_api()

                if self.frontend_process and self.frontend_process.poll() is not None:
                    logger.warning("⚠️  Frontend beendet - starte neu...")
                    self.start_frontend()

        except KeyboardInterrupt:
            logger.info("🛑 Development Launcher wird beendet...")
        except Exception as e:
            logger.error(f"❌ Fehler im Development Launcher: {e}")
        finally:
            self.cleanup()


def signal_handler(signum, frame):
    """Signal Handler für graceful shutdown"""
    logger.info("🛑 Signal empfangen - beende...")
    sys.exit(0)


def main():
    """Hauptfunktion"""
    # Signal Handler registrieren
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Launcher starten
    launcher = DevLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
