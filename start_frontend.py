#!/usr/bin/env python3
"""
Startscript für das RAGAgent Gradio Frontend
Startet das Gradio Interface für das Agentic RAG System
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ragagent.frontend import RAGFrontend
from ragagent.utils.logging import get_logger

logger = get_logger("frontend_server")


def main():
    """Hauptfunktion zum Starten des Frontend-Servers"""

    # Konfiguration
    config = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,  # Auf True setzen für öffentlichen Link
        "debug": True,
        "show_error": True,
        "quiet": False,
    }

    logger.info("🎨 Starte RAGAgent Gradio Frontend...")
    logger.info(
        f"📍 Frontend wird gestartet auf http://{config['server_name']}:{config['server_port']}"
    )
    logger.info("💻 Gradio Version: 5.34.0")
    logger.info("🎯 Monochrome Theme aktiviert")

    try:
        # Frontend initialisieren
        frontend = RAGFrontend()

        # Interface erstellen
        interface = frontend.create_interface()

        # Server starten
        interface.launch(**config)

    except KeyboardInterrupt:
        logger.info("🛑 Frontend wurde durch Benutzer gestoppt")
    except Exception as e:
        logger.error(f"❌ Fehler beim Starten des Frontends: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
