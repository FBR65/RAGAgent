#!/usr/bin/env python3
"""
Startscript für die RAGAgent API
Startet einen FastAPI-Server für das Agentic RAG System
"""

import uvicorn
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ragagent.api import app
from ragagent.utils.logging import get_logger

logger = get_logger("api_server")


def main():
    """Hauptfunktion zum Starten des API-Servers"""

    # Konfiguration
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "workers": 1,  # Wichtig für Konsistenz mit shared clients
    }

    logger.info("🚀 Starte RAGAgent API Server...")
    logger.info(
        f"📍 Server wird gestartet auf http://{config['host']}:{config['port']}"
    )
    logger.info("📖 API Dokumentation verfügbar unter: http://localhost:8000/docs")
    logger.info("🔄 Auto-Reload ist aktiviert für Entwicklung")

    try:
        # FastAPI Server starten
        uvicorn.run("ragagent.api:app", **config)
    except KeyboardInterrupt:
        logger.info("🛑 Server wurde durch Benutzer gestoppt")
    except Exception as e:
        logger.error(f"❌ Fehler beim Starten des Servers: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
