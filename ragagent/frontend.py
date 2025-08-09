"""
Gradio Frontend für das Agentic RAG System

Professionelles Frontend mit Monochrome Theme und vollem Layout.
"""

import gradio as gr
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from ragagent.pipeline import AgenticRAGPipeline, ProcessingRequest
from ragagent.models import PipelineConfig
from ragagent.utils.logging import get_logger


class RAGFrontend:
    """Frontend Klasse für das Agentic RAG System"""

    def __init__(self):
        self.logger = get_logger("rag_frontend")
        self.pipeline = AgenticRAGPipeline()
        self.session_state = {}

        # Theme Konfiguration
        self.theme = gr.Theme().Monochrome(
            primary_hue="blue", secondary_hue="slate", neutral_hue="slate"
        )

        # CSS für professionelles Aussehen
        self.custom_css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        
        .chat-message {
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            max-width: 80%;
        }
        
        .user-message {
            background-color: #f3f4f6;
            margin-left: auto;
            text-align: right;
        }
        
        .assistant-message {
            background-color: #e5e7eb;
            margin-right: auto;
        }
        
        .citation {
            background-color: #f9fafb;
            border-left: 3px solid #3b82f6;
            padding: 8px;
            margin: 4px 0;
            font-size: 0.9em;
        }
        
        .confidence-bar {
            height: 4px;
            background-color: #e5e7eb;
            border-radius: 2px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .confidence-fill {
            height: 100%;
            background-color: #3b82f6;
            transition: width 0.3s ease;
        }
        
        .loading-spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f4f6;
            border-top: 2px solid #3b82f6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-success {
            background-color: #10b981;
        }
        
        .status-error {
            background-color: #ef4444;
        }
        
        .status-processing {
            background-color: #f59e0b;
        }
        """

    def create_interface(self) -> gr.Interface:
        """Erstellt die Gradio Benutzeroberfläche"""

        with gr.Blocks(
            title="Agentic RAG System",
            theme=self.theme,
            css=self.custom_css,
            fill_height=True,
            fill_width=True,
        ) as interface:
            # Header
            gr.Markdown(
                """
                # Agentic RAG System
                
                Ein fortschrittliches Retrieval-Augmented Generation System mit agente-basierten Ansätzen.
                """
            )

            # Hauptbereich
            with gr.Row():
                # Linke Seite: Dokumentenverarbeitung
                with gr.Column(scale=1):
                    gr.Markdown("## Dokumentenverarbeitung")

                    # Datei Upload
                    file_upload = gr.File(
                        label="Dokument hochladen",
                        file_types=[".txt", ".pdf", ".docx", ".md", ".csv"],
                        file_count="single",
                    )

                    # Oder Text direkt eingeben
                    gr.Markdown("ODER")
                    document_text = gr.Textbox(
                        label="Oder Text direkt eingeben:",
                        lines=10,
                        placeholder="Fügen Sie hier Ihren Text ein...",
                    )

                    # Verarbeitungsparameter
                    with gr.Accordion("Erweiterte Einstellungen", open=False):
                        max_tokens = gr.Slider(
                            label="Maximale Token",
                            minimum=1000,
                            maximum=8000,
                            value=4000,
                            step=100,
                        )

                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.1,
                        )

                        enable_verification = gr.Checkbox(
                            label="Antwortverifikation aktivieren", value=True
                        )

                    # Verarbeitungsbutton
                    process_button = gr.Button(
                        "Dokument verarbeiten", variant="primary", size="lg"
                    )

                    # Status Anzeige
                    status_display = gr.HTML(
                        '<div class="status-indicator status-processing"></div>System bereit'
                    )

                    # Systeminformationen
                    with gr.Accordion("Systeminformationen", open=False):
                        system_info = gr.JSON(label="Systemkonfiguration", value={})

                # Rechte Seite: Chat-Interface
                with gr.Column(scale=2):
                    gr.Markdown("## Dokumentenanalyse")

                    # Chat Verlauf
                    chat_history = gr.Chatbot(
                        label="Konversationsverlauf", height=400, show_label=True
                    )

                    # Eingabebereich
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Ihre Frage:",
                            placeholder="Stellen Sie eine Frage zum hochgeladenen Dokument...",
                            lines=2,
                        )

                        send_button = gr.Button("Senden")

                    # Zusätzliche Aktionen
                    with gr.Row():
                        clear_button = gr.Button("Verlauf löschen", variant="secondary")
                        export_button = gr.Button("Exportieren", variant="secondary")

            # Ergebnisse Bereich
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Antwort")
                    answer_output = gr.Textbox(
                        label="Generierte Antwort",
                        lines=6,
                        max_lines=10,
                        show_label=True,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("## Details")
                    with gr.Tab("Zitate"):
                        citations_output = gr.JSON(label="Verwendete Zitate", value={})

                    with gr.Tab("Verifikation"):
                        verification_output = gr.JSON(
                            label="Verifikationsergebnis", value={}
                        )

                    with gr.Tab("Performance"):
                        performance_output = gr.JSON(
                            label="Performance-Metriken", value={}
                        )

            # Footer
            gr.Markdown(
                """
                ---
                Agentic RAG System - Version 0.1.0
                """
            )

            # Event Handler
            process_button.click(
                fn=self.process_document,
                inputs=[
                    file_upload,
                    document_text,
                    max_tokens,
                    temperature,
                    enable_verification,
                ],
                outputs=[status_display, system_info, chat_history],
            )

            send_button.click(
                fn=self.send_question,
                inputs=[question_input, chat_history],
                outputs=[
                    question_input,
                    chat_history,
                    answer_output,
                    citations_output,
                    verification_output,
                    performance_output,
                ],
            )

            clear_button.click(
                fn=self.clear_chat,
                outputs=[
                    chat_history,
                    answer_output,
                    citations_output,
                    verification_output,
                    performance_output,
                ],
            )

            export_button.click(
                fn=self.export_results,
                inputs=[
                    chat_history,
                    answer_output,
                    citations_output,
                    verification_output,
                ],
                outputs=[gr.Textbox(label="Exportierte Daten", lines=10)],
            )

            # Systeminformationen beim Start laden
            interface.load(fn=self.load_system_info, outputs=[system_info])

        return interface

    def process_document(
        self,
        file_path: Optional[str],
        text_input: str,
        max_tokens: int,
        temperature: float,
        enable_verification: bool,
    ) -> tuple:
        """Verarbeitet ein hochgeladenes Dokument oder eingegebenen Text"""

        try:
            status_html = '<div class="status-indicator status-processing"></div>Dokument wird verarbeitet...'

            # Bestimmen, ob Datei oder Text verwendet wird
            if file_path and file_path.strip():
                # Datei verarbeiten
                document_path = file_path
                document_type = Path(file_path).suffix.lower()
            elif text_input and text_input.strip():
                # Text in temporäre Datei schreiben
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as tmp_file:
                    tmp_file.write(text_input)
                    document_path = tmp_file.name
                    document_type = ".txt"
            else:
                raise ValueError(
                    "Bitte wählen Sie eine Datei aus oder geben Sie Text ein."
                )

            # Konfiguration aktualisieren
            config = PipelineConfig(
                agent={"max_tokens": max_tokens, "temperature": temperature},
                enable_verification=enable_verification,
            )

            # Pipeline mit neuer Konfiguration erstellen
            pipeline = AgenticRAGPipeline(config)

            # Dokument verarbeiten
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp_file:
                if file_path and file_path.strip():
                    # Datei lesen
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    tmp_file.write(content)
                else:
                    tmp_file.write(text_input)

                tmp_file_path = tmp_file.name

            # Systeminformationen vorbereiten
            system_info = {
                "status": "processing",
                "document_type": document_type,
                "config": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "enable_verification": enable_verification,
                },
            }

            # Chat Verlauf initialisieren
            chat_history = []

            # Temporäre Datei aufräumen
            if document_path != tmp_file_path:
                os.unlink(document_path)

            os.unlink(tmp_file_path)

            return status_html, system_info, chat_history

        except Exception as e:
            error_message = (
                f'<div class="status-indicator status-error"></div>Fehler: {str(e)}'
            )
            error_info = {"status": "error", "error": str(e)}
            return error_message, error_info, []

    def send_question(self, question: str, chat_history: List) -> tuple:
        """Sendet eine Frage und verarbeitet sie"""

        if not question or not question.strip():
            return "", chat_history, "", "", "", ""

        try:
            # Status aktualisieren
            status_html = '<div class="status-indicator status-processing"></div>Frage wird verarbeitet...'

            # Frage zum Chat Verlauf hinzufügen
            chat_history.append((question, ""))

            # Hier würde die eigentliche Verarbeitung stattfinden
            # Für den Mock: Simuliere eine Antwort
            time.sleep(2)  # Simulierte Verarbeitungszeit

            # Mock Antwort
            answer = f"Dies ist eine Beispielantwort auf Ihre Frage: '{question}'. Das Agentic RAG System würde hier eine echte Antwort basierend auf dem verarbeiteten Dokument generieren."

            citations = {
                "cited_chunks": ["1", "3", "5"],
                "total_chunks": 8,
                "relevance_score": 0.85,
            }

            verification = {
                "is_accurate": True,
                "confidence": "high",
                "explanation": "Die Antwort ist gut durch die Quellen gestützt",
                "issues_found": [],
            }

            performance = {
                "processing_time": 3.2,
                "steps_completed": 4,
                "tokens_processed": 1250,
            }

            # Antwort zum Chat Verlauf hinzufügen
            chat_history[-1] = (question, answer)

            return "", chat_history, answer, citations, verification, performance

        except Exception as e:
            error_message = f"Fehler bei der Verarbeitung: {str(e)}"
            chat_history.append((question, error_message))
            return "", chat_history, error_message, {}, {}, {}

    def clear_chat(self) -> tuple:
        """Löscht den Chat Verlauf"""
        return [], "", {}, {}, {}

    def export_results(
        self, chat_history: List, answer: str, citations: Dict, verification: Dict
    ) -> str:
        """Exportiert die Ergebnisse"""

        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chat_history": chat_history,
            "answer": answer,
            "citations": citations,
            "verification": verification,
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)

    def load_system_info(self) -> Dict:
        """Lädt Systeminformationen"""
        try:
            # Teste die Verbindung zum System
            info = self.pipeline.get_system_info()
            return {
                "version": info.get("version", "0.1.0"),
                "parsers": info.get("parsers", 0),
                "supported_formats": info.get("supported_formats", []),
                "config": {
                    "model": "qwen3:latest",
                    "ollama_url": "http://localhost:11434",
                },
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


def main():
    """Hauptfunktion zum Starten des Frontends"""
    frontend = RAGFrontend()

    interface = frontend.create_interface()

    # Interface starten
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        show_tips=False,
    )


if __name__ == "__main__":
    main()
