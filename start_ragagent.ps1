# RAGAgent System Launcher - PowerShell Version
# Startet API und Frontend Services für das Agentic RAG System

param(
    [string]$Service = "all",  # api, frontend, oder all
    [switch]$DevMode = $false   # Entwicklungsmodus mit Auto-Reload
)

# Farben für Output
$Colors = @{
    Info = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Header = "Magenta"
}

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $ColorValue = $Colors[$Color]
    if ($ColorValue -eq $null) {
        $ColorValue = "White"
    }
    
    Write-Host $Message -ForegroundColor $ColorValue
}

function Start-APIService {
    Write-ColoredOutput "[API] Starte API Server..." "Info"
    
    if ($DevMode) {
        Write-ColoredOutput "[DEV] Entwicklungsmodus aktiviert" "Warning"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "uv run python start_api.py"
    } else {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "uv run python start_api.py"
    }
    
    Write-ColoredOutput "[API] Server gestartet auf http://localhost:8000" "Success"
    Write-ColoredOutput "[DOC] API Dokumentation: http://localhost:8000/docs" "Info"
}

function Start-FrontendService {
    Write-ColoredOutput "[UI] Starte Frontend..." "Info"
    
    # Kurz warten, damit API Zeit hat zu starten
    Start-Sleep -Seconds 3
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "uv run python start_frontend.py"
    
    Write-ColoredOutput "[UI] Frontend gestartet auf http://localhost:7860" "Success"
    Write-ColoredOutput "[UI] Gradio Interface mit Monochrome Theme" "Info"
}

function Show-Header {
    Write-ColoredOutput "=" * 60 "Header"
    Write-ColoredOutput "RAGAgent System Launcher" "Header"
    Write-ColoredOutput "=" * 60 "Header"
    Write-ColoredOutput "Agentic RAG System mit Ollama" "Info"
    Write-ColoredOutput ""
}

function Show-Services {
    Write-ColoredOutput "Verfuegbare Services:" "Info"
    Write-ColoredOutput "  API Server: http://localhost:8000" "Info"
    Write-ColoredOutput "  Frontend: http://localhost:7860" "Info"
    Write-ColoredOutput "  API Docs: http://localhost:8000/docs" "Info"
    Write-ColoredOutput ""
}

# Hauptlogik
Show-Header

# Prüfe ob uv verfügbar ist
try {
    $uvVersion = uv --version
    Write-ColoredOutput "[OK] UV Package Manager: $uvVersion" "Success"
} catch {
    Write-ColoredOutput "[ERROR] UV Package Manager nicht gefunden!" "Error"
    Write-ColoredOutput "        Bitte installieren Sie UV: https://github.com/astral-sh/uv" "Warning"
    exit 1
}

# Prüfe aktuelles Verzeichnis
$currentDir = Get-Location
if (-not (Test-Path "ragagent")) {
    Write-ColoredOutput "[ERROR] Nicht im RAGAgent Projektverzeichnis!" "Error"
    Write-ColoredOutput "        Aktuelles Verzeichnis: $currentDir" "Warning"
    Write-ColoredOutput "        Wechseln Sie ins RAGAgent Verzeichnis und führen Sie das Script erneut aus." "Warning"
    exit 1
}

Write-ColoredOutput "[OK] Projektverzeichnis: $currentDir" "Success"
Write-ColoredOutput ""

# Services starten basierend auf Parameter
switch ($Service.ToLower()) {
    "api" {
        Write-ColoredOutput "[MODE] Starte nur API Service" "Info"
        Start-APIService
    }
    "frontend" {
        Write-ColoredOutput "[MODE] Starte nur Frontend Service" "Info"
        Start-FrontendService
    }
    "all" {
        Write-ColoredOutput "[MODE] Starte alle Services" "Info"
        Start-APIService
        Start-FrontendService
        Show-Services
        Write-ColoredOutput "[INFO] Schliessen Sie die PowerShell-Fenster zum Beenden" "Warning"
    }
    default {
        Write-ColoredOutput "[ERROR] Unbekannter Service: $Service" "Error"
        Write-ColoredOutput "        Verfügbare Optionen: api, frontend, all" "Warning"
        exit 1
    }
}

Write-ColoredOutput "[SUCCESS] RAGAgent Services erfolgreich gestartet!" "Success"
