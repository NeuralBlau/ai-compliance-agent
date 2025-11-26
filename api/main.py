import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Importiert den Compliance Agent
from agent.rag_agent import ComplianceAgent

# Initialisierung des FastAPI-App und des RAG-Agenten
app = FastAPI(
    title="NeuralBlau AI Compliance Agent API",
    description="REST API für die Compliance-Abfrage basierend auf Vektordatenbanken (RAG).",
    version="1.0.0",
)

# Globale Variable für den Compliance Agent
rag_agent: ComplianceAgent = None

# --- Pydantic Modelle für API-Datenstrukturen ---

class QueryRequest(BaseModel):
    """Definiert das Format für die eingehende Query-Anfrage."""
    query: str = Field(..., description="Die Compliance-Frage, die an den Agenten gestellt werden soll.", min_length=1)

class QueryResponse(BaseModel):
    """Definiert das Format für die ausgehende Antwort."""
    answer: str = Field(..., description="Die durch den RAG-Agenten generierte, faktenbasierte Antwort.")
    
# --- API LIFECYCLE HOOKS ---

@app.on_event("startup")
async def startup_event():
    """
    Diese Funktion wird einmal beim Start des Servers ausgeführt.
    Sie lädt den RAG-Index in den Speicher (RAM) der Anwendung.
    """
    global rag_agent
    print("Starte Agent und lade RAG-Index...")
    try:
        rag_agent = ComplianceAgent()
        # force_rebuild=False, um den bestehenden Index zu verwenden
        rag_agent.create_or_load_index(force_rebuild=False)
        print("RAG-Index erfolgreich geladen und Agent bereit.")
    except Exception as e:
        print(f"FATAL ERROR beim Laden des RAG-Agenten: {e}")
        # Server-Start verhindern, wenn der Agent nicht geladen werden kann
        raise RuntimeError(f"RAG-Agent konnte nicht initialisiert werden: {e}")

# --- API ENDPUNKTE ---

@app.get("/health")
def health_check():
    """Überprüft den Status des Servers und des RAG-Agenten."""
    status = "ready" if rag_agent else "loading"
    return {"status": status, "message": "Compliance Agent API ist aktiv."}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Verarbeitet eine Compliance-Anfrage und liefert die RAG-generierte Antwort.
    """
    if not rag_agent:
        raise HTTPException(status_code=503, detail="Agent noch nicht initialisiert.")
    
    try:
        # Führt die RAG-Abfrage durch
        response = rag_agent.query(request.query)
        
        return QueryResponse(answer=response)
        
    except Exception as e:
        print(f"Fehler bei der Abfrage: {e}")
        # Bietet einen hilfreichen Fehler für den Client
        raise HTTPException(
            status_code=500, 
            detail="Fehler bei der RAG-Verarbeitung. Bitte prüfen Sie die Server-Logs."
        )

# --- Server Start (für lokalen Test) ---
if __name__ == "__main__":
    # Achtung: Uvicorn wird im Dev Container mit dem Befehl in Schritt 3 gestartet
    # Dieser Block dient nur zum lokalen Test
    uvicorn.run(app, host="0.0.0.0", port=8000)