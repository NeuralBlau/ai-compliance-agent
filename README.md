# ai-compliance-agent

End-to-End MLOps Projekt zur Entwicklung eines Retrieval-Augmented Generation (RAG)-Agenten für Compliance-Abfragen. Beinhaltet alle Schritte von der Datenaufnahme bis zum Deployment ab.

Die Pipeline umfasst die Datenaufnahme von AWS S3, Tesseract OCR zur Dokumentenverarbeitung und die Indexierung in ChromaDB mit multilingualen Embeddings. Der RAG Core basiert auf LangChain LCEL. 
Das Deployment erfolgt als FastAPI/Uvicorn REST Service in einem Docker Dev Container.