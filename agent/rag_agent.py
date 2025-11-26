import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader

# WICHTIG: Chains kommen mit LangChain 1.0 aus langchain_classic
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# --- KONSTANTEN ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "chroma_db"
PROCESSED_DIR = "data/processed"

SYSTEM_PROMPT = (
    "Du bist der AI Compliance Agent von NeuralBlau. "
    "Beantworte Fragen ausschließlich auf Basis des dir im Kontext ('{context}') "
    "bereitgestellten Dokumentenmaterials. "
    "Antworte präzise, klar und professionell. "
    "Wenn die Information eindeutig nicht im Kontext steht, antworte: "
    "'Die angeforderte Information ist in den bereitgestellten Dokumenten nicht enthalten.'"
)


class ComplianceAgent:
    def __init__(self):
        # Embedding Modell
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY nicht gefunden. Bitte in .env setzen."
            )

        # Moderner Chat-LLM
        self.llm = ChatOpenAI(temperature=0)

        self.db = None
        self.retrieval_chain = None

    def create_or_load_index(self, force_rebuild: bool = False):
        """Erstellt den Vektordatenbank-Index oder lädt ihn und baut die Retrieval-Chain auf."""
        if os.path.exists(DB_PATH) and not force_rebuild:
            print("Lade bestehenden Chroma-Index...")
            self.db = Chroma(
                persist_directory=DB_PATH,
                embedding_function=self.embedding_function,
            )
        else:
            print("Erstelle neuen Chroma-Index...")
            self._build_index()

        # ---------- LCEL / moderne Chain-Struktur ----------

        # 1. Prompt-Template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )

        # 2. Dokument-Kette (Documents + Prompt → LLM)
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # 3. Retrieval-Chain (Retriever → Dokument-Kette)
        retriever = self.db.as_retriever(search_kwargs={"k": 8})
        self.retrieval_chain = create_retrieval_chain(retriever, document_chain)

        print("RAG Retrieval Chain erfolgreich aufgebaut.")

    def _build_index(self):
        """Lädt Textdateien, splittet sie und speichert sie in Chroma."""
        documents = []

        for filename in os.listdir(PROCESSED_DIR):
            if filename.endswith(".txt"):
                loader = TextLoader(
                    os.path.join(PROCESSED_DIR, filename),
                    encoding="utf-8",
                )
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Dokumente geladen und in {len(chunks)} Chunks aufgeteilt.")

        self.db = Chroma.from_documents(
            chunks,
            self.embedding_function,
            persist_directory=DB_PATH,
        )

        self.db.persist()
        print("Indexing abgeschlossen.")

    def query(self, question: str) -> str:
        """Führt die RAG-Abfrage durch (über die LCEL-Kette)."""
        if not self.retrieval_chain:
            raise RuntimeError(
                "Index wurde noch nicht geladen/erstellt. "
                "Bitte zuerst create_or_load_index() aufrufen."
            )

        result = self.retrieval_chain.invoke({"input": question})

        # `create_retrieval_chain` liefert standardmäßig 'answer' + 'context'
        return result["answer"]


if __name__ == "__main__":
    print("Starte RAG Core Test...")
    try:
        agent = ComplianceAgent()
        agent.create_or_load_index(force_rebuild=False)

        test_query = "Was sagt die MIT lizenz?"
        response = agent.query(test_query)

        print("\n--- RAG ANTWORT ---")
        print(f"Frage: {test_query}")
        print(f"Antwort: {response}")
    except EnvironmentError as e:
        print(f"Fehler: {e}. Bitte prüfe .env und OPENAI_API_KEY.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")