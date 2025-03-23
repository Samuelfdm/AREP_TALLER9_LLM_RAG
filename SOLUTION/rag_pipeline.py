import os
import getpass
import pinecone
import bs4
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Solicitar claves API si no están definidas
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

if "PINECONE_API_KEY" not in os.environ:
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

if "PINECONE_ENV" not in os.environ:
    os.environ["PINECONE_ENV"] = input("Enter your Pinecone environment (e.g., us-east-1-aws): ")

index_name = "langchain-test-index"

# Inicializar Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Verificar si el índice ya existe, si no, crearlo
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,  # Dimensión para OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.environ["PINECONE_ENV"])
    )

# Cargar contenido desde Wikipedia
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/LangChain",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(["p", "h1", "h2"])
    ),
)
documents = loader.load()

# Dividir el contenido en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)

# Generar embeddings con OpenAI
embedding_function = OpenAIEmbeddings()

# Conectar con Pinecone y guardar los embeddings
vector_db = PineconeVectorStore.from_documents(all_splits, embedding_function, index_name=index_name)

# Crear el retriever
retriever = vector_db.as_retriever()

# Inicializar modelo de lenguaje
llm = ChatOpenAI(model_name="gpt-4o-mini")

# Crear pipeline de RAG con retorno de documentos fuente
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

# Caso de prueba: consulta sobre langchain
query = "What is LangChain and how does it work?"
result = qa_chain(query)

# Mostrar respuesta y documentos fuente
print("\n===== Respuesta Generada =====")
print(result["result"])

print("\n===== Documentos Fuente =====")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']} - {doc.page_content[:200]}...")