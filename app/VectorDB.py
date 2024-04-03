from  langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS, Chroma
from  langchain_community.embeddings import GPT4AllEmbeddings, OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import torch

dataPath = 'data'
vectorDBPath = 'stores'
vectorDBName = 'db_faiss'


class VectorDatabase:
    def __init__(self):
        pass

    def createFromFilesByFAISS(self, dataPath, vectorDBPath):
        # Load data from files
        dirLoader = DirectoryLoader(dataPath, glob = '*.csv', loader_cls = CSVLoader)
        documents = dirLoader.load()

        # Split text into characters
        textSplitters = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        chunks = textSplitters.split_documents(documents)
        
        # Generate embeddings
        embeddings = GPT4AllEmbeddings(model_file = 'models/all-MiniLM-L6-v2')
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(vectorDBPath)

        return db
    
    def createFromDirByFAISS(self, dataPath, vectorDBPath):
        # Load data from files
        dirLoader = DirectoryLoader(dataPath)
        documents = dirLoader.load()

        # Split text into characters
        textSplitters = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        chunks = textSplitters.split_documents(documents)
        
        # Generate embeddings
        embeddings = GPT4AllEmbeddings(model_file = 'models/all-MiniLM-L6-v2')
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(vectorDBPath)

        return db

    def createFromFilesByChroma(self, dataPath, vectorDBPath):
        # Load data from files
        dirLoader = DirectoryLoader(dataPath, glob = '*.pdf', loader_cls = PyPDFLoader)
        documents = dirLoader.load()

        # Split text into characters
        textSplitters = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        chunks = textSplitters.split_documents(documents)
        
        # Generate embeddings
        device = 'cuda' if torch.cuda.is_available() else 'mps'
        sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': device})
        db = Chroma.from_documents(
                            documents=chunks, 
                            embedding=sentence_transformer_ef, 
                            persist_directory=vectorDBPath,
                            collection_name= "chroma_docs"
                        )
        db.persist()
        return db


if __name__ == "__main__":
    vectorDb = VectorDatabase()
    #vectorDb.createFromFilesByFAISS(dataPath = dataPath, vectorDBPath = vectorDBPath)
    vectorDb.createFromFilesByChroma(dataPath = dataPath, vectorDBPath = vectorDBPath)
    #vectorDb.createFromDir(dataPath = dataPath, vectorDBPath = vectorDBPath)