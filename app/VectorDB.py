from  langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from  langchain_community.embeddings import GPT4AllEmbeddings


dataPath = 'data'
vectorDBPath = 'stores/db_faiss'


class VectorDatabase:
    def __init__(self):
        pass

    def createFromFiles(self, dataPath, vectorDBPath):
        # Load data from files
        dirLoader = DirectoryLoader(dataPath, glob = '*.pdf', loader_cls = PyPDFLoader)
        documents = dirLoader.load()

        # Split text into characters
        textSplitters = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        chunks = textSplitters.split_documents(documents)
        
        # Generate embeddings
        embeddings = GPT4AllEmbeddings(model_file = 'models/all-MiniLM-L6-v2')
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(vectorDBPath)

        return db
    
    def createFromDir(self, dataPath, vectorDBPath):
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



vectorDb = VectorDatabase()
vectorDb.createFromFiles(dataPath = dataPath, vectorDBPath = vectorDBPath)
#vectorDb.createFromDir(dataPath = dataPath, vectorDBPath = vectorDBPath)