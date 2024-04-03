from langchain_community.llms import Ollama, LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from  langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

import torch

# Load LLM

class AppChain:
    def __init__(self):
        pass

    def loadModel(self, model):
        # llm = CTransformers(
        #     model = model,
        #     model_type = 'llama',
        #     max_new_tokens = 1024,
        #     temperature = 0.01,
        # )

        # llm = Ollama(
        #     model = model,
        #     callback_manager=CallbackManager(
        #         [StreamingStdOutCallbackHandler()],
        #     ),
        # )

        llm = LlamaCpp(
            model_path="models/llama-2-7b.Q5_0.gguf",
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
        )



        return llm
    
    def createPrompt(self, context, template):
        prompt = PromptTemplate(template = template, input_variables = ["context", "question"])
        
        return prompt

    def createChain(self, prompt, llm, db):
        llmChain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = 'stuff',
            #chain_type_kwargs = {'prompt' : prompt},
            retriever = db.as_retriever(),
            #return_source_documents = False,
        )

        return llmChain
    
    def readVectorDB(self, vectorDBPath, model):
        
        # Embeddings
        embedding = GPT4AllEmbeddings(model_file = model)
        db = FAISS.load_local(vectorDBPath, embedding, allow_dangerous_deserialization = True)

        return db
    
    def readVectorChromaDB(self, vectorDBPath, model):
        
        # Embeddings
        device = 'cuda' if torch.cuda.is_available() else 'mps'
        sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': device})
        
        db = Chroma(persist_directory="stores", 
                   embedding_function =sentence_transformer_ef,
                   collection_name="chroma_docs")

        collection = db.get()

        return db


# Load chain and model
if __name__ == "__main__":
    chain = AppChain()
    llm = chain.loadModel("llama2-uncensored")
    
    #db = chain.readVectorDB('stores/db_faiss', 'models/all-MiniLM-L6-v2')
    db = chain.readVectorChromaDB('stores', 'models/all-MiniLM-L6-v2')


    # Geneate prompt
    promtTemplate = """<|im_start|>system
    Use information from the following text to answer the questions below 
    {context}
    <|im_end|>
    <|im_start|>user
    {question}
    <|im_end|>
    <|im_start|>assistant
    """

    # Create prompt
    prompt = chain.createPrompt('context', promtTemplate)

    # Run chain
    llmChain = chain.createChain(prompt, llm, db)
    # reponse = llmChain.invoke({'query': 'What is a checkpoint ?'})
    # print(reponse)

    print ( llmChain("How much glaucoma is engouh ?"))

    #print( db.similarity_search("What is a checkpoint ?"))