import chromadb
from chromadb.utils import embedding_functions
import time
import multiprocessing as mp
import csv


class ChromaDB:
    def producer(self, fileName, batchSize, queue):
        with open(fileName, encoding='utf8') as file:
            # Start id=2 to match the id with the line number in the csv (skipping the row 1 column header)
            # Store the name of the menu items in this array. In Chroma, a "document" is a string i.e. name, sentence, paragraph, etc.
            # Store the corresponding menu item IDs in this array.
            # Each "document" needs a unique ID. This is like the primary key of a relational database
            lines = csv.reader(file)
            next(lines)
            id = 2
            documents = []  
            ids = [] 
            for line in lines:
                document = f"In the document, we have the information  \"{line[0]}\", \"{line[1]}\", "
                documents.append(document)
                ids.append(str(id))
                if len(ids)>=batchSize:
                    queue.put((documents, ids))
                    documents = []
                    ids = []

                id+=1

            # Queue last batch
            if(len(ids)>0):
                queue.put((documents, ids))
                

    # Worker function to get items from the queue
    def consumer(self, use_cuda, queue):
        chromaClient = chromadb.PersistentClient(path="stores")
        device = 'cuda' if use_cuda else 'cpu'
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2", device=device)
        collection = chromaClient.get_collection(name="got", embedding_function=sentence_transformer_ef)
        while True:
            batch = queue.get()
            if batch is None:
                break
            
            # Add to collection
            collection.add(
                documents=batch[0],
                ids=batch[1]
            )


if __name__ == "__main__":
    chromaDB = ChromaDB()
    chromaClient = chromadb.PersistentClient(path="stores")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    try:
        chromaClient.get_collection(name="got")
        chromaClient.delete_collection(name="got")
    except Exception as err:
        print(err)

    collection = chromaClient.create_collection(name="got", embedding_function=sentence_transformer_ef)
    queue = mp.Queue()

    # Create producer and consumer processes.
    producer_process = mp.Process(target= chromaDB.producer, args=('data/data-file.csv', 5000, queue,))
    consumer_process = mp.Process(target=chromaDB.consumer, args=(True, queue,))
    
    # Do not create multiple consumer processes, because ChromaDB is not multiprocess safe.
    start_time = time.time()

    # Start processes
    producer_process.start()
    consumer_process.start()

    # Wait for producer to finish producing
    producer_process.join()

    # Signal consumer to stop consuming by putting None into the queue. Need 2 None's to stop 2 consumers.    
    queue.put(None)

    # Wait for consumer to finish consuming
    consumer_process.join()

    print(f"Elapsed seconds: {time.time()-start_time:.0f} Record count: {collection.count()}")