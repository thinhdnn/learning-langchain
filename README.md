
# LangChain Learning

## Description

LangChain Learning is a project aimed at familiarizing users with the LangChain language learning platform through a series of guided tutorials and exercises. The project provides a hands-on approach to understanding the features and functionalities of LangChain, enabling users to maximize their learning experience.

## Installation

To get started with LangChain Learning, follow these steps:

- Clone the GitHub repository to your local machine:
    ```
    git clone https://github.com/langchain-learning/langchain-learning.git
    ```

- Install dependencies using pip:
    ```
    pip install -r requirements.txt
    ```

## Use
-  Create the vector database
    ```
    Use this for FAISS CPU: 
    vectorDb.createFromFilesByFAISS(dataPath = dataPath, vectorDBPath = vectorDBPath)

	Use this for ChromaDB with GPU
	vectorDb.createFromFilesByChroma(dataPath = dataPath, vectorDBPath = vectorDBPath)
    ```
    ```
    python /app/VectorDB.py
    ```
    
- Create the chromadb with multiple processing:
    ```
    python /app/ChrombaDB.py
    ```

- Run the model:
    ```
    python /app/AppChain.py
    ```

## Contributing

LangChain Learning is an open-source project, and contributions are welcome. Please fork the repository, make your changes, and submit a pull request.
