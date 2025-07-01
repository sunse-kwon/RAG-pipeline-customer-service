import argparse
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
# LANGCHAIN API
os.getenv('LANGCHAIN_TRACING_V2')
os.getenv('LANGCHAIN_ENDPOINT')
os.getenv('LANGCHAIN_API_KEY')

# OPEN AI API
os.getenv('OPENAI_API_KEY')



def rag_pipeline():
    #### INDEXING ####
    # Load
    file_path = '/Volumes/SUNSE/projects/rag-CS-chatbot/dataset/bd_hypodermic_catalog.pdf'
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Store
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    #### RETRIEVAL and GENERATION ####
    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # Prompt
    prompt = hub.pull("rlm/rag-prompt", api_url="https://api.smith.langchain.com")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Question
    response = rag_chain.invoke(args.query)

    return print(response)




if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="basic RAG chatbot")
    
    # Define command-line arguments
    parser.add_argument('--query', type=str,
                       help='question to ask LLM')    
    # Parse arguments
    args = parser.parse_args()
    
    # Start the server
    rag_pipeline()
