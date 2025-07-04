import argparse
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.load import dumps, loads

load_dotenv()

# LANGCHAIN API
os.getenv('LANGCHAIN_TRACING_V2')
os.getenv('LANGCHAIN_ENDPOINT')
os.getenv('LANGCHAIN_API_KEY')

# OPEN AI API
os.getenv('OPENAI_API_KEY')



# callback function used in the rag pipeline
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    #### RETRIEVAL and GENERATION ####
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    ## prompt
    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    # Retrieve
    # question = "What is task decomposition for LLM agents?"
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    # docs = retrieval_chain.invoke({"question":question})
 

    from operator import itemgetter

    # RAG
    template = """Answer the following question based on this context and if there is no answer in context, say "there is no answer based on document":

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    response = final_rag_chain.invoke({"question":args.query})

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
    