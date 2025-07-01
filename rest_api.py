from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = Flask(__name__)

# LANGCHAIN API
os.getenv('LANGCHAIN_TRACING_V2')
os.getenv('LANGCHAIN_ENDPOINT')
os.getenv('LANGCHAIN_API_KEY')

# OPEN AI API
os.getenv('OPENAI_API_KEY')

# Initialize components once (could be moved to a separate initialization function)
file_path = '/Volumes/SUNSE/projects/rag-CS-chatbot/dataset/bd_hypodermic_catalog.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.smith.langchain.com")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400
        
        response = rag_chain.invoke(data['query'])
        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)