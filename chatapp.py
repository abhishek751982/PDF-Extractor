import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print(f"✅ FAISS index created with {len(text_chunks)} documents.")



def get_conversational_chain():
    prompt_template = """
You are an AI assistant that provides detailed explanations based on the given context.
Summarize or explain the most relevant information from the provided context.

Context:
{context}

Question:
{question}

If the question is vague, provide a general summary of the most relevant details.

Answer:
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain




def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question, k=10)  # Retrieve more documents

    if not docs:
        print("❌ No relevant documents retrieved!")
        st.write("❌ No relevant documents found. Try rephrasing your query.")
        return
    else:
        print("✅ Retrieved Documents:")
        for i, doc in enumerate(docs):
            print(f"Document {i+1}:\n{doc.page_content[:500]}\n")  # Show first 500 characters

    chain = get_conversational_chain()

    # Extract text from retrieved documents to use as context
    context = "\n\n".join([doc.page_content for doc in docs])

    print(f"📄 Context Passed to LLM:\n{context}")
    print(f"📄 Question: {user_question}")

    # Ensure context isn't empty before passing it
    if not context.strip():
        print("⚠️ Context is empty! LLM will not get useful information.")
        st.write("⚠️ Retrieved context is empty. Try re-uploading the document or refining your question.")
        return

    # Corrected invocation to match LangChain's expected inputs
    response = chain.invoke({"input_documents": docs, "question": user_question})

    print("🤖 Full AI Response:", response)  # Print full response before accessing any key

    # Check if `output_text` exists in the response
    if "output_text" in response:
        ai_response = response["output_text"]
    elif "text" in response:
        ai_response = response["text"]
    else:
        ai_response = f"⚠️ Unexpected response format: {response}"

    st.write("Reply: ", ai_response)



def main():
    st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): # user friendly message.
                raw_text = get_pdf_text(pdf_docs) # get the pdf text
                text_chunks = get_text_chunks(raw_text) # get the text chunks
                get_vector_store(text_chunks) # create vector store
                st.success("Done")
        
        st.write("---")


if __name__ == "__main__":
    main()