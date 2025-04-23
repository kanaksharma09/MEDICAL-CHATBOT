import os
import streamlit as st
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    local_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=512,
        temperature=0.5
    )
    llm = HuggingFacePipeline(pipeline=local_pipeline)
    return llm

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        You are a helpful and professional medical assistant. Answer the user's medical question in a clear, complete, and well-structured paragraph using only the information provided in the context.

        Always aim to be factual, detailed, and easy to understand. Do not guess. If the answer cannot be found in the context, just say you don't know.

        Context:
        {context}

        Question:
        {question}

        Answer in a well-explained paragraph:
        """

        try:
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            # Format readable source content (not raw metadata)
            sources_text = "\n\n".join(
                f"📄 **Page {doc.metadata.get('page', '?')}**:\n{doc.page_content.strip()[:500]}..."
                for doc in source_documents
            )

            st.chat_message('assistant').markdown(f"**Answer:**\n{result}")
            st.chat_message('assistant').markdown(f"---\n**Sources:**\n{sources_text}")

            st.session_state.messages.append({'role': 'assistant', 'content': f"**Answer:**\n{result}\n\n---\n**Sources:**\n{sources_text}"})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
