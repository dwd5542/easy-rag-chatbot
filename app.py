import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
client=genai.Client(api_key=os.getenv("gemini_api_key"))

@st.cache_data
def load_and_embed(pdf_path):
    reader=PdfReader(pdf_path)
    pdf_text=""
    for page in reader.pages:
        pdf_text+=page.extract_text()
    
    chunk_size=500
    overlap=100
    step=chunk_size-overlap
    chunks=[]
    for i in range(0,len(pdf_text),step):
        chunks.append(pdf_text[i:i+chunk_size])
    
    chunk_embeddings=[]
    for i in range(0,len(chunks),100):
        batch=chunks[i:i+100]
        result=client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch
        )
        for e in result.embeddings:
            chunk_embeddings.append(e.values)
    return chunks, chunk_embeddings

def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def search(query,chunks,chunk_embeddings,top_k=3):
    query_result=client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_embedding=query_result.embeddings[0].values

    scores=[]
    for i in range(len(chunks)):
        score=cosine_similarity(query_embedding,chunk_embeddings[i])
        scores.append((score,chunks[i]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scores[:top_k]]

# ----- 화면 -----    
st.title("📄 PDF RAG chatbot")
st.write("pdf 문서에 대해 질문해보세요.")

chunks, chunk_embeddings=load_and_embed("doc.pdf")

user_input=st.text_input("query:")

if user_input:
    relevant_chunks=search(user_input,chunks,chunk_embeddings)
    context="\n\n".join(relevant_chunks)

    prompt=f""" 다음 문서 내용을 참고해서 질문에 답해줘.
문서에 없는 내용이면 모른다고 답해줘.

문서 내용:
{context}

질문:{user_input}
"""

    response=client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.write(response.text)