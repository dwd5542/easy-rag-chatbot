import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)

reader = PdfReader("doc.pdf")
pdf_text=""
for page in reader.pages:
    pdf_text += page.extract_text()
print("pdf에서 뽑은 글자 수 :", len(pdf_text))

chunk_size=100
overlap=20
step=chunk_size-overlap
chunks=[]
for i in range(0, len(pdf_text), step):
    chunk=pdf_text[i:i+chunk_size]
    chunks.append(chunk)

print("chunk conut:", len(chunks))
print("first chunk preview:\n", chunks[0])

print("\nchunks embedding...")

chunk_embeddings=[]

for i in range(0,len(chunks),chunk_size):
    batch=chunks[i:i+100]
    result=client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch
    )
    for e in result.embeddings:
        chunk_embeddings.append(e.values)

print("embeddings count:", len(chunk_embeddings))
print("first embedding length:", len(chunk_embeddings[0]))

def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

def search(query,top_k=3):
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
    threshold=0.65
    selected=[chunk for score, chunk in scores if score>=threshold]

    if not selected:
        selected=[chunk for score, chunk in scores[:top_k]]

    return selected

while True:
    user_input = input("Enter your message(if you want to exit, type 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    relevant_chunks=search(user_input)
    context="\n\n".join(relevant_chunks)

    prompt = f"""다음 문서 내용을 참고해서 질문에 답해줘.
문서에 없는 내용이면 모른다고 답해줘.

문서 내용:
{context}

질문: {user_input}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print(response.text)

