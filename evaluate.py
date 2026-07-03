test_set = [
    {"question": "이 논문이 제안하는 모델 이름은?", "keyword": "Transformer"},
    {"question": "인코더는 몇 개의 층으로 이루어져 있나?", "keyword": "6"},
    {"question": "어떤 종류의 어텐션을 사용하나?", "keyword": "attention"},
    {"question": "포지셔널 인코딩에 어떤 함수를 사용하나?", "keyword": "sine"},
    {"question": "성능을 평가한 작업은?", "keyword": "translation"},
    {"question": "디코더는 몇 개의 층으로 이루어져 있나?", "keyword": "6"},
    {"question": "여러 어텐션을 병렬로 쓰는 방식의 이름은?", "keyword": "multi-head"},
    {"question": "이 모델이 없애버린 기존 구조는 무엇인가?", "keyword": "recurrence"},
    {"question": "어텐션 계산에서 무엇으로 나눠 스케일링하는가?", "keyword": "dk"},
    {"question": "각 층에 있는 어텐션 외의 다른 서브층은?", "keyword": "feed-forward"},
    {"question": "훈련에 사용한 GPU는?", "keyword": "P100"},
    {"question": "영어-독일어 번역은 어떤 데이터셋으로 평가했나?", "keyword": "WMT"},
    {"question": "번역 품질을 재는 평가 지표는?", "keyword": "BLEU"},
    {"question": "학습에 사용한 옵티마이저는?", "keyword": "Adam"},
    {"question": "과적합 방지를 위해 사용한 기법은?", "keyword": "dropout"},
]

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

sentences=pdf_text.split(".")

chunk_size=500
chunks=[]
current=""
for sentence in sentences:
    if len(current)+len(sentence)<chunk_size:
        current+=sentence+". "
    else:
        chunks.append(current)
        current=sentence+". "
if current:
    chunks.append(current)

print("chunk conut:", len(chunks))

print("\nchunks embedding...")

chunk_embeddings=[]

for i in range(0,len(chunks),100):
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
    threshold=0.55
    selected=[chunk for score, chunk in scores if score>=threshold]

    if not selected:
        selected=[chunk for score, chunk in scores[:top_k]]

    return selected


correct=0

for item in test_set:
    question=item["question"]
    keyword=item["keyword"]

    relevant_chunks=search(question,chunks,chunk_embeddings)
    combined=" ".join(relevant_chunks)

    if keyword.lower() in combined.lower():
        result="O"
        correct+=1
    else:
        result="X"
    print(f"[{result}] {question} (키워드: {keyword})")
print(f"\n정답률: {correct}/{len(test_set)}")

