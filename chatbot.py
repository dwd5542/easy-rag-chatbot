import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")

client = genai.Client(api_key=api_key)

history=[]

while True:
    user_input = input("Enter your message(if you want to exit, type 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    history.append({"role": "user", "parts": [{"text": user_input}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history
    )

    answer = response.text
    print(answer)

    history.append({"role": "model", "parts": [{"text": answer}]})
