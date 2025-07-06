import easyocr
from PIL import Image
import numpy as np
import io
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Initialize OpenRouter model (DeepSeek R1 - FREE)
key = st.secrets["openai"]["key"]
llm = ChatOpenAI(
    openai_api_key=key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1:free",
    temperature=0
)

# Chat-style prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that solves math problems step-by-step."),
    ("human", "{question}")
])

# Combine into LangChain pipeline
chain = prompt | llm

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extracts text from an image using EasyOCR.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    results = reader.readtext(image_np, detail=0)
    extracted_text = " ".join(results).strip()

    return extracted_text

def solve_math_problem_from_image(image_bytes: bytes) -> str:
    """
    Extracts the math problem using OCR and uses an LLM to solve it.
    """
    extracted_text = extract_text_from_image(image_bytes)

    if not extracted_text:
        raise ValueError("No text could be extracted from the image.")

    response = chain.invoke({"question": extracted_text})
    return response.content.strip()


