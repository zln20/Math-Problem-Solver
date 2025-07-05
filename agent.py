from PIL import Image
import numpy as np
import io
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Setup LLM using OpenRouter DeepSeek
llm = ChatOpenAI(
    openai_api_key="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",
    openai_api_base="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1:free",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that solves math problems step-by-step."),
    ("human", "{question}")
])

chain = prompt | llm

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from an image using EasyOCR.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    results = reader.readtext(image_np, detail=0)
    extracted_text = " ".join(results).strip()

    return extracted_text

def solve_math_problem_from_image(image_bytes: bytes) -> str:
    """
    Extract the math problem text from the image and solve it via LLM.
    """
    extracted_text = extract_text_from_image(image_bytes)

    if not extracted_text:
        raise ValueError("No text could be extracted from the image.")

    response = chain.invoke({"question": extracted_text})
    return response.content.strip()
