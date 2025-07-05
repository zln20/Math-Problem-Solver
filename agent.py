from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import io
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

reader = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    ocr_version='PP-OCRv3'
)

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
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    results = reader.ocr(image_np, cls=False)
    extracted_text = " ".join([line[1][0] for line in results]).strip()
    return extracted_text

def solve_math_problem_from_image(image_bytes: bytes) -> str:
    extracted_text = extract_text_from_image(image_bytes)
    if not extracted_text:
        raise ValueError("No text could be extracted from the image.")
    response = chain.invoke({"question": extracted_text})
    return response.content.strip()
