import streamlit as st
from agent import solve_math_problem_from_image, extract_text_from_image

st.set_page_config(page_title="Math Solver", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Math Problem Solver</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Upload an image of a math question</p>", unsafe_allow_html=True)

st.markdown("---")

with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown("""
    1. Click on **Browse files** to upload an image of a math question.
    2. The system will extract text using OCR and display the recognized question.
    3. It will then pass the question to an LLM and show you the step-by-step solution.
    
    **Best Results:** Use clear printed images (handwritten may not be accurate).
    """)


import re

def clean_latex_answer(text):
    return re.sub(r"\\boxed{(.*?)}", r"\1", text)

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()

    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Extracting text from image..."):
        try:
            question = extract_text_from_image(image_bytes)
            if question:
                with st.container():
                    st.subheader("📝 Extracted Question")
                    st.code(question, language="markdown")
            else:
                st.error("Could not extract any text from the image.")
        except Exception as e:
            st.error(f"OCR Error: {e}")

    with st.spinner("Solving the problem..."):
        try:
            solution = solve_math_problem_from_image(image_bytes)
            with st.container():
                st.success("Solution")
                st.write(clean_latex_answer(solution))
        except Exception as e:
            st.error(f"Error: {e}")
