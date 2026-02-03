import streamlit as st
from model_def import load_all, generate

st.set_page_config(page_title="Code → Docstring Generator")

st.title("🧠 Code Documentation Generator")
st.write("Paste Python code and generate a docstring using a Transformer model.")

code = st.text_area(
    "Python Function",
    height=200,
    placeholder="def add(a, b):\n    return a + b"
)

if st.button("Generate Docstring"):
    if code.strip():
        # Prepend prompt to code
        code_with_prompt = f"""# Write a Python docstring explaining the purpose of this code
{code}
"""
        with st.spinner("Generating..."):
            try:
                tokenizer, model = load_all()
                output = generate(code_with_prompt, tokenizer, model)
                st.subheader("Generated Docstring")
                st.code(output)
            except Exception as exc:
                st.error(
                    "Model failed to load. Ensure PyTorch is installed correctly "
                    "and required system libraries are available.\n\n"
                    f"Details: {exc}"
                )
    else:
        st.warning("Please paste some code first!")
