# Code Documentation Generator (Code-to-Text)

An end-to-end Transformer-based Generative AI project that automatically generates Python docstrings from source code.
The entire NLP pipeline — tokenization, Transformer architecture, training, inference, and UI — is built from scratch using PyTorch.

This project demonstrates how modern LLM-style systems work internally, without relying on pre-trained APIs or external language models.
## Project Overview

The application takes a Python function as input and generates a human-readable docstring describing its behavior.
It is designed as a learning-focused, engineering-complete GenAI system, covering:
Data preprocessing
Tokenization
Encoder–Decoder Transformer design
Model training with checkpoints
Inference and decoding
Interactive UI deployment

## Features
- End-to-end Transformer from scratch (no prebuilt models)
- Real dataset training for code-to-docstring generation (NLP)
- Tokenization with Byte-Pair Encoding (BPE) + ByteLevel decoding
- Encoder–Decoder attention with positional encoding and causal masking
- Inference with top-k sampling for stable + varied output
- Interactive Streamlit UI for docstring generation
- Render-ready deployment configuration

## Folder Structure
```
Code Documentation Generator (Code-to-Text)/
├─ app.py                         # Streamlit UI
├─ model_def.py                   # Transformer model + generation logic
├─ checkpoint.pt                  # Model weights (checkpoint)
├─ code_doc_tokenizer.json        # Tokenizer
├─ requirements.txt               # Python dependencies
├─ Procfile                       # Render start command
├─ render.yaml                    # Render service config
├─ .streamlit/
│  └─ config.toml                 # Streamlit theme/config
├─ Code_Documentation_Generator_(Code_to_Text).ipynb
├─ __pycache__/
└─ .gitignore
```

## How It Works
1. You paste Python code in the Streamlit UI.
2. The app prepends a prompt requesting a docstring.
3. The input is tokenized using BPE + ByteLevel decoding.
4. The encoder–decoder Transformer processes the input with attention, positional encoding, and masking.
5. The decoder generates the docstring using top-k sampling.
6. The result is displayed in the app.

## Transformer + NLP Details
- Implemented a Transformer architecture from scratch in PyTorch.
- Encoder–Decoder attention blocks with feed-forward layers and layer normalization.
- Positional encoding for sequence order awareness.
- Causal masking in the decoder to prevent peeking at future tokens.
- Trained on a real-world code–documentation dataset with checkpointing.
- Tokenization via Byte-Pair Encoding (BPE), decoded with ByteLevel.

## Tech Stack
- Python, PyTorch
- Transformers (attention, masking, embeddings)
- Tokenizers (BPE / ByteLevel)
- Streamlit (UI)

## Local Setup
1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the Streamlit app.

## Run Locally
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Render)
This project is Render-ready.
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

You can deploy using either the `Procfile` or `render.yaml`.

## Notes
- Ensure `checkpoint.pt` and `code_doc_tokenizer.json` are present in the project root.
- The model loads weights from `checkpoint.pt` (or `final_model.pt` if present).

## License
This project is for educational/demo purposes.