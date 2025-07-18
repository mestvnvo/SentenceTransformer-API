import gradio as gr
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(text):
    if not text.strip():
        return "No text provided."
    embedding = model.encode(text)
    return embedding.tolist()

# api w/ gradio
api = gr.Interface(
    fn=embed,
    inputs=gr.Textbox(label="Enter Text"),
    outputs=gr.JSON(label="embedding vector")
)

api.launch(show_api=True)
