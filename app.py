from sentence_transformers import SentenceTransformer, util
import gradio as gr

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def compute_similarity(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    score = max(0.0, min(1.0, (similarity + 1) / 2))
    return {"similarity score": round(score, 4)}

iface = gr.Interface(
    fn=compute_similarity,
    inputs=[gr.Textbox(label="Text 1"), gr.Textbox(label="Text 2")],
    outputs="json",
    title="Text Similarity Checker"
)

iface.launch()
