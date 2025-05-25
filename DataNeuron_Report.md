
# Part A: Semantic Similarity Model Development

**Candidate:** Karthik Murali M  
**Assignment:** DataNeuronAI – Semantic Similarity API  

---

## 🧠 1. Problem Overview

The goal was to build a model that predicts how semantically similar two paragraphs are, on a scale from **0 (dissimilar)** to **1 (highly similar)**. This is a classical **Semantic Textual Similarity (STS)** task, commonly used in NLP for duplicate detection, question matching, and more.

---

## 📊 2. Dataset Exploration and Analysis

### ✅ What Was Done:
- Loaded dataset using `pandas`
- Inspected for nulls, structure, data types
- Analyzed character and word lengths in `text1` and `text2`
- Plotted histograms for paragraph length distributions
- Counted common tokens, bigrams, and trigrams

### 🔍 Why:
To check:
- Input sizes (important for transformer model limits)
- Biases or imbalance between `text1` and `text2`
- Vocabulary richness
- Repeated phrases or boilerplate text

### 📈 Insights:
- Paragraphs had high length variance (some up to 4000+ words)
- Mean, median, and distribution were consistent across both text columns
- Vocabulary aligned with a **news** or **informative article** corpus
- Top words: "government", "minister", "police", etc.

---

## 🤖 3. Preprocessing

Basic text cleaning was applied:
- Lowercasing
- Removal of URLs, emails, special characters
- Trimming whitespace

This ensured consistency during embedding generation.

---

## 🧠 4. Model Selection

### 🔍 Chosen Model:
- `sentence-transformers/all-MiniLM-L6-v2`

### ✅ Why This Model?
- Light and fast (~80MB)
- Pretrained for sentence-level semantic tasks
- Outperforms vanilla BERT on similarity benchmarks
- Handles long text (~512 tokens per segment)

---

## 📐 5. Similarity Scoring Method

### ➤ How it Works:
1. Convert both paragraphs to embeddings
2. Compute cosine similarity using:
```python
cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()
```
3. Normalize:
```python
final_score = (cosine_score + 1) / 2
```

### ✅ Why Normalize?
- Cosine similarity returns values in [-1, 1]
- The task expects values in [0, 1] range

---

## 🧪 6. Testing & Output

The score is:
- 0.0 for completely unrelated paragraphs
- 1.0 for highly similar or paraphrased content

### 🧾 Example Output:
```json
{
  "text1": "The Prime Minister visited London...",
  "text2": "The head of state arrived in the UK capital...",
  "similarity score": 0.83
}
```

---

## ✅ Summary

- Performed thorough EDA to validate dataset quality
- Used SBERT (`MiniLM-L6-v2`) for robust embeddings
- Calculated similarity via cosine distance
- Ensured score fits [0, 1] range for submission format

---

# Part B: Model Deployment

## 🛠 Objective

Deploy the semantic similarity model built in Part A as a **live API** that:
- Accepts `text1` and `text2` as JSON input
- Returns a similarity score between 0 and 1 in this format:
```json
{ "similarity score": 0.87 }
```

---

## 🔧 Tools Used

| Component        | Technology               |
|------------------|--------------------------|
| Deployment       | Hugging Face Spaces      |
| App framework    | Gradio                   |
| Model framework  | sentence-transformers    |
| Base model       | all-MiniLM-L6-v2         |
| Hosting cost     | $0 – 100% free tier      |

---

## 🧱 Project Structure

```
app.py               # Main Gradio app with inference logic
requirements.txt     # All dependencies for Hugging Face to install
```

---

## 📜 `app.py`

```python
from sentence_transformers import SentenceTransformer, util
import gradio as gr

# Load the SBERT model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define the similarity scoring function
def compute_similarity(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    score = max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to [0, 1]
    return {"similarity score": round(score, 4)}

# Set up Gradio UI
iface = gr.Interface(
    fn=compute_similarity,
    inputs=[gr.Textbox(label="Text 1"), gr.Textbox(label="Text 2")],
    outputs="json",
    title="Text Similarity Checker"
)

# Launch the app
iface.launch()
```

---

## 📦 `requirements.txt`

```txt
gradio
sentence-transformers
torch
```

---

## 🧠 Why Gradio + Hugging Face?

- ✅ **FastAPI was too heavy** for free hosts like Vercel or Replit due to `torch` dependencies.
- ✅ Gradio + Hugging Face Spaces deploys with **no Docker**, **no billing**, and **no CLI config**.
- ✅ Hugging Face provides **free hosting** and GPU (if needed) for smaller models.

---

## 🌍 Live URL

**Access the app here:**  
🔗 https://huggingface.co/spaces/Karthix1/STS

---

## 🧪 How to Test It

1. Visit the link above.
2. Enter any two English paragraphs into the input fields.
3. Click "Submit".
4. The API returns a JSON response like:
```json
{ "similarity score": 0.8125 }
```

This exactly matches the required format for the assignment.

---

## 🚫 Limitations

- Gradio UI is not a REST endpoint (but still testable interactively).
- Cold start latency: 3–5 seconds on first load.
- `torch` is still heavy even with a small model.
- Suitable for single-pair similarity — not optimized for bulk or real-time inference.

---

## ✅ Summary

Using Gradio and Hugging Face Spaces allowed me to:

- Deploy the model with zero infrastructure cost
- Deliver a testable API as required by the assignment
- Avoid deployment blockers like torch + Docker + GCP permission hell

It's a clean, reliable solution for demonstrating model capability in a production-like setting.
---
