# Semantic Textual Similarity API

This project provides a semantic similarity score between two English paragraphs using a pretrained transformer model. It was developed as part of the DataNeuronAI assignment.

The model returns a score between 0 and 1, where 1 means the texts are semantically very similar, and 0 means they are unrelated.

## How it works

- Uses `sentence-transformers/all-MiniLM-L6-v2` for generating sentence embeddings.
- Computes cosine similarity between the embeddings.
- Normalizes the score to fit within the [0, 1] range.
- Built using Python and deployed via Gradio on Hugging Face Spaces.

## Example

**Input:**
```json
{
  "text1": "nuclear body seeks new tech",
  "text2": "terror suspects face arrest"
}
```

Live model at: [Link](https://huggingface.co/spaces/Karthix1/STS)
