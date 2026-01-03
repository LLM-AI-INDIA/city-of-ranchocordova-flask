"""
Unified Chatbot for Rancho Cordova
===================================

Handles both Energy Agent and Customer Service Agent
with integrated visualization support.
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_loader import get_data_loader

# Import visualization module
from .visualizations import generate_visualization

# Globals
_llm = None
_embedder = None
_chunks = None
_chunk_embeddings = None
_energy_df = None
_cs_df = None
_dept_df = None


def initialize_models():
    """Load LLM, embedder, KB and dataframes once."""
    print("##### CALLING initialize_models()\n")
    global _llm, _embedder, _chunks, _chunk_embeddings
    global _energy_df, _cs_df, _dept_df

    if _llm is not None:
        return

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    print("Loading Rancho Cordova models...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    _llm = (model, tokenizer)
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    base_path = os.path.join(os.path.dirname(__file__), "data")
    _energy_df = pd.read_csv(os.path.join(base_path, "Energy.txt"))
    _cs_df = pd.read_csv(os.path.join(base_path, "CustomerService.txt"))
    _dept_df = pd.read_csv(
        os.path.join(base_path, "Department-city of Rancho Cordova.txt")
    )

    print("Loading enhanced energy datasets...")
    loader = get_data_loader()
    print("✅ Enhanced datasets loaded")

    # Build chunk KB
    _chunks = []

    # ENERGY TABLE
    for _, row in _energy_df.iterrows():
        _chunks.append(
            f"ENERGY_RECORD | "
            f"CustomerID={row['CustomerID']} | "
            f"AccountType={row['AccountType']} | "
            f"Month={row['Month']} | "
            f"EnergyConsumption_kWh={row['EnergyConsumption_kWh']}"
        )

    # CUSTOMER SERVICE
    for _, row in _cs_df.iterrows():
        text_row = " | ".join([f"{col}={row[col]}" for col in _cs_df.columns])
        _chunks.append(f"CS_RECORD | {text_row}")

    # DEPARTMENTS
    for _, row in _dept_df.iterrows():
        text_row = " | ".join([f"{col}={row[col]}" for col in _dept_df.columns])
        _chunks.append(f"DEPT_RECORD | {text_row}")

    _chunk_embeddings = _embedder.encode(_chunks, convert_to_numpy=True)

    print("✅ Rancho models initialized.")


def retrieve(query, top_k=12):
    """Retrieve most relevant chunks for the query."""
    print("##### CALLING retrieve()")
    global _embedder, _chunks, _chunk_embeddings

    if _embedder is None:
        initialize_models()

    q_emb = _embedder.encode([query], convert_to_numpy=True)[0]

    scores = (
        _chunk_embeddings
        @ q_emb
        / (np.linalg.norm(_chunk_embeddings, axis=1) * np.linalg.norm(q_emb))
    )

    top_idx = np.argsort(scores)[::-1][:top_k]

    retrieved_lines = [
        line for i, line in enumerate(_chunks) if i in top_idx and line.strip()
    ]

    return "\n".join(retrieved_lines)


def detect_agent_type(query, retrieved_text):
    """
    Detect which agent type (energy/customer_service) is most relevant.
    """
    # Check retrieved chunks
    if "ENERGY_RECORD" in retrieved_text:
        return "energy"
    elif "CS_RECORD" in retrieved_text:
        return "customer_service"

    # Fallback to query keywords
    query_lower = query.lower()
    energy_kw = [
        "energy",
        "consumption",
        "kwh",
        "residential",
        "commercial",
        "electricity",
    ]
    cs_kw = ["call", "customer", "service", "ticket", "complaint", "billing question"]

    if any(kw in query_lower for kw in energy_kw):
        return "energy"
    elif any(kw in query_lower for kw in cs_kw):
        return "customer_service"

    return "energy"  # Default


SYSTEM_PROMPT = """
You are the official chatbot for the City of Rancho Cordova.

You answer questions ONLY using the information found in the city logs below.
However, you MAY provide light interpretation, summaries, or clarifications—
as long as they are clearly grounded in the logs.

Very important rules:
- You MUST reference the city records directly in your answer.
- You MAY combine or summarize related log entries.
- You MAY generalize slightly if it helps clarity.
- DO NOT invent specific policies, numeric data, or programs that are not in the logs.
- If the logs do not contain the answer, say:
  "I don't have that information in the city records."

When showing visualizations, briefly describe what the chart shows.

Your tone: concise, helpful, professional.
"""


def generate_answer(query, agent_type=None):
    """
    Generate answer with optional visualization data.

    Args:
        query: User's query
        agent_type: 'energy' or 'customer_service' (optional, will auto-detect)

    Returns:
        {
            'answer': str,
            'retrieved_text': str,
            'visualization': dict or None,
            'agent_type': str
        }
    """
    print("##### CALLING generate_answer()")

    if _llm is None:
        initialize_models()

    model, tokenizer = _llm

    # Retrieve relevant context
    retrieved = retrieve(query)

    # Auto-detect agent type if not provided
    if not agent_type:
        agent_type = detect_agent_type(query, retrieved)

    print(f"Detected agent type: {agent_type}")

    # Generate visualization if needed
    visualization = None
    if agent_type == "energy":
        visualization = generate_visualization(query, "energy", _energy_df)
    elif agent_type == "customer_service":
        visualization = generate_visualization(query, "customer_service", _cs_df)

    # Generate text answer
    prompt = f"""
<s>{SYSTEM_PROMPT}</s>
<CONTEXT>{retrieved}</CONTEXT>
<USER>{query}</USER>
<ASSISTANT>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return {
        "answer": answer.strip(),
        "retrieved_text": retrieved,
        "visualization": visualization,
        "agent_type": agent_type,
    }
