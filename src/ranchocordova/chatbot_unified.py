"""
Unified Chatbot for Rancho Cordova - FIXED FULLY DYNAMIC RAG
=============================================================

FIXES:
1. Hourly data parsing (handles wide-format CSV correctly)
2. Customer service detection (water bill, pothole, etc.)
3. Better agent type detection
4. Simplified energy tips (no reliance on bad CSV format)
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

    # ENERGY TABLE (original)
    for _, row in _energy_df.iterrows():
        _chunks.append(
            f"ENERGY_RECORD | "
            f"CustomerID={row['CustomerID']} | "
            f"AccountType={row['AccountType']} | "
            f"Month={row['Month']} | "
            f"EnergyConsumption_kWh={row['EnergyConsumption_kWh']}"
        )

    # CUSTOMER SERVICE (original)
    for _, row in _cs_df.iterrows():
        text_row = " | ".join([f"{col}={row[col]}" for col in _cs_df.columns])
        _chunks.append(f"CS_RECORD | {text_row}")

    # DEPARTMENTS (original)
    for _, row in _dept_df.iterrows():
        text_row = " | ".join([f"{col}={row[col]}" for col in _dept_df.columns])
        _chunks.append(f"DEPT_RECORD | {text_row}")

    # ========================================================================
    # DYNAMIC: Extract knowledge from actual CSV files
    # ========================================================================

    _chunks.extend(_create_energy_saving_tips())  # FIXED - No CSV parsing errors
    _chunks.extend(_extract_benchmark_insights(base_path))
    _chunks.extend(_extract_tou_rate_insights(base_path))
    _chunks.extend(_extract_rebate_insights(base_path))

    print(f"✅ Total RAG chunks: {len(_chunks)}")

    _chunk_embeddings = _embedder.encode(_chunks, convert_to_numpy=True)

    print("✅ Rancho models initialized.")


# ============================================================================
# FIXED: Energy-Saving Tips (No complex CSV parsing)
# ============================================================================


def _create_energy_saving_tips() -> list:
    """
    Create comprehensive energy-saving tips chunks.
    Based on industry standards, not broken CSV parsing.
    """
    chunks = []

    # General energy-saving advice
    chunks.append(
        "ENERGY_SAVING_TIPS | "
        "Question=How_to_lower_electric_bill | "
        "Answer=To lower your electric bill: 1) Shift major appliance usage to off-peak hours "
        "(before 4pm or after 9pm weekdays), 2) Upgrade to ENERGY STAR appliances, "
        "3) Set thermostat to 78°F in summer and 68°F in winter, 4) Switch to LED light bulbs, "
        "5) Seal air leaks around windows and doors, 6) Use ceiling fans to circulate air, "
        "7) Run dishwasher and laundry with full loads only, 8) Unplug devices when not in use, "
        "9) Apply for SMUD rebate programs for qualifying upgrades."
    )

    chunks.append(
        "ENERGY_SAVING_TIPS | "
        "Question=How_to_save_energy_at_home | "
        "Answer=Save energy at home by: using programmable or smart thermostats, washing clothes "
        "in cold water, air-drying dishes instead of heat dry, turning off lights when leaving rooms, "
        "unplugging chargers and electronics when not in use, using power strips to eliminate phantom loads, "
        "closing blinds during hot afternoons, weatherstripping doors and windows, servicing HVAC systems "
        "regularly, and replacing old appliances with ENERGY STAR certified models."
    )

    chunks.append(
        "ENERGY_SAVING_TIPS | "
        "Question=Best_times_to_run_appliances | "
        "Answer=Best times to run major appliances: Run dishwasher, washing machine, dryer, and pool pump "
        "during off-peak hours - before 4pm or after 9pm on weekdays, or anytime on weekends. "
        "Peak hours (4-9pm weekdays) have the highest electricity rates. Shifting usage to off-peak "
        "can save 30-50% on energy costs for these appliances."
    )

    chunks.append(
        "ENERGY_SAVING_TIPS | "
        "Question=Best_time_to_run_dishwasher | "
        "Answer=Best time to run dishwasher is during off-peak hours: before 4pm or after 9pm on weekdays, "
        "or anytime on weekends. Also: run only when full, use air-dry setting instead of heat dry, "
        "and scrape (don't rinse) dishes before loading. This can save significant energy and water."
    )

    chunks.append(
        "APPLIANCE_ENERGY_INFO | "
        "Topic=Appliances_that_use_most_energy | "
        "Answer=The appliances that typically use the most energy in homes are: "
        "1) HVAC systems (heating/cooling) - about 40-50% of home energy use, "
        "2) Water heater - about 15-20%, "
        "3) Washer and dryer - about 10-15%, "
        "4) Lighting - about 8-12%, "
        "5) Refrigerator - about 5-8%, "
        "6) Electric oven and stove - about 3-5%, "
        "7) Dishwasher - about 1-2%, "
        "8) TV and electronics - about 3-5%. "
        "Pool pumps, if present, can use 15-20% of home energy."
    )

    chunks.append(
        "TIME_OF_USE_INFO | "
        "Topic=Peak_and_Off_Peak_Hours | "
        "Peak_Hours=4pm-9pm weekdays | "
        "Off_Peak_Hours=Before 4pm and after 9pm weekdays, all day weekends | "
        "Description=SMUD uses time-of-use rates. Peak hours (4-9pm weekdays) have the highest rates. "
        "Off-peak hours offer significantly lower rates. Plan major energy usage during off-peak times to save money."
    )

    print(f"  ✓ Created {len(chunks)} energy-saving tip chunks")
    return chunks


# ============================================================================
# DYNAMIC CSV EXTRACTION (Benchmark, TOU, Rebates)
# ============================================================================


def _extract_benchmark_insights(base_path: str) -> list:
    """Dynamically extract utility comparison insights from CSV."""
    chunks = []
    csv_path = os.path.join(base_path, "CA_Benchmarks.csv")

    if not os.path.exists(csv_path):
        print("  ⚠️  CA_Benchmarks.csv not found")
        return chunks

    try:
        df = pd.read_csv(csv_path)

        # Create a chunk for each utility/home type combination
        for _, row in df.iterrows():
            chunks.append(
                f"UTILITY_COMPARISON | "
                f"Utility={row['Utility_or_CCA']} | "
                f"Type={row.get('Utility_Type', 'N/A')} | "
                f"Home_Type={row['Home_Type']} | "
                f"Avg_Monthly_kWh={row['Avg_Monthly_Usage_kWh']} | "
                f"Avg_Annual_kWh={row['Avg_Annual_Usage_kWh']} | "
                f"Rate_per_kWh=${row['Avg_Rate_usd_per_kWh']} | "
                f"Avg_Monthly_Bill=${row['Est_Avg_Monthly_Bill_usd']}"
            )

        # Create comparison insights (SMUD vs others)
        smud_data = df[df["Utility_or_CCA"] == "SMUD"]
        if not smud_data.empty:
            smud_avg_rate = smud_data["Avg_Rate_usd_per_kWh"].mean()

            # Compare with PG&E
            pge_data = df[df["Utility_or_CCA"] == "PG&E"]
            if not pge_data.empty:
                pge_avg_rate = pge_data["Avg_Rate_usd_per_kWh"].mean()
                savings_pct = (pge_avg_rate - smud_avg_rate) / pge_avg_rate * 100

                chunks.append(
                    f"UTILITY_SAVINGS | "
                    f"Comparison=SMUD_vs_PGE | "
                    f"SMUD_Rate=${smud_avg_rate:.3f}/kWh | "
                    f"PGE_Rate=${pge_avg_rate:.3f}/kWh | "
                    f"Savings={savings_pct:.0f}% | "
                    f"Description=SMUD residential customers save approximately {savings_pct:.0f}% on electricity "
                    f"rates compared to PG&E. SMUD's average rate is ${smud_avg_rate:.2f}/kWh vs PG&E's ${pge_avg_rate:.2f}/kWh."
                )

        print(f"  ✓ Extracted {len(chunks)} benchmark insights")

    except Exception as e:
        print(f"  ⚠️  Error processing benchmarks: {e}")

    return chunks


def _extract_tou_rate_insights(base_path: str) -> list:
    """Dynamically extract TOU rate information from CSV."""
    chunks = []
    csv_path = os.path.join(base_path, "SMUD_TOU_Rates.csv")

    if not os.path.exists(csv_path):
        print("  ⚠️  SMUD_TOU_Rates.csv not found, using defaults")
        # Add default TOU info
        chunks.append(
            "TOU_RATES_INFO | "
            "Peak_Period=4pm-9pm weekdays | Peak_Rate=Higher | "
            "Off_Peak_Period=All other times | Off_Peak_Rate=Lower | "
            "Description=SMUD offers time-of-use rates with peak pricing 4-9pm weekdays. "
            "Use electricity during off-peak hours to save money."
        )
        return chunks

    try:
        df = pd.read_csv(csv_path)

        # Create a chunk for each rate period
        for _, row in df.iterrows():
            chunk_parts = [f"{col}={row[col]}" for col in df.columns]
            chunks.append(f"TOU_RATES | {' | '.join(chunk_parts)}")

        print(f"  ✓ Extracted {len(chunks)} TOU rate insights")

    except Exception as e:
        print(f"  ⚠️  Error processing TOU rates: {e}")

    return chunks


def _extract_rebate_insights(base_path: str) -> list:
    """Dynamically extract rebate program information from CSV."""
    chunks = []
    csv_path = os.path.join(base_path, "SMUD_Rebates.csv")

    if not os.path.exists(csv_path):
        print("  ⚠️  SMUD_Rebates.csv not found, using defaults")
        # Add default rebate info
        chunks.append(
            "REBATE_PROGRAMS | "
            "Question=Energy_rebate_programs_Rancho_Cordova | "
            "Answer=Yes! Rancho Cordova residents served by SMUD qualify for energy rebate programs including: "
            "HVAC system upgrades, pool pump replacements, appliance rebates for ENERGY STAR certified products, "
            "LED lighting, smart thermostats, weatherization improvements, and solar panel installations. "
            "Contact SMUD at 1-888-742-7683 or visit smud.org/rebates for current programs and eligibility."
        )
        return chunks

    try:
        df = pd.read_csv(csv_path)

        # Create a chunk for each rebate program
        for _, row in df.iterrows():
            chunk_parts = [f"{col}={row[col]}" for col in df.columns]
            chunks.append(f"REBATE_PROGRAM | {' | '.join(chunk_parts)}")

        # Create summary
        if "Rebate_Amount" in df.columns and "Program_Name" in df.columns:
            total_programs = len(df)
            programs_list = ", ".join(df["Program_Name"].head(5).tolist())

            chunks.append(
                f"REBATE_SUMMARY | "
                f"Total_Programs={total_programs} | "
                f"Programs_Available={programs_list} | "
                f"Description=SMUD offers {total_programs} rebate programs for Rancho Cordova residents. "
                f"Visit smud.org/rebates or call 1-888-742-7683 for details."
            )

        print(f"  ✓ Extracted {len(chunks)} rebate insights")

    except Exception as e:
        print(f"  ⚠️  Error processing rebates: {e}")

    return chunks


# ============================================================================
# FIXED: Better Agent Type Detection
# ============================================================================


def detect_agent_type(query, retrieved_text):
    """
    FIXED: Better detection of energy vs customer service queries.
    """
    query_lower = query.lower()

    # CUSTOMER SERVICE keywords (check FIRST, higher priority)
    cs_keywords = [
        "water bill",
        "trash",
        "garbage",
        "pothole",
        "streetlight",
        "street light",
        "permit",
        "building",
        "fence",
        "construction",
        "city hall",
        "department",
        "who do i call",
        "who do i contact",
        "report a",
        "pay my",
        "payment",
        "city services",
        "public works",
        "planning",
        "zoning",
    ]

    # Check query for customer service keywords FIRST
    if any(kw in query_lower for kw in cs_keywords):
        print(
            f"  → Detected CS keywords in query: {[kw for kw in cs_keywords if kw in query_lower]}"
        )
        return "customer_service"

    # Check retrieved chunks for customer service content
    if "DEPT_RECORD" in retrieved_text or "CS_RECORD" in retrieved_text:
        print(f"  → Detected CS chunks in retrieved text")
        return "customer_service"

    # ENERGY keywords
    energy_keywords = [
        "energy",
        "electricity",
        "kwh",
        "bill",
        "consumption",
        "appliance",
        "hvac",
        "rebate",
        "save energy",
        "lower",
        "peak",
        "off-peak",
        "rate",
        "utility",
        "dishwasher",
        "dryer",
        "air conditioning",
    ]

    # Check for energy indicators
    if any(kw in query_lower for kw in energy_keywords):
        print(
            f"  → Detected energy keywords: {[kw for kw in energy_keywords if kw in query_lower]}"
        )
        return "energy"

    if any(
        indicator in retrieved_text
        for indicator in ["ENERGY", "APPLIANCE", "REBATE", "UTILITY"]
    ):
        print(f"  → Detected energy chunks in retrieved text")
        return "energy"

    # Default to customer_service if ambiguous (safer for city queries)
    print(f"  → Defaulting to customer_service")
    return "customer_service"


# ============================================================================
# Rest of code (unchanged)
# ============================================================================


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
    """Generate answer with optional visualization data."""
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
            max_new_tokens=200,
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
