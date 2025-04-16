import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
import openai
import os
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="🧾 Reconciliation Dashboard", layout="wide")
st.title("🔗 Accounting Reconciliation Agent")

# Sidebar config
st.sidebar.header("Upload Data")
bank_file = st.sidebar.file_uploader("Upload Bank CSV", type=["csv"])
ledger_file = st.sidebar.file_uploader("Upload Ledger CSV", type=["csv"])

st.sidebar.header("Matching Thresholds")
vendor_thresh = st.sidebar.slider("Vendor Similarity (Fuzzy)", 0, 100, 70)
amount_thresh = st.sidebar.number_input("Max Amount Difference", value=100.0)
date_thresh = st.sidebar.slider("Max Date Difference (days)", 0, 30, 5)
use_llm = st.sidebar.checkbox("Use LLM Verification", value=False)

# Load and display data
def load_data():
    bank_df = pd.read_csv(bank_file)
    ledger_df = pd.read_csv(ledger_file)
    return bank_df, ledger_df

# Build graph
@st.cache_data
def build_graph(bank_df, ledger_df):
    G = nx.Graph()
    for i, row in bank_df.iterrows():
        G.add_node(f"bank_{i}", **row.to_dict(), source="bank")
    for j, row in ledger_df.iterrows():
        G.add_node(f"ledger_{j}", **row.to_dict(), source="ledger")

    for i, b_row in bank_df.iterrows():
        for j, l_row in ledger_df.iterrows():
            sim = fuzz.ratio(str(b_row['Vendor']), str(l_row['Vendor']))
            amt_diff = abs(b_row['Amount'] - l_row['Amount'])
            date_diff = abs((pd.to_datetime(b_row['Date']) - pd.to_datetime(l_row['Date'])).days)
            if sim >= vendor_thresh and amt_diff <= amount_thresh and date_diff <= date_thresh:
                G.add_edge(f"bank_{i}", f"ledger_{j}", weight=sim)
    return G

# LLM verification
@st.cache_data(show_spinner=False)
def verify_with_llm(bank, ledger):
    prompt = f"""
You are reconciling financial records. Determine if these represent the same transaction. Answer only Yes or No.
Bank: Vendor={bank['Vendor']}, Amount={bank['Amount']}, Date={bank['Date']}
Ledger: Vendor={ledger['Vendor']}, Amount={ledger['Amount']}, Date={ledger['Date']}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a finance assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return "yes" in response.choices[0].message.content.lower()

# Graph rendering
def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    bank_nodes = [n for n, d in G.nodes(data=True) if d.get('source') == 'bank']
    ledger_nodes = [n for n, d in G.nodes(data=True) if d.get('source') == 'ledger']
    nx.draw_networkx_nodes(G, pos, nodelist=bank_nodes, node_color='skyblue', label='Bank')
    nx.draw_networkx_nodes(G, pos, nodelist=ledger_nodes, node_color='lightgreen', label='Ledger')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=7)
    ax.axis('off')
    st.pyplot(fig)

if bank_file and ledger_file:
    bank_df, ledger_df = load_data()
    st.subheader("Bank Transactions")
    st.dataframe(bank_df)
    st.subheader("Ledger Transactions")
    st.dataframe(ledger_df)

    G = build_graph(bank_df, ledger_df)
    st.subheader("🔍 Matching Graph")
    draw_graph(G)

    matched = []
    unmatched = []

    for b_node in [n for n in G.nodes if n.startswith("bank_")]:
        neighbors = list(G.neighbors(b_node))
        if not neighbors:
            unmatched.append(b_node)
        else:
            verified = False
            for l_node in neighbors:
                if use_llm:
                    if verify_with_llm(G.nodes[b_node], G.nodes[l_node]):
                        matched.append((b_node, l_node))
                        verified = True
                        break
                else:
                    matched.append((b_node, l_node))
                    verified = True
                    break
            if not verified:
                unmatched.append(b_node)

    st.subheader("✅ Matched Pairs")
    matched_df = pd.DataFrame([{
        "Bank": b,
        "Ledger": l,
        "Bank Vendor": G.nodes[b]['Vendor'],
        "Ledger Vendor": G.nodes[l]['Vendor']
    } for b, l in matched])
    st.dataframe(matched_df)

    st.subheader("❌ Unmatched Bank Transactions")
    unmatched_df = pd.DataFrame([{
        "Bank": u,
        **G.nodes[u]
    } for u in unmatched])
    st.dataframe(unmatched_df)

    st.download_button("Download Matched CSV", matched_df.to_csv(index=False), "matched.csv")
    st.download_button("Download Unmatched CSV", unmatched_df.to_csv(index=False), "unmatched.csv")

    st.success(f"{len(matched)} transactions matched, {len(unmatched)} unmatched.")