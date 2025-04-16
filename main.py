import pandas as pd
import networkx as nx
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import openai
import os
import matplotlib.pyplot as plt

# Load .env file to get OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load data from CSV files
def load_data():
    bank_df = pd.read_csv('../data/bank.csv')
    ledger_df = pd.read_csv('../data/ledger.csv')
    return bank_df, ledger_df

# Extract entities and relations for the graph
def build_graph(bank_df, ledger_df):
    G = nx.Graph()
    for idx, row in bank_df.iterrows():
        node_id = f"bank_{idx}"
        G.add_node(node_id, type='bank', vendor=row['Vendor'], amount=row['Amount'], date=row['Date'])
    for idx, row in ledger_df.iterrows():
        node_id = f"ledger_{idx}"
        G.add_node(node_id, type='ledger', vendor=row['Vendor'], amount=row['Amount'], date=row['Date'])

    for b_idx, bank_row in bank_df.iterrows():
        for l_idx, ledger_row in ledger_df.iterrows():
            vendor_sim = fuzz.ratio(bank_row['Vendor'], ledger_row['Vendor'])
            amount_diff = abs(bank_row['Amount'] - ledger_row['Amount'])
            date_diff = abs(pd.to_datetime(bank_row['Date']) - pd.to_datetime(ledger_row['Date'])).days

            vendor_score = vendor_sim
            amount_score = max(0, 100 - (amount_diff / 50) * 100)
            date_score = max(0, 100 - (date_diff / 5) * 100)

            final_score = (vendor_score * 0.5) + (amount_score * 0.3) + (date_score * 0.2)

            if final_score >= 75:
                bank_node = f"bank_{b_idx}"
                ledger_node = f"ledger_{l_idx}"
                G.add_edge(bank_node, ledger_node, weight=final_score)
    return G

# Use graph to retrieve similar matches
def graph_rag_retriever(graph, bank_node):
    matches = []
    for node in graph.neighbors(bank_node):
        if graph[bank_node][node]['weight'] >= 75:
            matches.append(node)
    return matches

# Hallucination elimination using GPT
def hallucination_elimination(bank_node, matches):
    if not matches:
        return []

    prompt = f"""You are a reconciliation assistant. The following ledger nodes may match the bank transaction {bank_node}.\n\nLedger nodes: {matches}\n\nWhich ones look like real matches? Reply with a list of ledger node IDs that are likely valid. If none, return an empty list."""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial reconciliation expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    import ast
    try:
        verified = ast.literal_eval(response.choices[0].message.content)
        return verified if isinstance(verified, list) else []
    except Exception as e:
        print("⚠️ Error parsing GPT response:", e)
        return []

# Generate summary for unmatched transactions
def generate_summary(unmatched_transactions):
    summary = f"Summary of unmatched transactions:\n"
    for txn in unmatched_transactions:
        summary += f"Transaction {txn} needs manual review.\n"
    return summary

# Visualize graph connections
def visualize_graph(G):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(14, 10))
    bank_nodes = [n for n in G.nodes if n.startswith("bank_")]
    ledger_nodes = [n for n in G.nodes if n.startswith("ledger_")]
    nx.draw_networkx_nodes(G, pos, nodelist=bank_nodes, node_color="skyblue", label="Bank Transactions")
    nx.draw_networkx_nodes(G, pos, nodelist=ledger_nodes, node_color="lightgreen", label="Ledger Entries")
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.legend()
    plt.title("Transaction Matching Graph (Bank ↔️ Ledger)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    bank_df, ledger_df = load_data()
    G = build_graph(bank_df, ledger_df)
    unmatched_transactions = []
    final_matches = {}

    for idx, _ in bank_df.iterrows():
        bank_node = f"bank_{idx}"
        matches = graph_rag_retriever(G, bank_node)
        if not matches:
            unmatched_transactions.append(bank_node)
        else:
            verified = hallucination_elimination(bank_node, matches)
            if verified:
                final_matches[bank_node] = verified
            else:
                unmatched_transactions.append(bank_node)

    print("\n✅ Matched Transactions:")
    for bank, ledgers in final_matches.items():
        print(f"🔗 {bank} matched with {', '.join(ledgers)}")

    print("\n🚨 Unmatched Transactions:")
    for txn in unmatched_transactions:
        print(f"❌ {txn}")

    summary = generate_summary(unmatched_transactions)
    print("\n📄 Summary:")
    print(summary)

if __name__ == "__main__":
    main()