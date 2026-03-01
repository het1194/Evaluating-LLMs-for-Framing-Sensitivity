import pandas as pd 
import numpy as np 
from math import log2 
from collections import Counter 
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity

#clean
def load_and_normalize(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding="cp1252", dtype={"sentence_id": str})
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1", dtype={"sentence_id": str})
    df.columns = df.columns.str.strip()
    df["item_id"] = df["sentence_id"].apply(lambda x: int(x.split(".")[0]))
    df["frame_id"] = df["sentence_id"].apply(lambda x: int(x.split(".")[1]) if "." in x else 0)

    records = []
    model_map = {
        "llama": ("rating of llama", "llama reasoning"),
        "qwen": ("rating of qwen", "response by qwen"),
        "gpt_oss": ("rating of gpt_oss", "response by gpt_oss"),
        "mistral": ("rating by mistral", "response by mistral")
    }

    for _, row in df.iterrows():
        for model, (r_col, t_col) in model_map.items():
            val = row[r_col]
            if pd.notna(val):
                try:
                    rating = int(val)
                except ValueError:
                    continue
                records.append({
                    "item_id": row["item_id"],
                    "frame_id": row["frame_id"],
                    "model_id": model,
                    "rating": rating,
                    "rationale": str(row[t_col])
                })
    return pd.DataFrame(records)

#all metrcs
def polarity(r):
    if r <= 2: return "D"
    if r == 3: return "N"
    return "A"

def entropy(counts):
    total = sum(counts.values())
    return -sum((c/total) * log2(c/total) for c in counts.values())

def drift_magnitude(df):
    drift_df = df.groupby(["model_id", "item_id"])["rating"].apply(lambda g: g.max() - g.min()).reset_index(name="drift")
    return drift_df, drift_df.groupby("model_id")["drift"].mean()

def frame_sensitivity(df):
    rows = []
    for (model, item), g in df.groupby(["model_id", "item_id"]):
        baseline = g["rating"].mean()
        for _, r in g.iterrows():
            rows.append({"model_id": model, "frame_id": r["frame_id"], "deviation": abs(r["rating"] - baseline)})
    fs_df = pd.DataFrame(rows)
    return fs_df, fs_df.groupby(["model_id", "frame_id"])["deviation"].mean()

def polarity_flip_count(df):
    flip_df = df.groupby(["model_id", "item_id"])["rating"].apply(
        lambda g: int("D" in set(g.apply(polarity)) and "A" in set(g.apply(polarity)))
    ).reset_index(name="flip")
    return flip_df, flip_df.groupby("model_id")["flip"].mean()

def category_entropy(df):
    ent_df = df.groupby(["model_id", "item_id"])["rating"].apply(
        lambda g: entropy(Counter(g.apply(polarity)))
    ).reset_index(name="entropy")
    return ent_df, ent_df.groupby("model_id")["entropy"].mean()

def stability_score(avg_drift, avg_flip, avg_entropy):
    drift_norm = avg_drift / 4.0
    flip_norm = avg_flip
    entropy_norm = avg_entropy / log2(3)
    instability = np.mean([drift_norm, flip_norm, entropy_norm], axis=0)
    return 1 - instability

def compute_intra_model_mean(df):
    results = []
    for model_id, model_data in df.groupby("model_id"):
        sims = []
        for item_id, item_data in model_data.groupby("item_id"):
            embs = np.vstack(item_data["embedding"].values)
            if len(embs) > 1:
                sim_matrix = cosine_similarity(embs)
                ui = np.triu_indices(len(embs), k=1)
                sims.extend(sim_matrix[ui])
        results.append({"model_id": model_id, "mean_intra_sim": np.mean(sims)})
    return pd.DataFrame(results).set_index("model_id")

def compute_inter_model_mean(df):
    models = sorted(df["model_id"].unique())
    pair_sims = {m1: {m2: [] for m2 in models} for m1 in models}

    for (item_id, frame_id), prompt_data in df.groupby(["item_id", "frame_id"]):
        prompt_embs = {row["model_id"]: row["embedding"] for _, row in prompt_data.iterrows()}
        for m1 in models:
            for m2 in models:
                if m1 in prompt_embs and m2 in prompt_embs:
                    sim = cosine_similarity([prompt_embs[m1]], [prompt_embs[m2]])[0][0]
                    pair_sims[m1][m2].append(sim)

    mean_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for m1 in models:
        for m2 in models:
            mean_matrix.loc[m1, m2] = np.mean(pair_sims[m1][m2])
    return mean_matrix

#main
def print_metrics(df):
    drift_df, avg_drift = drift_magnitude(df)
    flip_df, avg_flip = polarity_flip_count(df)
    ent_df, avg_entropy = category_entropy(df)
    fs_df, frame_profile = frame_sensitivity(df)
    stability = stability_score(avg_drift, avg_flip, avg_entropy)

    print("\nDrift Magnitude")
    print(avg_drift)
    print("\nPolarity Flip Rate")
    print(avg_flip)
    print("\nCategory Entropy")
    print(avg_entropy)
    print("\nFrame Sensitivity Profile")
    print(frame_profile)
    print("\nStability Score")
    print(stability)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    df["embedding"] = list(embedder.encode(df["rationale"].tolist(), show_progress_bar=True))
    intra_means = compute_intra_model_mean(df)
    print("\nIntra-Model Mean Similarity")
    print(intra_means)

    inter_means = compute_inter_model_mean(df)
    print("\nInter-Model Similarity (Mean)")
    print(inter_means)

if __name__ == "__main__":
    df = load_and_normalize("./output.csv")
    print_metrics(df)
