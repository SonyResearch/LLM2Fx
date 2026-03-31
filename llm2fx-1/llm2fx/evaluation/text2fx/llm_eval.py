import os
import pandas as pd
from datasets import load_dataset
import argparse
import random
import ast
import json
import numpy as np
from llm2fx.evaluation.metrics.dist_matching import compute_mmd, compute_similarity_stats
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--gt_embeds_dir", type=str)
parser.add_argument("--pred_embeds_dir", type=str)
parser.add_argument("--model_type", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()
INST_TYPES = ['drums', 'guitar', 'piano']

def load_llm_embedding(model_type="mistral_7b", inst_type="drums", fx_type="eq", text='bright'):
    embedding_dir = f"{args.pred_embeds_dir}/{model_type}/{fx_type}/{inst_type}/{text}/audio"
    if not os.path.exists(embedding_dir):
        return None
    embeddings, dev_ids = [], []
    for fname in os.listdir(embedding_dir):
        if fname.endswith(".npy"):
            embedding = np.load(os.path.join(embedding_dir, fname))
            embeddings.append(embedding)
            dev_ids.append(fname.split(".")[0])
    embeddings = np.stack(embeddings, axis=0)
    return embeddings, dev_ids

def eval_results(fx_type, inst_type, text, eval_set):
    eval_emb = [np.load(f"{args.gt_embeds_dir}/{fx_type}/{_id}/{inst_type}.npy") for _id in eval_set]
    eval_emb = np.stack(eval_emb, axis=0)
    dev_emb, dev_ids= load_llm_embedding(model_type=args.model_type, inst_type=inst_type, fx_type=fx_type, text=text)
    if eval_emb.ndim == 3:
        eval_emb = eval_emb.mean(axis=1)
    if dev_emb.ndim == 3:
        dev_emb = dev_emb.mean(axis=1)
    mmd_score = compute_mmd(dev_emb, eval_emb)
    n_samples = len(eval_set)
    return mmd_score, n_samples, dev_ids
def main():
    eval_db = load_dataset("seungheondoh/socialfx-gen-eval")
    results = []
    for fx_type in ['eq', 'reverb']:
        df_eval = pd.DataFrame(eval_db[fx_type])
        for _, row in df_eval.iterrows():
            for inst_type in INST_TYPES:
                text = row["input"]
                eval_set = row['output']
                mmd_score, n_samples, dev_ids = eval_results(fx_type, inst_type, text, eval_set)
                os.makedirs(f"{args.pred_embeds_dir}/{fx_type}/{inst_type}", exist_ok=True)
                with open(f"{args.pred_embeds_dir}/{fx_type}/{inst_type}/{text}.json", "w") as f:
                    json.dump({"fx_type": fx_type, "inst_type": inst_type, "text":text, "dev_ids": dev_ids}, f, indent=4)
                results.append({
                    "fx_type": fx_type,
                    "text": text,
                    "inst_type": inst_type,
                    "mmd_score": mmd_score,
                    "n_samples": n_samples
                })
    df_results = pd.DataFrame(results)
    os.makedirs(args.save_dir, exist_ok=True)
    df_results.to_csv(f"{args.save_dir}/{args.model_type}.csv", index=False)

if __name__ == "__main__":
    main()
