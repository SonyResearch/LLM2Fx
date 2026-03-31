import os
import pandas as pd
from datasets import load_dataset
import argparse
import random
import ast
import json
import numpy as np
from llm4mp.evaluation.metrics.dist_matching import compute_mmd, compute_similarity_stats
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--fx_emb_type", type=str, default="STITO")
args = parser.parse_args()

EMB_DIR = "/data3/seungheon/fx_embedding_new"
DEV_DIR = "/data3/seungheon/fx_embedding_outputs"
INST_TYPES = ['drums', 'guitar', 'piano']
STOP_WORDS = ['happy', 'cold']


def load_dsp_embs(path):
    dsp_emb = np.load(path, allow_pickle=True)
    embs = []
    temp_dict = dict(dsp_emb.item())
    for k,v in temp_dict.items():
        embs.append(v)
    embs = np.hstack(embs)
    return embs

def load_dev_embeddings(fx_type, inst_type, eval_emb):
    dev_ids = os.listdir(f"{DEV_DIR}/{args.fx_emb_type}/lowerbound/random_params/{fx_type}/{inst_type}/audio")
    if args.fx_emb_type == "DSP":
        dev_emb = np.stack([load_dsp_embs(f"{DEV_DIR}/{args.fx_emb_type}/lowerbound/random_params/{fx_type}/{inst_type}/audio/{_id}") for _id in dev_ids], axis=0)
    else:
        dev_emb = np.stack([np.load(f"{DEV_DIR}/{args.fx_emb_type}/lowerbound/random_params/{fx_type}/{inst_type}/audio/{_id}") for _id in dev_ids], axis=0)
    dev_emb = dev_emb[:50]
    dev_ids = dev_ids[:50]
    return dev_emb, dev_ids

def eval_results(fx_type, inst_type, text, eval_set):
    if args.fx_emb_type == "DSP":
        eval_emb = np.stack([load_dsp_embs(f"{EMB_DIR}/{args.fx_emb_type}/{fx_type}/{_id}/{inst_type}.npy") for _id in eval_set], axis=0)
    else:
        eval_emb = np.stack([np.load(f"{EMB_DIR}/{args.fx_emb_type}/{fx_type}/{_id}/{inst_type}.npy") for _id in eval_set], axis=0)
    dev_emb, dev_ids = load_dev_embeddings(fx_type, inst_type, eval_emb)
    if eval_emb.ndim == 3:
        eval_emb = eval_emb.mean(axis=1)
    if dev_emb.ndim == 3:
        dev_emb = dev_emb.mean(axis=1)
    mmd_score = compute_mmd(dev_emb, eval_emb)
    return mmd_score, dev_ids

def main():
    args = parser.parse_args()
    eval_db = load_dataset("seungheondoh/socialfx-gen-eval")
    results = []
    for fx_type in ['eq', 'reverb']:
        df_eval = pd.DataFrame(eval_db[fx_type])
        for _, row in df_eval.iterrows():
            for inst_type in INST_TYPES:
                text = row["input"]
                if (fx_type == "eq" and text in STOP_WORDS):
                    continue
                ids = row['output']
                n_samples = len(ids)
                eval_set = set(ids)
                mmd_score, dev_ids = eval_results(fx_type, inst_type, text, eval_set)
                print("lowerbound", text, len(dev_ids))
                os.makedirs(f"./exp/lowerbound/{fx_type}/{inst_type}", exist_ok=True)
                with open(f"./exp/lowerbound/{fx_type}/{inst_type}/dev_ids.json", "w") as f:
                    json.dump({"fx_type": fx_type, "inst_type": inst_type, "dev_ids": dev_ids}, f, indent=4)

                results.append({
                    "fx_type": fx_type,
                    "inst_type": inst_type,
                    "mmd_score": mmd_score,
                    "n_samples": n_samples
                })
    df_results = pd.DataFrame(results)
    os.makedirs("./exp/lowerbound", exist_ok=True)
    df_results.to_csv(f"./exp/lowerbound/{args.fx_emb_type}.csv", index=False)

if __name__ == "__main__":
    main()
