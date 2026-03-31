import os
import pandas as pd
from datasets import load_dataset
import argparse
import random
import ast
import numpy as np
from llm4mp.evaluation.metrics.dist_matching import compute_mmd, compute_similarity_stats

parser = argparse.ArgumentParser()
parser.add_argument("--fx_emb_type", type=str, default="STITO")
args = parser.parse_args()

EMB_DIR = "/data3/seungheon/fx_embedding_new"
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

def eval_results(fx_type, inst_type, text, eval_set, dev_set):
    if args.fx_emb_type == "DSP":
        eval_emb = [load_dsp_embs(f"{EMB_DIR}/{args.fx_emb_type}/{fx_type}/{_id}/{inst_type}.npy") for _id in eval_set]
        dev_emb = [load_dsp_embs(f"{EMB_DIR}/{args.fx_emb_type}/{fx_type}/{_id}/{inst_type}.npy") for _id in dev_set]
    else:
        eval_emb = [np.load(f"{EMB_DIR}/{args.fx_emb_type}/{fx_type}/{_id}/{inst_type}.npy") for _id in eval_set]
        dev_emb = [np.load(f"{EMB_DIR}/{args.fx_emb_type}/{fx_type}/{_id}/{inst_type}.npy") for _id in dev_set]
    eval_emb = np.stack(eval_emb, axis=0)
    dev_emb = np.stack(dev_emb, axis=0)
    if eval_emb.ndim == 3:
        eval_emb = eval_emb.mean(axis=1)
    if dev_emb.ndim == 3:
        dev_emb = dev_emb.mean(axis=1)
    mmd_score = compute_mmd(dev_emb, eval_emb)
    return mmd_score

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
                random.shuffle(ids)
                n_samples = len(ids)
                eval_set = ids[:n_samples//2]
                dev_set = ids[n_samples//2:]
                mmd_score = eval_results(fx_type, inst_type, text, eval_set, dev_set)
                results.append({
                    "fx_type": fx_type,
                    "text": text,
                    "inst_type": inst_type,
                    "mmd_score": mmd_score,
                    "n_samples": n_samples
                })
    df_results = pd.DataFrame(results)
    os.makedirs("./exp/upperbound", exist_ok=True)
    df_results.to_csv(f"./exp/upperbound/{args.fx_emb_type}.csv", index=False)

if __name__ == "__main__":
    main()
