import pandas as pd
import json

from tqdm import tqdm

N_TOP = 3

bm25_run_path = "evaluation/bm25/run.dev.small.tsv"
docs_paths = ["collections/docs00.json", "collections/docs01.json", "collections/docs02.json", "collections/docs03.json",
              "collections/docs04.json", "collections/docs05.json", "collections/docs06.json", "collections/docs07.json",
              "collections/docs08.json"]

bm25_df = pd.read_csv(bm25_run_path, sep='	', header=0, names=['qid', 'pid', 'rank'])
bm25_df = bm25_df.loc[bm25_df['rank'] <= N_TOP]

qids = list(set(bm25_df['qid'].to_list()))
pids = list(set(bm25_df['pid'].to_list()))
ranks = bm25_df['rank'].to_list()

print('Number of qids: ', len(qids))
print('Number of pids: ', len(pids))
print()

small_pids = {}
for i, doc in enumerate(docs_paths):
    with open(doc, "r") as f:
        for p in tqdm(f.readlines(), desc='document '+str(i)+' processing'):
            passage = json.loads(p)
            if int(passage['id']) in pids:
                small_pids[passage['id']] = passage['contents']

print('Number of passage found: ', len(small_pids))
print()

with open('passages.bm25.small.json', 'w') as fp:
    json.dump(small_pids, fp)
        


