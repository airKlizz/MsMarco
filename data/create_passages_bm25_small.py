import json
from tqdm import tqdm
import argparse
import pandas as pd


'''
Convert collections json docs to a dict {pid: passage}
'''
docs_path = ["collections/docs00.json",
             "collections/docs01.json",
             "collections/docs02.json",
             "collections/docs03.json",
             "collections/docs04.json",
             "collections/docs05.json",
             "collections/docs06.json",
             "collections/docs07.json",
             "collections/docs08.json"]

passages = {}
for doc_path in docs_path:
    with open(doc_path, 'r') as f:
        lines = f.readlines()
        print(
            "Number of lines: {} for {}".format(len(lines), doc_path)
        )
        for line in tqdm(lines, desc="convert lines to dict"):
            passage = json.loads(line)
            passages[passage['id']] = passage['contents']
            

print(
    "Passages | type: {}, length: {}".format(type(passages), len(passages))
)


'''
Find pids list to read
'''
parser = argparse.ArgumentParser()
parser.add_argument("n_top", type=int, help="number of passages to re-rank after BM25")
args = parser.parse_args()


bm25_run_path = "evaluation/bm25/run.dev.small.tsv"

bm25_df = pd.read_csv(bm25_run_path, sep='	', header=None, names=['qid', 'pid', 'rank'])
bm25_df = bm25_df.loc[bm25_df['rank'] <= args.n_top]

qids = list(set(bm25_df['qid'].to_list()))
pids = list(set(bm25_df['pid'].to_list()))
ranks = bm25_df['rank'].to_list()

print('Number of qids: ', len(qids))
print('Number of pids: ', len(pids))
print()


'''
Create passages small json file
'''
small_passages = {}
for pid in tqdm(pids, desc="Creating small passages"):
    small_passages[pid] = passages[str(pid)]

print('Number of passage found: ', len(small_passages))
print()

with open('passages/passages.bm25.small.json', 'w') as fp:
    json.dump(small_passages, fp)
        
