import pandas as pd
import json

from tqdm import tqdm

qrels_test_path = "qrels/qrels.test.tsv"
docs_paths = ["collections/docs00.json", "collections/docs01.json", "collections/docs02.json", "collections/docs03.json",
              "collections/docs04.json", "collections/docs05.json", "collections/docs06.json", "collections/docs07.json",
              "collections/docs08.json"]

qrels_df = pd.read_csv(qrels_test_path, sep=' ', header=0, names=['qid', 'Q0', 'pid', 'rating'])

qids = list(set(qrels_df['qid'].to_list()))
pids = list(set(qrels_df['pid'].to_list()))
rating = qrels_df['rating'].to_list()

print('Number of qids: ', len(qids))
print('Number of pids: ', len(pids))
print()
print('Number of rate 0: ', rating.count(0))
print('Number of rate 1: ', rating.count(1))
print('Number of rate 2: ', rating.count(2))
print('Number of rate 3: ', rating.count(3))
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

with open('passages.test.small.json', 'w') as fp:
    json.dump(small_pids, fp)
        


