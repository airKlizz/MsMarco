import json
from tqdm import tqdm
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
qrels_test_path = "qrels/qrels.test.tsv"

qrels_df = pd.read_csv(qrels_test_path, sep=' ', header=None, names=['qid', 'Q0', 'pid', 'rating'])

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

'''
Create passages small json file
'''
small_passages = {}
for pid in tqdm(pids, desc="Creating small passages"):
    small_passages[pid] = passages[str(pid)]

print('Number of passage found: ', len(small_passages))
print()

with open('passages.test.small.json', 'w') as fp:
    json.dump(small_passages, fp)
        


