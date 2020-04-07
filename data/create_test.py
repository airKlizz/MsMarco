import pandas as pd
import json

from tqdm import tqdm

qrels_test_path = "qrels/qrels.test.tsv"
queries_test_path = "queries/queris.test.tsv"
passages_test_path = "passages/passages.test.small.json"

# QRELS
qrels_df = pd.read_csv(qrels_test_path, sep=' ', header=None, names=['qid', 'Q0', 'pid', 'rating'])

qids = qrels_df['qid'].to_list()
pids = qrels_df['pid'].to_list()
rates = qrels_df['rating'].to_list()

print('Number of qrels: ', len(qids))
print()
print('Number of rate 0: ', rates.count(0))
print('Number of rate 1: ', rates.count(1))
print('Number of rate 2: ', rates.count(2))
print('Number of rate 3: ', rates.count(3))
print()

# QUERIES
queries_df = pd.read_csv(queries_test_path, sep='\t', header=None, names=['qid', 'querie'])
queries = dict(zip(queries_df['qid'].to_list(), queries_df['querie'].to_list()))

print('Number of queries: ', len(queries))
print()

# PASSAGES
with open(passages_test_path) as json_file:
    passages = json.load(json_file)

print('Number of passages: ', len(passages))
print()


with open('test.tsv', 'w') as f:
    for i, (qid, pid) in enumerate(zip(qids, pids)):
        line = str(qid)+'\t'+queries[str(qid)]+'\t'+str(pid)+'\t'+passages[str(pid)]+'\t'+str(rates[i])+'\n'
        f.write(line)
