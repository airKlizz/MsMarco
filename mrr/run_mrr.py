import pandas as pd
import json
from tqdm import tqdm

class EvaluationQuery():
    def __init__(self, qid, pids, queries_dict, passages_dict, batch_size):
        self.qid = qid
        self.query = queries_dict[qid]
        self.passages = {}
        for pid in pids:
            self.passages[pid] = passages_dict[str(pid)]
        self.batch_size = batch_size

    def __str__(self):
        return "<{} qid:{}, pids:{}>".format(type(self), self.qid, self.passages.keys())
    
    def score(self, scorer):
        self.scores = scorer.score_query_passages(self.query, self.passages.values(), self.batch_size)
        pids_sorted = [pid for _,pid in sorted(list(zip(self.scores, self.passages.keys())), key=lambda x: x[0], reverse=True)]
        score_str = ""
        for rank, pid in enumerate(pids_sorted):
            score_str = score_str + "{}\t{}\t{}\n".format(self.qid, pid, rank+1)
        return score_str



class EvaluationQueries():
    def __init__(self, bm25_path, queries_path, passages_path, n_top):
        '''
        Read BM25 top 1000 and reduce to n_top results per query
        '''
        bm25_df = pd.read_csv(bm25_path, sep='	', header=None, names=['qid', 'pid', 'rank'])
        bm25_df = bm25_df.loc[bm25_df['rank'] <= n_top]
        all_qids = bm25_df['qid'].to_list()

        '''
        Read queries and passages files to create dicts
        '''
        # QUERIES
        queries_df = pd.read_csv(queries_path, sep='\t', header=None, names=['qid', 'querie'])
        queries = dict(zip(queries_df['qid'].to_list(), queries_df['querie'].to_list()))
        # PASSAGES
        with open(passages_path) as json_file:
            passages = json.load(json_file)

        '''
        Create list of EvaluationQuery object
        '''
        self.evaluation_queries = []
        for qid in set(all_qids):
            df = bm25_df.loc[bm25_df['qid'] == qid]
            pids = df['pid'].to_list()
            self.evaluation_queries.append(EvaluationQuery(qid, pids, queries, passages, n_top))

    def __str__(self):
        s = '<EvaluationQueries '
        for i, evaluation_query in enumerate(self.evaluation_queries):
            s += evaluation_query.__str__()
            if i == 9:
                s += '...'
                break
        s += ' />'
        return s

    def score(self, scorer, output_path, number_of_queries=None):
        score_str = ""
        print('MMR Evaluation on {} queries'.format(len(self.evaluation_queries)))
        for evaluation_query in tqdm(self.evaluation_queries[:number_of_queries], desc="MMR Evaluation in progress"):
            score_str = score_str + evaluation_query.score(scorer)
        f = open(output_path, "w")
        f.write(score_str)
        f.close()