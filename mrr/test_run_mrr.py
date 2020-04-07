from run_mrr import EvaluationQueries
from msmarco_eval import compute_metrics_from_files

class FakeScorer():
    def __init__(self):
        pass

    def score_query_passages(self, query, passages, batch_size):
        return [1] * len(passages)

bm25_path = "../data/evaluation/bm25/run.dev.small.tsv"
queries_path = "../data/queries/queries.dev.small.tsv"
passages_path = "../data/passages/passages.bm25.small.json"
candidate_path = "../data/evaluation/test/run.tsv"
reference_path = "../data/evaluation/gold/qrels.dev.small.tsv"

n_top = 50
n_queries_to_evaluate = None

mrr_ref = 0.18741227770955546

model = FakeScorer()

mrr = EvaluationQueries(bm25_path, queries_path, passages_path, n_top)

mrr.score(model, candidate_path, n_queries_to_evaluate)
mrr_metrics = compute_metrics_from_files(reference_path, candidate_path)

assert int(1000*mrr_ref) == int(1000*mrr_metrics['MRR @10']), "Test failed"

print(mrr_metrics)
print('Test ok')
