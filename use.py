import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
from model.scorer import Scorer

from newspaper import Article
from googlesearch import search
from rank_bm25 import BM25Okapi

class Passage():
    def __init__(self, text, source, date):
        self.text = text
        self.source = source
        self.date = date
    
    def __eq__(self, other): 
        if not isinstance(other, Passage):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.text == other.text

    def __repr__(self):
        return 'Date: {}\nSource: {}\n{}'.format(self.date, self.source, self.text)

    def __str__(self):
        return 'Date: {}\nSource: {}\n{}'.format(self.date, self.source, self.text)

class Ranker():
    def __init__(self, topic, model_name, weights_path, max_length=256, num_classes=2):
        self.topic = topic
        self.url_done = []
        self.passages = []
        self.bm25_scores = []

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.scorer = Scorer(tokenizer, TFAutoModel, max_length, num_classes)
        self.scorer.from_pretrained(model_name)
        self.scorer(tf.zeros([1, 3, 256], tf.int32))
        self.scorer.load_weights(weights_path)
        self.scorer.compile(run_eagerly=True)

    def find_passages(self, num_urls, wikipedia=True):
        urls = query_to_urls(self.topic, num_urls, wikipedia=wikipedia)
        for url in urls:
            if url in self.url_done: continue
            try:
                passages = passages_from_url(url)
                for passage in passages:
                    if passage in self.passages:
                        continue
                    self.passages.append(passage)
                self.url_done += url
            except:
                pass

    def run_bm25(self):
        tokenized_corpus = [passage.text.split(" ") for passage in self.passages]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_topic = self.topic.split(" ")
        self.bm25_scores = bm25.get_scores(tokenized_topic)
        assert len(self.passages) == len(self.bm25_scores)

    def get_bm25_top(self, top_n=100):
        if len(self.passages) != len(self.bm25_scores):
            self.run_bm25()
        passages_scores = list(zip(*sorted(zip(self.passages, self.bm25_scores), key=lambda x: x[1], reverse=True)))
        return passages_scores[0][:top_n], passages_scores[1][:top_n]

    def get_rerank_top(self, top_n=100, top_n_bm25=100, batch_size=16):
        bm25_top, _ = self.get_bm25_top(top_n_bm25)
        bm25_top_text = [passage.text for passage in bm25_top]
        rerank_scores = self.scorer.score_query_passages(self.topic, bm25_top_text, batch_size)
        passages_scores = list(zip(*sorted(zip(bm25_top, rerank_scores), key=lambda x: x[1], reverse=True)))
        return passages_scores[0][:top_n], passages_scores[1][:top_n]

def query_to_urls(query, num_urls, wikipedia=True):
    urls = []
    for url in search(query, lang='en', num=num_urls, stop=num_urls, extra_params={'lr': 'lang_en'}):
        if wikipedia or "wikipedia" not in url:
            urls.append(url)
    return urls

def passages_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    date = article.publish_date
    passages = []
    for passage in text.split('\n\n'):
        if is_clean(passage):
            passages.append(Passage(passage, url, date))
    return passages

def is_clean(passage):
    # Too short
    if len(passage.split(' ')) < 10:
        return False
    # List
    if len(passage.split('\n')) > 2:
        return False
    # Items
    if len(passage.split('\t')) > 2:
        return False
    return True


