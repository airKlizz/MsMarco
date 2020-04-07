from newspaper import Article
from googlesearch import search
from rank_bm25 import BM25Okapi

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

def passages_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    passages = []
    for passage in text.split('\n\n'):
        if is_clean(passage):
            passages.append(passage)
    return passages

def query_to_urls(query, num_urls, wikipedia=True):
    urls = []
    for url in search(query, lang='en', num=num_urls, stop=num_urls, extra_params={'lr': 'lang_en'}):
        if wikipedia or "wikipedia" not in url:
            urls.append(url)
    return urls

def query_to_passages(query, num_urls, wikipedia=True):
    urls = query_to_urls(query, num_urls, wikipedia=wikipedia)
    passages = []
    for url in urls:
        try:
            passages += passages_from_url(url)
        except:
            print('Request forbidden for {}'.format(url))
    return passages

def bm25(query, passages, n_top):
    tokenized_corpus = [doc.split(" ") for doc in passages]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    return bm25.get_top_n(tokenized_query, passages, n=n_top)

def query_to_top_n_passages(query, num_urls, n_top, wikipedia=True):
    passages = query_to_passages(query, num_urls, wikipedia)
    return bm25(query, passages, n_top), passages
        
