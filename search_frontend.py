from flask import Flask, request, jsonify
import sys
import math
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
import nltk
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex, MultiFileReader

# --- App Configuration ---
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- Tokenizer Setup ---
nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]

# --- Global Data & Paths ---
# folder containing all bin/pkl files
POSTINGS_DIR = 'postings' 

# placeholders
idx_body = None
idx_title = None
idx_anchor = None
pagerank_dict = {}
pageview_dict = {}
titles_dict = {}

# bm25 stats
bm25_body_avgdl = 0
bm25_title_avgdl = 0
bm25_anchor_avgdl = 0

# document lengths dictionary for cosine similarity
doc_len_body = {}

def load_data():
    """
    load all indices and helper dictionaries from local disk.
    """
    global idx_body, idx_title, idx_anchor, pagerank_dict, pageview_dict, titles_dict
    global bm25_body_avgdl, bm25_title_avgdl, bm25_anchor_avgdl, doc_len_body
    
    print("loading data from local disk...")

    # 1. load body index
    try:
        idx_body = InvertedIndex.read_index(POSTINGS_DIR, 'index_body', '') 
        idx_body.posting_locs_dir = os.path.join(POSTINGS_DIR, 'body')
        print("body index loaded")
    except Exception as e:
        print(f"error loading body index: {e}")

    # 2. load title index
    try:
        idx_title = InvertedIndex.read_index(POSTINGS_DIR, 'index_title', '') 
        idx_title.posting_locs_dir = os.path.join(POSTINGS_DIR, 'title')
        print("title index loaded")
    except Exception as e:
        print(f"error loading title index: {e}")

    # 3. load anchor index
    try:
        idx_anchor = InvertedIndex.read_index(POSTINGS_DIR, 'index_anchor', '') 
        idx_anchor.posting_locs_dir = os.path.join(POSTINGS_DIR, 'anchor')
        print("anchor index loaded")
    except Exception as e:
        print(f"error loading anchor index: {e}")

    # 4. load pagerank
    try:
        with open('pagerank.pkl', 'rb') as f:
            pagerank_dict = pickle.load(f)
        print(f"pagerank loaded ({len(pagerank_dict)} keys)")
    except Exception as e:
        print(f"pagerank not found: {e}")

    # 5. load pageviews
    try:
        with open('pageviews.pkl', 'rb') as f:
            pageview_dict = pickle.load(f)
        print(f"pageviews loaded ({len(pageview_dict)} keys)")
    except Exception as e:
        print(f"pageviews not found: {e}")

    # 6. load titles dictionary
    try:
        with open('titles.pkl', 'rb') as f:
            titles_dict = pickle.load(f)
        print(f"titles loaded ({len(titles_dict)} keys)")
    except Exception as e:
        print(f"titles dictionary not found: {e}")

    # 7. load bm25 stats and doc lengths
    try:
        with open('bm25_stats.pkl', 'rb') as f:
            stats = pickle.load(f)
            bm25_body_avgdl = stats.get('avg_body_len', 320.0)
            bm25_title_avgdl = stats.get('avg_title_len', 2.5)
            bm25_anchor_avgdl = stats.get('avg_anchor_len', 3.0)
            
            # load document lengths for search_body normalization
            # if the file exists but key is missing, default to empty dict
            doc_len_body = stats.get('doc_lengths', {})
            
            print(f"bm25 stats loaded (doc_len_body keys: {len(doc_len_body)})")
    except:
        print("bm25 stats not found, using defaults")
        bm25_body_avgdl = 320.0
        bm25_title_avgdl = 2.5
        bm25_anchor_avgdl = 3.0
        doc_len_body = {} # avoid crash if file missing

    print("data loading finished")

# run loading
load_data()

# --- Helper Functions ---

def calc_bm25(query_tokens, index, avgdl, k1=1.2, b=0.75):
    """
    calculate bm25 score for a given query and index
    uses idf from the index and tf from posting lists
    assumes average document length for normalization to save memory
    """
    scores = Counter()
    if index is None: return scores
    
    # total number of docs (N) - using pagerank size as approximation for corpus size
    N = len(pagerank_dict) if pagerank_dict else 6348910
    
    for term in query_tokens:
        if term in index.df:
            # calculate inverse document frequency
            df = index.df[term]
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            
            try:
                # read posting list from disk for the specific term
                # passing empty strings because we read from local disk
                posting_list = index.read_a_posting_list("", term, "")
                
                for doc_id, tf in posting_list:
                    # bm25 formula with avgdl approximation
                    # this assumes document length is approx avgdl
                    numerator = idf * tf * (k1 + 1)
                    denominator = tf + k1 
                    scores[doc_id] += numerator / denominator
            except:
                continue
                
    return scores

def merge_results(bm25_body, bm25_title, bm25_anchor, pr_dict, w_body=0.35, w_title=0.45, w_anchor=0.20, w_pr=1.5):
    """
    merge scores from different sources using linear combination
    apply log transformation to pagerank to smooth the values
    """
    all_docs = set(bm25_body.keys()) | set(bm25_title.keys()) | set(bm25_anchor.keys())
    final_scores = []
    
    for doc_id in all_docs:
        # retrieve scores, default to 0 if doc not in specific index
        s_body = bm25_body.get(doc_id, 0.0)
        s_title = bm25_title.get(doc_id, 0.0)
        s_anchor = bm25_anchor.get(doc_id, 0.0)
        
        # get pagerank score and apply log smoothing
        pr_val = pr_dict.get(doc_id, 0.0)
        pr_score = math.log(pr_val + 1, 10) # +1 to avoid log(0)
        
        # weighted sum
        score = (w_body * s_body) + \
                (w_title * s_title) + \
                (w_anchor * s_anchor) + \
                (w_pr * pr_score)
        
        final_scores.append((doc_id, score))
    
    # return sorted list by score descending
    return sorted(final_scores, key=lambda x: x[1], reverse=True)

# --- API Endpoints ---

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # tokenize the query using the helper function we defined earlier
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    # define weights for the ensemble based on testing
    # title gets high weight because it's a strong signal
    w_body = 0.35
    w_title = 0.45
    w_anchor = 0.20
    w_pr = 1.5 # weight for log(pagerank)

    # calculate bm25 scores for each index using our helper function
    # we use the stats (avgdl) loaded during startup
    scores_body = calc_bm25(tokens, idx_body, bm25_body_avgdl)
    scores_title = calc_bm25(tokens, idx_title, bm25_title_avgdl)
    scores_anchor = calc_bm25(tokens, idx_anchor, bm25_anchor_avgdl)

    # merge scores from all sources
    # use a set to gather all unique doc ids encountered
    all_doc_ids = set(scores_body.keys()) | set(scores_title.keys()) | set(scores_anchor.keys())

    final_scores = []

    for doc_id in all_doc_ids:
        # retrieve individual scores, defaulting to 0 if not found
        s_body = scores_body.get(doc_id, 0.0)
        s_title = scores_title.get(doc_id, 0.0)
        s_anchor = scores_anchor.get(doc_id, 0.0)

        # get pagerank score and apply log smoothing
        # adding 1 to avoid log(0) for very small values
        pr_val = pagerank_dict.get(doc_id, 0.0)
        pr_score = math.log(pr_val + 1, 10)

        # calculate final score using linear combination
        total_score = (w_body * s_body) + \
                      (w_title * s_title) + \
                      (w_anchor * s_anchor) + \
                      (w_pr * pr_score)

        final_scores.append((doc_id, total_score))

    # sort by score in descending order to get best results first
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # take the top 100 results
    top_100 = final_scores[:100]

    # map doc ids to titles using the dictionary loaded in setup
    # fallback to doc_id string if title is missing
    res = [(str(doc_id), titles_dict.get(doc_id, str(doc_id))) for doc_id, score in top_100]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # tokenize the query using the staff provided tokenizer
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    # initialize counters for query term frequencies and document scores
    query_counts = Counter(tokens)
    scores = Counter()

    # set total number of documents for idf calculation
    # using pagerank size or a default approx for wikipedia size
    N = len(pagerank_dict) if pagerank_dict else 6348910

    # check if body index is loaded to avoid errors
    if idx_body:
        # iterate over unique terms in the query
        for term, q_tf in query_counts.items():
            if term in idx_body.df:
                # calculate idf using log10
                df = idx_body.df[term]
                idf = math.log(N / df, 10)

                # calculate query weight w_q = tf * idf
                w_q = q_tf * idf

                try:
                    # read posting list for the term from local storage
                    # this returns a list of (doc_id, tf)
                    pl = idx_body.read_a_posting_list("", term, "")

                    # iterate over the posting list and accumulate dot product
                    for doc_id, tf in pl:
                        # calculate document weight w_d = tf * idf
                        # score += w_q * w_d
                        scores[doc_id] += w_q * (tf * idf)
                except:
                    continue

    # normalize scores by document length to get cosine similarity
    # cosine sim = (A . B) / (||A|| * ||B||)
    # we ignore query norm ||A|| because it is constant for ranking
    final_scores = []
    
    for doc_id, dot_product in scores.items():
        # retrieve document length from the loaded stats
        # fallback to 1 if missing to avoid division by zero
        doc_len = doc_len_body.get(doc_id, 1)
        
        sim_score = dot_product / doc_len
        final_scores.append((doc_id, sim_score))

    # sort by similarity score descending
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # retrieve titles for the top 100 documents
    res = [(str(doc_id), titles_dict.get(doc_id, str(doc_id))) for doc_id, score in final_scores[:100]]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # 1. tokenize the query using the staff provided tokenizer
    tokens = tokenize(query)
    
    if not tokens:
        return jsonify(res)
    
    # 2. keep only unique terms to satisfy the distinct words requirement
    query_terms = set(tokens)
    
    # dictionary to store the score for each document
    scores = Counter()
    
    if idx_title is None:
        return jsonify([])

    # 3. iterate over each unique term in the query
    for term in query_terms:
        if term in idx_title.df:
            try:
                # read the posting list locally 
                # passing empty strings because we are reading from local disk not bucket
                posting_list = idx_title.read_a_posting_list("", term, "")
                
                # for each document where the term appears add 1 to its score
                for doc_id, _ in posting_list:
                    scores[doc_id] += 1
            except:
                continue

    # 4. sort the results by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 5. format the results as list of tuples (doc_id, title)
    res = [(str(doc_id), titles_dict.get(doc_id, str(doc_id))) for doc_id, score in sorted_scores]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # 1. tokenize the query using the staff provided tokenizer
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)
    
    # 2.use a set to keep only unique query terms for distinct counting
    unique_tokens = set(tokens)
    
    # 3. dictionary to accumulate scores per document
    scores = Counter()
    
    # 4. verify index is loaded
    if idx_anchor is None:
        return jsonify([])
    
    # 5. iterate over each unique term
    for term in unique_tokens:
        if term in idx_anchor.df:
            try:
                # read the posting list locally
                # passing empty string as bucket name implies local read
                posting_list = idx_anchor.read_a_posting_list("", term, "")
                
                # increment score for every document linked by this term
                for doc_id, _ in posting_list:
                    scores[doc_id] += 1
            except:
                continue
                
    # 6. sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # format results using titles dictionary or fallback to doc id
    res = [(str(doc_id), titles_dict.get(doc_id, str(doc_id))) for doc_id, score in sorted_scores]

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # iterate over the list of ids provided in the request
    for doc_id in wiki_ids:
        # fetch pagerank score from the global dictionary loaded at startup
        # use 0.0 as default value if the article id is not found
        score = pagerank_dict.get(doc_id, 0.0)
        res.append(score)

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # iterate over the document ids provided in the request
    for doc_id in wiki_ids:
        # retrieve pageview count from the global dictionary
        # default to 0 if the id is missing from the dictionary
        views = pageview_dict.get(doc_id, 0)
        res.append(views)

    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
