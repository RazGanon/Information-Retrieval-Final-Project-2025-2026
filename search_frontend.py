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
# FIXED: Matching the directory name we created in the VM
POSTINGS_DIR = 'postings_gcp' 

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

    # 1. Load body index
    try:
        #load the index from local disk (bucket_name=None)
        idx_body = InvertedIndex.read_index(POSTINGS_DIR, 'index_body', None)
        #link the index to the .bin files folder
        idx_body.posting_locs_dir = POSTINGS_DIR
        print("body index loaded")
    except Exception as e:
        print(f"error loading body index: {e}")

    # 2. load title index
    #try:
    #    idx_title = InvertedIndex.read_index(POSTINGS_DIR, 'index_title', '') 
    #    idx_title.posting_locs_dir = POSTINGS_DIR
    #    print("title index loaded")
    #except Exception as e:
    #    print(f"error loading title index: {e}")

    # 3. load anchor index
    #try:
    #    idx_anchor = InvertedIndex.read_index(POSTINGS_DIR, 'index_anchor', '') 
    #    idx_anchor.posting_locs_dir = POSTINGS_DIR
    #    print("anchor index loaded")
    #except Exception as e:
    #    print(f"error loading anchor index: {e}")

    # 4. load pagerank
    try:
        with open(os.path.join(POSTINGS_DIR, 'pagerank.pkl'), 'rb') as f:
            pagerank_dict = pickle.load(f)
        print(f"pagerank loaded ({len(pagerank_dict)} keys)")
    except Exception as e:
        print(f"pagerank not found: {e}")

    # 5. load pageviews
    try:
        with open(os.path.join(POSTINGS_DIR, 'pageviews.pkl'), 'rb') as f:
            pageview_dict = pickle.load(f)
        print(f"pageviews loaded ({len(pageview_dict)} keys)")
    except Exception as e:
        print(f"pageviews not found: {e}")

    # 6. load titles dictionary
    try:
        with open(os.path.join(POSTINGS_DIR, 'id2titles.pkl'), 'rb') as f:
            titles_dict = pickle.load(f)
        print(f"titles loaded ({len(titles_dict)} keys)")
    except Exception as e:
        print(f"titles dictionary not found: {e}")

    # 7. load bm25 stats and doc lengths
    try:
        with open(os.path.join(POSTINGS_DIR, 'bm25_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
            bm25_body_avgdl = stats.get('avg_body_len', 320.0)
            bm25_title_avgdl = stats.get('avg_title_len', 2.5)
            bm25_anchor_avgdl = stats.get('avg_anchor_len', 3.0)
            
            doc_len_body = stats.get('doc_lengths', {})
            
            print(f"bm25 stats loaded (doc_len_body keys: {len(doc_len_body)})")
    except:
        print("bm25 stats not found, using defaults")
        bm25_body_avgdl = 320.0
        bm25_title_avgdl = 2.5
        bm25_anchor_avgdl = 3.0
        doc_len_body = {}

    print("data loading finished")

# run loading
load_data()

# --- Helper Functions ---

def calc_bm25(query_tokens, index, avgdl, k1=1.2, b=0.75):
    """
    calculate bm25 score for a given query and index
    """
    scores = Counter()
    if index is None: return scores
    
    # total number of docs (N)
    N = len(pagerank_dict) if pagerank_dict else 6348910
    
    for term in query_tokens:
        if term in index.df:
            df = index.df[term]
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            
            try:
                # FIXED: Passing POSTINGS_DIR as base_dir so it finds the bin files
                posting_list = index.read_a_posting_list(POSTINGS_DIR, term, None)
                
                for doc_id, tf in posting_list:
                    numerator = idf * tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (avgdl)) # Note: simplified without doc len per doc
                    scores[doc_id] += numerator / denominator
            except:
                continue
                
    return scores

def merge_results(bm25_body, bm25_title, bm25_anchor, pr_dict, w_body=0.35, w_title=0.45, w_anchor=0.20, w_pr=1.5):
    """
    merge scores from different sources using linear combination
    """
    all_docs = set(bm25_body.keys()) | set(bm25_title.keys()) | set(bm25_anchor.keys())
    final_scores = []
    
    for doc_id in all_docs:
        s_body = bm25_body.get(doc_id, 0.0)
        s_title = bm25_title.get(doc_id, 0.0)
        s_anchor = bm25_anchor.get(doc_id, 0.0)
        
        pr_val = pr_dict.get(doc_id, 0.0)
        pr_score = math.log(pr_val + 1, 10) 
        
        score = (w_body * s_body) + \
                (w_title * s_title) + \
                (w_anchor * s_anchor) + \
                (w_pr * pr_score)
        
        final_scores.append((doc_id, score))
    
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
    
    # 1. tokenize query using the helper function
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    # 2. calculate bm25 scores for each index
    # using helper function defined globally.
    # weights logic: title and anchor are strong signals, body is noisy.
    scores_title = calc_bm25(tokens, idx_title, bm25_title_avgdl)
    scores_anchor = calc_bm25(tokens, idx_anchor, bm25_anchor_avgdl)
    scores_body = calc_bm25(tokens, idx_body, bm25_body_avgdl)

    # 3. merge scores
    # weights configuration (optimized for wikipedia structure):
    # title & anchor: high precision signals.
    # body: recall signal (low weight to avoid noise).
    # pr & pv: quality signals (log-smoothed).
    w_title = 0.6
    w_anchor = 0.4
    w_body = 0.05 
    w_pr = 0.5
    w_pv = 0.2     

    # set of all candidate docs from all indices
    all_doc_ids = set(scores_title.keys()) | set(scores_anchor.keys()) | set(scores_body.keys())

    final_scores = []

    for doc_id in all_doc_ids:
        # retrieve bm25 scores (default 0 if not found)
        s_title = scores_title.get(doc_id, 0.0)
        s_anchor = scores_anchor.get(doc_id, 0.0)
        s_body = scores_body.get(doc_id, 0.0)
        
        # get pagerank score and apply log smoothing
        # avoid log(0) errors
        pr_val = pagerank_dict.get(doc_id, 0.0)
        pr_score = math.log(pr_val + 1, 10) if pr_val > 0 else 0

        # get pageviews score and apply log smoothing (required for full grade)
        pv_val = pageview_dict.get(doc_id, 0)
        pv_score = math.log(pv_val + 1, 10) if pv_val > 0 else 0

        # calculate final score using linear combination
        total_score = (w_title * s_title) + \
                      (w_anchor * s_anchor) + \
                      (w_body * s_body) + \
                      (w_pr * pr_score) + \
                      (w_pv * pv_score)

        final_scores.append((doc_id, total_score))

    # 4. sort by score in descending order to get best results first
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # take the top 100 results
    top_100 = final_scores[:100]

    # 5. map doc ids to titles using the dictionary loaded in setup
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
    
    # 1. tokenize the query using the staff provided tokenizer
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    # 2. initialize counters
    query_counts = Counter(tokens)
    scores = Counter()

    # set corpus size for idf calculation
    # using pagerank size as approximation or default wikipedia size
    N = len(pagerank_dict) if pagerank_dict else 6348910

    # verify body index is loaded
    if idx_body:
        # iterate over unique terms in the query
        for term, q_tf in query_counts.items():
            if term in idx_body.df:
                # calculate idf using log10
                df = idx_body.df[term]
                idf = math.log(N / df, 10)

                # calculate query weight w_q = tf * idf
                # note: for ranking order, normalizing query vector is not strictly necessary
                w_q = q_tf * idf

                try:
                    # critical fix: use POSTINGS_DIR to find bin files in the correct folder
                    pl = idx_body.read_a_posting_list(POSTINGS_DIR, term, None)

                    # iterate over the posting list and accumulate dot product
                    for doc_id, tf in pl:
                        # calculate document weight w_d = tf * idf
                        # accumulate dot product: score += w_q * w_d
                        scores[doc_id] += w_q * (tf * idf)
                except:
                    continue

    # 3. normalize scores by document length to get cosine similarity
    # cosine sim = (A . B) / (||A|| * ||B||)
    # we divide by doc norm which should be pre-calculated in doc_len_body
    final_scores = []
    
    for doc_id, dot_product in scores.items():
        # retrieve document length/norm from loaded stats
        # fallback to 1 to avoid division by zero
        doc_len = doc_len_body.get(doc_id, 1)
        
        sim_score = dot_product / doc_len
        final_scores.append((doc_id, sim_score))

    # 4. sort by similarity score descending
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # 5. retrieve titles for the top 100 documents
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
                # critical change: use postings_dir to find the bin files
                posting_list = idx_title.read_a_posting_list(POSTINGS_DIR, term, "")
                
                # for each document where the term appears add 1 to its score
                for doc_id, _ in posting_list:
                    scores[doc_id] += 1
            except:
                continue

    # 4. sort the results by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 5. format the results as list of tuples (doc_id, title)
    # this is where we use the data from id2titles.pkl
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
    
    # 2. use a set to keep only unique query terms for distinct counting
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
                # critical fix: use postings_dir to locate the bin files
                # passing empty string "" causes file not found error on vm
                posting_list = idx_anchor.read_a_posting_list(POSTINGS_DIR, term, "")
                
                # increment score for every document linked by this term
                # binary ranking: just counting distinct query terms
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
    app.run(host='0.0.0.0', port=8080, debug=False)
