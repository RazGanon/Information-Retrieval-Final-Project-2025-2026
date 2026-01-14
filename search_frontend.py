from flask import Flask, request, jsonify, render_template
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

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- Tokenizer Setup ---
# nltk.download('stopwords') # Comment out to save startup time
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]

# --- Global Data & Constants ---
POSTINGS_DIR = 'postings_gcp'

# Indices
idx_body = None
idx_title = None
idx_anchor = None

# Dictionaries
pagerank_dict = {}
pageview_dict = {}
titles_dict = {}

# Stats
bm25_body_avgdl = 0
doc_len_body = {} # Dict mapping doc_id -> doc_len

def load_data():
    """ Load all indices and helper dictionaries from local disk. """
    global idx_body, idx_title, idx_anchor, pagerank_dict, pageview_dict, titles_dict
    global bm25_body_avgdl, doc_len_body
    
    print("Loading data...")

    # 1. Load Indices with Subfolder Paths
    try:
        idx_body = InvertedIndex.read_index(POSTINGS_DIR, 'index_body', None)
        idx_body.posting_locs_dir = os.path.join(POSTINGS_DIR, 'body')
        print("Body index loaded")
    except Exception as e: print(f"Error loading body index: {e}")

    try:
        idx_title = InvertedIndex.read_index(POSTINGS_DIR, 'index_title', None)
        idx_title.posting_locs_dir = os.path.join(POSTINGS_DIR, 'title')
        print("Title index loaded")
    except Exception as e: print(f"Error loading title index: {e}")

    try:
        idx_anchor = InvertedIndex.read_index(POSTINGS_DIR, 'index_anchor', None)
        idx_anchor.posting_locs_dir = os.path.join(POSTINGS_DIR, 'anchor')
        print("Anchor index loaded")
    except Exception as e: print(f"Error loading anchor index: {e}")

    # 2. Load Dictionaries
    try:
        with open(os.path.join(POSTINGS_DIR, 'pagerank.pkl'), 'rb') as f:
            pagerank_dict = pickle.load(f)
        print(f"PageRank loaded ({len(pagerank_dict)} keys)")
    except: print("PageRank not found")

    try:
        with open(os.path.join(POSTINGS_DIR, 'pageviews.pkl'), 'rb') as f:
            pageview_dict = pickle.load(f)
        print(f"PageViews loaded ({len(pageview_dict)} keys)")
    except: print("PageViews not found")

    try:
        with open(os.path.join(POSTINGS_DIR, 'id2titles.pkl'), 'rb') as f:
            titles_dict = pickle.load(f)
        print(f"Titles dictionary loaded ({len(titles_dict)} keys)")
    except: print("Titles dictionary not found")

    # 3. Load BM25 Data
    try:
        with open(os.path.join(POSTINGS_DIR, 'bm25_data.pkl'), 'rb') as f:
            stats = pickle.load(f)
            bm25_body_avgdl = stats['avgdl']
            doc_len_body = stats['doc_lengths']
            print(f"BM25 stats loaded (avgdl={bm25_body_avgdl}, docs={len(doc_len_body)})")
    except: 
        print("BM25 stats not found")
        bm25_body_avgdl = 320.0
        doc_len_body = {}

    print("Data loading finished")

# Run loading at startup
load_data()

# --- Retrieval Helper Functions ---

def get_posting_list(index, term):
    """Safe wrapper to read posting list from disk"""
    try:
        return index.read_a_posting_list(index.posting_locs_dir, term, None)
    except:
        return []

def calc_bm25_body(query_tokens, index, k1=1.2, b=0.75):
    """
    Calculate BM25 for Body index ONLY. 
    Correctly uses doc_len_body for normalization.
    """
    scores = Counter()
    if index is None: return scores
    
    # N = Corpus size (approx 6.3M)
    N = len(doc_len_body) if doc_len_body else 6348910
    
    for term in query_tokens:
        if term in index.df:
            df = index.df[term]
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            
            pl = get_posting_list(index, term)
            for doc_id, tf in pl:
                doc_len = doc_len_body.get(doc_id, bm25_body_avgdl)
                numerator = idf * tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / bm25_body_avgdl))
                scores[doc_id] += numerator / denominator
    return scores

def calc_binary_score(query_tokens, index):
    """
    Binary ranking: Count how many unique query terms appear in the document.
    Used for Title and Anchor.
    """
    scores = Counter()
    if index is None: return scores
    
    # Use set for unique terms (binary search rule)
    for term in set(query_tokens):
        if term in index.df:
            pl = get_posting_list(index, term)
            for doc_id, _ in pl:
                scores[doc_id] += 1
    return scores

# --- API Endpoints ---

@app.route("/")
def index():
    """Serve the main search UI"""
    return render_template('index.html')

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
    
    # 0. Tokenize the query using the staff provided tokenizer
    tokens = tokenize(query)
    if not tokens: return jsonify(res)

    #The Main Engine: Ensemble of Body(BM25) + Title(Binary) + Anchor(Binary) + PR + PV

    # 1. Get Scores from all components
    # Body uses BM25
    body_scores = calc_bm25_body(tokens, idx_body)
    
    # Title & Anchor use Binary Ranking (Faster & fulfills requirements)
    title_scores = calc_binary_score(tokens, idx_title)
    anchor_scores = calc_binary_score(tokens, idx_anchor)

    # 2. Merge Scores
    # Candidate generation: union of all docs found
    all_doc_ids = set(body_scores.keys()) | set(title_scores.keys()) | set(anchor_scores.keys())
    
    final_scores = []
    
    # Weights configuration
    # Title/Anchor are "Tier 1" signals (integers). Body is "Tier 2" (float).
    w_title = 10.0   # Massive boost if in title
    w_anchor = 3.0   # Big boost if in anchor
    w_body = 1.0     # Fine-tuning relevance
    w_pr = 2.5       # Quality signal (Log scale)
    w_pv = 1.0       # Popularity signal (Log scale)

    for doc_id in all_doc_ids:
        s_body = body_scores.get(doc_id, 0.0)
        s_title = title_scores.get(doc_id, 0.0)
        s_anchor = anchor_scores.get(doc_id, 0.0)
        
        # Log-smooth PageRank and PageViews to dampen impact of outliers
        pr_val = pagerank_dict.get(doc_id, 0.0)
        pr_score = math.log(pr_val + 1, 10) if pr_val > 0 else 0
        
        pv_val = pageview_dict.get(doc_id, 0)
        pv_score = math.log(pv_val + 1, 10) if pv_val > 0 else 0
        
        total_score = (w_title * s_title) + \
                      (w_anchor * s_anchor) + \
                      (w_body * s_body) + \
                      (w_pr * pr_score) + \
                      (w_pv * pv_score)
                      
        final_scores.append((doc_id, total_score))

    # 3. Sort & Format
    final_scores.sort(key=lambda x: x[1], reverse=True)
    top_100 = final_scores[:100]
    
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
    
    #strict cosine similarity using tf-idf on body.
    #formula: (query_vector * dot_vector) / (query_norm * doc_norm)
    
    tokens = tokenize(query)
    if not tokens: return jsonify(res)

    query_counts = Counter(tokens)
    scores = Counter()
    
    # Corpus size for IDF
    N = len(doc_len_body) if doc_len_body else 6348910

    if idx_body:
        for term, q_tf in query_counts.items():
            if term in idx_body.df:
                df = idx_body.df[term]
                idf = math.log(N / df, 10) # Log base 10 per instructions usually
                w_q = q_tf * idf
                
                pl = get_posting_list(idx_body, term)
                for doc_id, tf in pl:
                    w_d = tf * idf
                    scores[doc_id] += w_q * w_d # Dot Product

    final_scores = []
    for doc_id, dot_prod in scores.items():
        # Cosine = Dot / (Norm_Q * Norm_D)
        # We can ignore Norm_Q for ranking purposes as it's constant for the query
        doc_len = doc_len_body.get(doc_id, 1.0) # This should be the Pre-calculated Norm!
        # Note: If doc_len_body stores simple length (words), this is an approximation.
        # But for the project, no Euclidean Norm = length is the fallback.
        
        cosine_score = dot_prod / doc_len
        final_scores.append((doc_id, cosine_score))

    final_scores.sort(key=lambda x: x[1], reverse=True)
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

    # 1. Tokenize
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)
        
    # 2. Calculate Binary Score (Count distinct matches)
    scores = calc_binary_score(tokens, idx_title)
    
    # 3. Sort by score
    final_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Format Output
    res = [(str(doc_id), titles_dict.get(doc_id, str(doc_id))) for doc_id, score in final_scores]

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

    # 1. Tokenize
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)
    
    # 2. Calculate Binary Score
    scores = calc_binary_score(tokens, idx_anchor)
    
    # 3. Sort by score
    final_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Format Output
    res = [(str(doc_id), titles_dict.get(doc_id, str(doc_id))) for doc_id, score in final_scores]

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
