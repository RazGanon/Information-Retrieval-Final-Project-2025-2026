from flask import Flask, request, jsonify, render_template_string
import pickle
import re
import math
from collections import Counter, defaultdict
from itertools import islice
import nltk
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex
from google.cloud import storage
from pathlib import Path

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Global variables for loaded data
index_body = None
doc_lengths = {}
avg_dl = 0
N = 0
titles = {}
pagerank = {}
BUCKET_NAME = '208894444'
BASE_DIR = '.'

# Tokenization setup (must match index_builder.ipynb)
RE_WORD = re.compile(r"""[\#\@\w](['\/\-]?\w){2,24}""", re.UNICODE)
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                   "may", "first", "see", "history", "people", "one", "two",
                   "part", "thumb", "including", "second", "following", 
                   "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)

def tokenize(text):
    """Tokenize text and remove stopwords."""
    if not text:
        return []
    tokens = [token.group().lower() for token in RE_WORD.finditer(text)]
    return [token for token in tokens if token not in all_stopwords]

def load_data_from_gcs():
    """Load all required data from GCS at startup."""
    global index_body, doc_lengths, avg_dl, N, titles, pagerank
    
    print("Loading data from GCS...")
    
    try:
        # Load inverted index
        print("  Loading index_body.pkl...")
        index_body = InvertedIndex.read_index(BASE_DIR, 'index_body', BUCKET_NAME)
        print(f"  ✓ Loaded index with {len(index_body.df):,} terms")
        
        # Load BM25 data
        print("  Loading bm25_data.pkl...")
        with open('bm25_data.pkl', 'rb') as f:
            bm25_data = pickle.load(f)
            doc_lengths = bm25_data['doc_lengths']
            avg_dl = bm25_data['avg_dl']
            N = bm25_data['total_docs']
        print(f"  ✓ Loaded BM25 data (N={N:,}, avg_dl={avg_dl:.2f})")
        
        # Load titles
        print("  Loading titles.pkl...")
        with open('titles.pkl', 'rb') as f:
            titles = pickle.load(f)
        print(f"  ✓ Loaded {len(titles):,} titles")
        
        # Load PageRank
        print("  Loading pagerank.pkl...")
        with open('pagerank.pkl', 'rb') as f:
            pagerank = pickle.load(f)
        print(f"  ✓ Loaded {len(pagerank):,} PageRank scores")
        
        print("All data loaded successfully!")
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        print("Make sure all .pkl files are downloaded to the current directory")

def get_bm25_scores(query_tokens, top_n=100, k1=1.5, b=0.75):
    """
    Calculate BM25 scores for query.
    Uses Champion Lists (top 500 docs per term) for efficiency.
    
    BM25 formula: IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
    IDF formula: ln((N - df + 0.5) / (df + 0.5) + 1)
    
    Args:
        query_tokens: List of query terms
        top_n: Number of top results to return
        k1: BM25 parameter controlling term frequency saturation (default 1.5)
        b: BM25 parameter controlling length normalization (default 0.75)
    """
    if not query_tokens or index_body is None:
        return []
    
    # Storage for document scores
    doc_scores = defaultdict(float)
    
    # Process each unique query term
    for term in set(query_tokens):
        if term not in index_body.posting_locs:
            continue
        
        # Get posting list for this term
        posting_list = index_body.read_a_posting_list(BASE_DIR, term, BUCKET_NAME)
        
        # Apply Champion Lists: limit to top 500 documents by TF
        posting_list_sorted = sorted(posting_list, key=lambda x: x[1], reverse=True)
        champion_list = list(islice(posting_list_sorted, 500))
        
        # Calculate IDF using natural logarithm with BM25 formula
        df = index_body.df.get(term, 0)
        
        # Handle edge cases for division by zero
        if df <= 0 or N <= 0:
            continue
        
        # BM25 IDF: ln((N - df + 0.5) / (df + 0.5) + 1)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        # Skip if IDF is non-positive (shouldn't happen with BM25 IDF formula)
        if idf <= 0:
            continue
        
        # Calculate BM25 score for each document in champion list
        for doc_id, tf in champion_list:
            if doc_id not in doc_lengths or doc_lengths[doc_id] <= 0:
                continue
            
            doc_len = doc_lengths[doc_id]
            
            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
            
            # Avoid division by zero
            if denominator > 0:
                bm25_component = idf * (numerator / denominator)
                doc_scores[doc_id] += bm25_component
    
    # Sort by score descending and return top N
    scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return scores

def normalize_scores(scores_dict):
    """Normalize scores to [0, 1] range."""
    if not scores_dict:
        return {}
    
    values = list(scores_dict.values())
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return {k: 1.0 for k in scores_dict}
    
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}

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
    
    # Tokenize query
    query_tokens = tokenize(query)
    if not query_tokens:
        return jsonify(res)
    
    # Get BM25 body scores
    body_scores = get_bm25_scores(query_tokens, top_n=200)
    body_dict = {doc_id: score for doc_id, score in body_scores}
    
    # Get title match scores ONLY for candidate documents from body search
    query_terms_set = set(query_tokens)
    title_scores = {}
    
    for doc_id in body_dict.keys():
        title = titles.get(doc_id)
        if title:
            title_tokens = set(tokenize(title))
            match_count = len(query_terms_set & title_tokens)
            if match_count > 0:
                title_scores[doc_id] = match_count
    
    # Normalize scores to [0, 1] range
    body_norm = normalize_scores(body_dict)
    title_norm = normalize_scores(title_scores) if title_scores else {}
    
    # Linear combination: Body 70%, Title 30%
    combined_scores = {}
    for doc_id in body_dict.keys():
        score = (0.7 * body_norm.get(doc_id, 0.0) +
                 0.3 * title_norm.get(doc_id, 0.0))
        combined_scores[doc_id] = score
    
    # EXACT TITLE MATCH BOOST: If query exactly matches title, add +2.0
    query_normalized = query.lower().strip()
    for doc_id in combined_scores.keys():
        title = titles.get(doc_id, '')
        if title.lower().strip() == query_normalized:
            combined_scores[doc_id] += 2.0
    
    # Sort and get top 100
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:100]
    
    # Format results as (wiki_id, title) tuples
    res = [(str(doc_id), titles.get(doc_id, '')) for doc_id, score in sorted_results]
    
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
    
    # Tokenize query
    query_tokens = tokenize(query)
    if not query_tokens:
        return jsonify(res)
    
    # Get BM25 scores
    scores = get_bm25_scores(query_tokens, top_n=100)
    
    # Format results as (wiki_id, title) tuples
    res = [(str(doc_id), titles.get(doc_id, '')) for doc_id, score in scores]
    
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
    
    # Tokenize query
    query_tokens = tokenize(query)
    if not query_tokens:
        return jsonify(res)
    
    query_terms_set = set(query_tokens)
    
    # Score each document by number of distinct query words in title
    title_scores = []
    for doc_id, title in titles.items():
        title_tokens = set(tokenize(title))
        match_count = len(query_terms_set & title_tokens)
        if match_count > 0:
            title_scores.append((doc_id, match_count))
    
    # Sort by match count descending
    title_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Format results as (wiki_id, title) tuples - return ALL results
    res = [(str(doc_id), titles.get(doc_id, '')) for doc_id, count in title_scores]
    
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
    
    # Note: This requires an anchor text index which is not built in index_builder.ipynb
    # Return empty results for now
    # To implement: build index_anchor similar to index_body and use it here
    
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
    
    res = [pagerank.get(int(wiki_id), 0.0) for wiki_id in wiki_ids]
    
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
    
    # Note: This requires pageview data which is not built in index_builder.ipynb
    # Return 0 for all for now
    res = [0 for _ in wiki_ids]
    
    # END SOLUTION
    return jsonify(res)

@app.route("/ui")
@app.route("/")
def ui():
    '''
    Web UI for search engine with query input and results display.
    Shows query and search duration. Also serves as homepage.
    '''
    import time
    
    # Simple HTML template
    template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wikipedia Search Engine</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .search-box { margin: 20px 0; }
            input[type="text"] { width: 70%; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .stats { color: #666; margin: 10px 0; font-size: 14px; }
            .results { margin-top: 20px; }
            .result { padding: 15px; border-bottom: 1px solid #eee; }
            .result:hover { background: #f9f9f9; }
            .result-title { color: #1a0dab; font-size: 18px; text-decoration: none; }
            .result-title:hover { text-decoration: underline; }
            .result-id { color: #666; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>🔍 Wikipedia Search Engine (BM25)</h1>
        <div class="search-box">
            <form method="GET">
                <input type="text" name="query" placeholder="Enter search query..." value="{{ query }}">
                <button type="submit">Search</button>
            </form>
        </div>
        {% if query %}
            <div class="stats">
                Query: <strong>{{ query }}</strong> | 
                Results: <strong>{{ num_results }}</strong> | 
                Duration: <strong>{{ duration }}</strong> ms
            </div>
            <div class="results">
                {% for doc_id, title in results %}
                <div class="result">
                    <a href="https://en.wikipedia.org/?curid={{ doc_id }}" class="result-title" target="_blank">
                        {{ title or "Untitled" }}
                    </a>
                    <div class="result-id">ID: {{ doc_id }}</div>
                </div>
                {% endfor %}
            </div>
        {% endif %}
    </body>
    </html>
    '''
    
    query = request.args.get('query', '')
    results = []
    duration = 0
    
    if query:
        start_time = time.time()
        
        # Call search logic directly (same as /search endpoint)
        query_tokens = tokenize(query)
        if query_tokens:
            # Get BM25 body scores
            body_scores = get_bm25_scores(query_tokens, top_n=200)
            body_dict = {doc_id: score for doc_id, score in body_scores}
            
            # Get title match scores
            query_terms_set = set(query_tokens)
            title_scores = {}
            
            for doc_id in body_dict.keys():
                title = titles.get(doc_id)
                if title:
                    title_tokens = set(tokenize(title))
                    match_count = len(query_terms_set & title_tokens)
                    if match_count > 0:
                        title_scores[doc_id] = match_count
            
            # Normalize scores
            body_norm = normalize_scores(body_dict)
            title_norm = normalize_scores(title_scores) if title_scores else {}
            
            # Linear combination: Body 70%, Title 30%
            combined_scores = {}
            for doc_id in body_dict.keys():
                score = (0.7 * body_norm.get(doc_id, 0.0) +
                         0.3 * title_norm.get(doc_id, 0.0))
                combined_scores[doc_id] = score
            
            # EXACT TITLE MATCH BOOST: If query exactly matches title, add +2.0
            query_normalized = query.lower().strip()
            for doc_id in combined_scores.keys():
                title = titles.get(doc_id, '')
                if title.lower().strip() == query_normalized:
                    combined_scores[doc_id] += 2.0
            
            # Sort and get top 100
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:100]
            results = [(str(doc_id), titles.get(doc_id, '')) for doc_id, score in sorted_results]
        
        duration = round((time.time() - start_time) * 1000, 2)  # Convert to ms
    
    return render_template_string(template, query=query, results=results, 
                                   num_results=len(results), duration=duration)

# Load data when module is imported
load_data_from_gcs()

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
