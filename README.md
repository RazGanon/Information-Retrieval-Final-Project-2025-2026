# Wikipedia Search Engine 🔎

> **Course:** Information Retrieval, Ben-Gurion University of the Negev (2025-2026)  
> **Authors:** 208894444, [Partner ID]

## 📖 Overview

This project is a scalable, efficient search engine capable of retrieving and ranking documents from the entire English Wikipedia corpus (over 6 million articles). The system is built using **Python**, **Flask**, and **Google Cloud Platform (GCP)**. It utilizes **PySpark** for offline index construction and efficient inverted index structures for real-time retrieval.

The engine supports multi-signal ranking, combining TF-IDF/BM25 scores from document bodies, titles, and anchor text with link analysis (PageRank) and popularity metrics (PageViews).

## 🚀 Features

* **Sub-second Latency:** Optimized for fast retrieval using memory-mapped posting lists and lightweight in-memory data structures.
* **Ensemble Ranking:** A weighted scoring system that combines:
    * **BM25** (Body, Title, and Anchor text).
    * **PageRank** (Link authority).
    * **PageViews** (Traffic popularity).
* **Specialized Search Routes:**
    * Main Search (Optimized ensemble).
    * Body Search (Cosine Similarity/TF-IDF).
    * Title Search (Binary/Distinct ranking).
    * Anchor Search (Binary/Distinct ranking).
* **Cloud Deployment:** Fully configured to run on GCP Compute Engine.

## 🛠 Architecture

The system operates in two distinct phases:

1.  **Offline Indexing (MapReduce/Spark):**
    * Processing the Wikipedia dump (Parquet files).
    * Tokenization, Stopword removal, and Stemming (where applicable).
    * Calculation of TF-IDF, Document Lengths, and Inverted Indices.
    * Generation of `PageRank` and `PageViews` statistics.
    * **Output:** Binary posting files (`.bin`) and metadata pickles (`.pkl`).

2.  **Online Retrieval (Flask App):**
    * Loads indices and statistics into memory at startup.
    * Parses user queries.
    * Retrieves candidate documents from posting lists.
    * Computes scores and returns ranked results via a RESTful API.

### 🧠 Ranking Logic (The "Ensemble")
The main search endpoint (`/search`) uses a linear combination of normalized scores:

$$Score(q, d) = W_{body} \cdot BM25(q, d_{body}) + W_{title} \cdot BM25(q, d_{title}) + W_{anchor} \cdot BM25(q, d_{anchor}) + W_{pr} \cdot \log_{10}(PageRank(d))$$

## 📂 Repository Structure

```text
.
├── search_frontend.py       # 🚀 Main Flask application (Entry Point)
├── inverted_index_gcp.py    # 📚 Class for reading/writing Inverted Indices
├── startup_script_gcp.sh    # ☁️ Script for initializing GCP VM environment
├── run_frontend_in_gcp.sh   # ☁️ Commands to deploy the engine
├── requirements.txt         # 📦 Python dependencies
├── postings/                # 📂 Directory containing binary posting files
│   ├── title/               # Title index .bin files
│   ├── anchor/              # Anchor index .bin files
│   └── body/                # Body index .bin files
├── index_title.pkl          # Index metadata (Title)
├── index_anchor.pkl         # Index metadata (Anchor)
├── index_body.pkl           # Index metadata (Body)
├── pagerank.pkl             # Dict: {doc_id: rank_score}
├── pageviews.pkl            # Dict: {doc_id: monthly_views}
├── titles.pkl               # Dict: {doc_id: title_string}
└── bm25_stats.pkl           # Dict: {avg_dl, doc_lengths}

```
# ⚙️ Installation & Setup
Prerequisites

    Python 3.8+

    Approximately 8GB-16GB RAM (depending on index size).

    Google Cloud SDK (if deploying to GCP).

Local Setup

    Clone the repository using "https://github.com/RazGanon/Information-Retrieval-Final-Project-2025-2026.git"
    
Install dependencies:

    pip install -r requirements.txt
    python -m nltk.downloader stopwords

Place Data Files: Ensure all generated .pkl files and the postings/ folder are in the root directory.

Run the Server using 

    python search_frontend.py

🔌 API Reference
1. Main Search

Returns the top 100 relevant results using the full ranking ensemble.

    URL: /search

    Method: GET

    Params: query (string)

    Example: http://localhost:8080/search?query=machine+learning

2. Search Body

Returns top 100 results based on Cosine Similarity/BM25 of the article body.

    URL: /search_body

    Method: GET

    Params: query (string)

3. Search Title

Returns all results where query terms appear in the title, ranked by the count of distinct query words.

    URL: /search_title

    Method: GET

    Params: query (string)

4. Search Anchor

Returns all results where query terms appear in anchor text pointing to the page, ranked by distinct query words.

    URL: /search_anchor

    Method: GET

    Params: query (string)

5. Get PageRank / PageView

Helper endpoints to fetch pre-computed statistics.

    URL: /get_pagerank OR /get_pageview

    Method: POST

    Body: JSON list of integers [12, 453, 991]

    Response: JSON list of values.

☁️ GCP Deployment

To deploy on Google Cloud Platform:

    Create a VM instance (e.g., n1-standard-4).

    Reserve a static external IP address.

    Upload the code and data to the instance.

    Run the provided startup script: "sudo bash startup_script_gcp.sh"

    Access the engine via your external IP on port 8080.

📊 Evaluation Metrics

The engine was optimized and evaluated using the following criteria:

    Latency: Average query processing time < 35 seconds (optimized to < 1s).

    Quality: Precision@10 > 0.1 on the test set.

    Efficiency: Efficient memory management (loading only necessary data structures).
