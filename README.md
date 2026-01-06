Wikipedia Search Engine (Information Retrieval Project)

This repository contains the implementation of a scalable search engine capable of retrieving and ranking documents from the entire English Wikipedia corpus (over 6 million articles). The system is built using Python, Flask, and Google Cloud Platform (GCP), utilizing PySpark for indexing and efficient inverted index structures for retrieval.
🚀 Features

    Sub-second Query Latency: Optimized for fast retrieval using inverted indices and memory mapping.

    Multi-Signal Ranking: Uses an ensemble of ranking methods including:

        BM25 (Body, Title, and Anchor text).

        PageRank (Link analysis for authority).

        PageViews (Traffic analysis for popularity).

    Ranking Refinement: Implements a linear combination of scores with log-normalization for PageRank.

    Specialized Search Routes: Dedicated API endpoints for title search, anchor text search, and body search.

    Cloud Ready: Designed to run on GCP Compute Engine instances.

🛠 Architecture

The project consists of two main phases:

    Index Construction (Offline): Large-scale processing of the Wikipedia dump using Apache Spark (Dataproc). This phase produces inverted indices (posting lists), TF-IDF statistics, and document metadata.

    Retrieval Engine (Online): A Flask web server that loads the compressed indices and serves search queries via a RESTful API.

Search Logic (The "Ensemble")

The main search algorithm (/search) calculates a composite score S(q,d) for a query q and document d:

$$ S(q, d) = w_{body} \cdot BM25_{body} + w_{title} \cdot BM25_{title} + w_{anchor} \cdot BM25_{anchor} + w_{pr} \cdot \log_{10}(PageRank(d)) $$

Current Weights: Body (0.35), Title (0.45), Anchor (0.20), PageRank (1.5).
📂 Repository Structure
Plaintext
