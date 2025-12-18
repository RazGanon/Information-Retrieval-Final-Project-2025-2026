# Information-Retrieval-Final-Project-2025-2026
High-performance search engine designed to index and search the entire English Wikipedia corpus. It utilizes core Information Retrieval concepts to provide relevant results quickly.

Features

    Full-Text Search: Efficient querying across the entire English Wikipedia dataset.

    Tokenization & Preprocessing: Includes stemming, stop-word removal, and case folding.

    Inverted Index: A custom-built inverted index for fast document retrieval.

    Ranking: Uses the TF-IDF (Term Frequency-Inverse Document Frequency) scoring model or BM25 to rank results by relevance.

    Scalability: Designed to handle large-scale XML/JSON dumps from Wikipedia.

    Architecture

The system is divided into three main phases:

    Parsing: Extracting clean text from the Wikipedia XML/JSON dump.

    Indexing: Building a compressed inverted index stored on disk.

    Querying: A search interface that processes user input and returns the top k ranked documents.
