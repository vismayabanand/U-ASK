import psycopg2
import heapq
import time
import pandas as pd
import numpy as np
from collections import Counter
from geopy.distance import geodesic
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# **Connect to PostgreSQL**
try:
    pg_conn = psycopg2.connect(
        dbname="tweet_data",
        user="postgres",
        password="Ranjitha@14",  
        host="localhost",
        port="5432"
    )
    pg_cursor = pg_conn.cursor()
    print(" Connected to PostgreSQL successfully!")
except psycopg2.OperationalError as e:
    print(" PostgreSQL connection failed. Check credentials and try again.")
    print(f"Error: {e}")
    exit(1)

# **Compute Cosine Similarity for Textual Match**
def compute_textual_similarity(query_keywords, doc_keywords):
    if not doc_keywords or len(doc_keywords) == 0:
        return 0.0  

    query_vector = Counter(query_keywords)
    doc_vector = Counter(doc_keywords)
    all_words = list(set(query_vector.keys()).union(set(doc_vector.keys())))

    query_vec = np.array([query_vector[word] for word in all_words]).reshape(1, -1)
    doc_vec = np.array([doc_vector[word] for word in all_words]).reshape(1, -1)

    return cosine_similarity(query_vec, doc_vec)[0][0]

# **Fetch Indexed Data Using R-Tree for Spatial Filtering & Negative Keyword Filtering**
def fetch_spatial_candidates(query_location, spatial_range, negative_keywords):
    print("\nFetching spatially relevant tweets using R-Tree filtering...")

    # Retrieve tweets within the spatial range using PostGIS
    pg_cursor.execute("""
        SELECT tweet_id, latitude, longitude, keywords 
        FROM tweets 
        WHERE ST_DWithin(
            ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
            %s
        );
    """, (query_location[1], query_location[0], spatial_range))

    data = pg_cursor.fetchall()
    
    candidates = []
    for row in data:
        keywords = row[3]

        if keywords is None:
            keywords = []
        elif isinstance(keywords, str):
            keywords = [word.strip() for word in keywords.split(",") if word.strip()]

        # **Apply Negative Keyword Filtering**
        if any(neg_kw in keywords for neg_kw in negative_keywords):
            continue  # Exclude tweets with negative keywords

        candidates.append({
            "tweet_id": row[0],
            "latitude": row[1],
            "longitude": row[2],
            "keywords": keywords
        })

    print(f" Retrieved {len(candidates)} spatial candidates after filtering.\n")
    return candidates

# **RCA Algorithm: kNN Search with Spatial + Text Filtering**
def rca_knn_query(query_location, query_keywords, negative_keywords, spatial_range, k=10):
    """
    RCA kNN Query with Negative Keyword Filtering and Evaluation.
    """
    start_time = time.time()  # Start execution time tracking

    spatial_candidates = fetch_spatial_candidates(query_location, spatial_range, negative_keywords)

    print(f" Running RCA kNN Query on {len(spatial_candidates)} candidates...")

    min_heap = []
    processed_docs = 0
    lambda_value = 0.5  # RCA balances textual similarity & spatial distance

    def process_document(doc):
        nonlocal processed_docs
        doc_location = (doc["latitude"], doc["longitude"])
        doc_keywords = set(doc["keywords"])

        # Compute spatial distance
        spatial_distance = geodesic(query_location, doc_location).kilometers

        # Compute textual similarity
        textual_similarity = compute_textual_similarity(query_keywords, doc_keywords)

        # Compute combined score
        combined_score = (lambda_value * textual_similarity) + ((1 - lambda_value) * (1 / max(spatial_distance, 1e-3)))

        processed_docs += 1
        return (combined_score, doc["tweet_id"], doc["longitude"], doc["latitude"])

    # **Parallel Processing**
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_document, spatial_candidates))

    # **Efficient Min-Heap Merging**
    for result in results:
        if result:
            heapq.heappush(min_heap, result)
            if len(min_heap) > k:
                heapq.heappop(min_heap)

    end_time = time.time()
    execution_time = round(end_time - start_time, 3)

    # **Final Sorted Results (Optional: Sort by Score)**
    top_k_results = sorted([heapq.heappop(min_heap) for _ in range(len(min_heap))], key=lambda x: x[0], reverse=True)
    retrieved_tweet_ids = {tweet[1] for tweet in top_k_results}

    # **Prepare Evaluation Summary**
    evaluation_summary = pd.DataFrame({
        "Metric": ["Execution Time (s)", "Processed Documents"],
        "Value": [execution_time, processed_docs]
    })

    # **Print Evaluation Results**
    print("\n🔹 RCA kNN Query Execution Metrics:")
    print(evaluation_summary.to_string(index=False))

    return top_k_results

# **Execute RCA Query with Negative Keyword Filtering**
if __name__ == "__main__":
    query_location = (34.0522, -118.2437)  # Los Angeles, CA
    query_keywords = ["beauty"]  # Required keywords
    negative_keywords = ["boring"]  # Exclude tweets with these words
    spatial_range = 5000  # 5km radius

    top_k_results = rca_knn_query(query_location, query_keywords, negative_keywords, spatial_range, k=15)

    print("\n🔹 **Top-k Results from RCA kNN Query with Negative Filtering:**")
    for rank, (score, tweet_id, lon, lat) in enumerate(top_k_results, 1):
        print(f"{rank}. Tweet ID: {tweet_id}, Location: ({lat}, {lon}), Score: {score:.4f}")
