import heapq
import json
import time
import re
import numpy as np
import pandas as pd
from collections import Counter
from geopy.distance import geodesic
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# **Load TEQ Index from JSON**
try:
    with open("quadtree_index.json", "r") as f:
        location_table = json.load(f)
    with open("textual_index.json", "r") as f:
        textual_index = json.load(f)

    print(f" TEQ Index Loaded Successfully!")
except Exception as e:
    print(f" Failed to load TEQ index: {e}")
    exit(1)

# **Compute Jaccard Similarity Between Two Result Sets**
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# **Compute Textual Similarity**
def compute_textual_similarity(query_keywords, doc_keywords):
    if not doc_keywords or len(doc_keywords) == 0:
        return 0.0
    if not any(qk in doc_keywords for qk in query_keywords):
        return 0.0  

    query_vector = Counter(query_keywords)
    doc_vector = Counter(doc_keywords)
    all_words = list(set(query_vector.keys()).union(set(doc_vector.keys())))

    query_vec = np.array([query_vector[word] for word in all_words]).reshape(1, -1)
    doc_vec = np.array([doc_vector[word] for word in all_words]).reshape(1, -1)

    return cosine_similarity(query_vec, doc_vec)[0][0]

# **POWER Algorithm Without Relevant Tweets**
def power_algorithm(query_location, query_keywords, negative_keywords, k=10):
    """
    Implements POWER Algorithm with execution time and ranking analysis.
    """
    start_time = time.time()  # Track execution time

    # Fetch candidate tweets
    candidate_tweet_ids = set()
    for keyword in query_keywords:
        if keyword in textual_index:
            candidate_tweet_ids.update(tweet_id for tweet_id, _ in textual_index[keyword])

    # Negative keyword filtering
    filtered_tweet_ids = []
    for tweet_id in candidate_tweet_ids:
        tweet_keywords = set(kw for kw, _ in textual_index.get(str(tweet_id), []))
        if not any(re.search(rf"\b{re.escape(neg_kw)}\b", " ".join(tweet_keywords), re.IGNORECASE) for neg_kw in negative_keywords):
            filtered_tweet_ids.append(tweet_id)

    print(f" Starting POWER Algorithm on {len(filtered_tweet_ids)} tweets...")

    # **Ranking Tweets**
    min_heap = []
    processed_docs = 0
    lambda_value = 0.5  # Balance factor for textual vs. spatial scoring

    # **Parallel Processing**
    def process_document(tweet_id):
        nonlocal processed_docs
        if str(tweet_id) not in location_table:
            return None

        lon, lat = location_table[str(tweet_id)]
        doc_location = (lat, lon)
        if lon == 0.0 and lat == 0.0:
            return None  

        doc_keywords = set(kw for kw, _ in textual_index.get(str(tweet_id), []))
        spatial_distance = geodesic(query_location, doc_location).kilometers
        textual_similarity = compute_textual_similarity(query_keywords, doc_keywords)

        combined_score = (lambda_value * textual_similarity) + ((1 - lambda_value) * (1 / max(spatial_distance, 1e-3)))

        processed_docs += 1
        return (combined_score, tweet_id, lon, lat)

    # **Execute in Parallel**
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_document, filtered_tweet_ids))

    for result in results:
        if result:
            heapq.heappush(min_heap, result)
            if len(min_heap) > k:
                heapq.heappop(min_heap)

    end_time = time.time()
    execution_time = round(end_time - start_time, 3)

    # **Final Sorted Results**
    top_k_results = [heapq.heappop(min_heap) for _ in range(len(min_heap))][::-1]
    retrieved_tweet_ids = [tweet[1] for tweet in top_k_results]

    # **Prepare Evaluation Summary**
    evaluation_summary = pd.DataFrame({
        "Metric": ["Execution Time (s)"],
        "Value": [execution_time]
    })

    # **Print Evaluation Results**
    print("\n🔹 POWER Algorithm Execution Metrics:")
    print(evaluation_summary.to_string(index=False))

    return top_k_results

# **Execute POWER Algorithm**
if __name__ == "__main__":
    query_location = (34.0522, -118.2437)  # Los Angeles, CA
    query_keywords = ["beauty"]
    negative_keywords = ["boring"]
    top_k_results = power_algorithm(query_location, query_keywords, negative_keywords, k=15)

    print("\n **Top-k Results from POWER Algorithm:**")
    for rank, (score, tweet_id, lon, lat) in enumerate(top_k_results, 1):
        print(f"{rank}.  Tweet ID: {tweet_id},  Location: ({lat}, {lon}),  Score: {score:.4f}")
