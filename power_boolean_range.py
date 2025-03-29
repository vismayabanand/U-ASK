import psycopg2
import re
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from geopy.distance import geodesic

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
    print("Connected to PostgreSQL successfully!")
except psycopg2.OperationalError as e:
    print("PostgreSQL connection failed. Check credentials and try again.")
    print(f"Error: {e}")
    exit(1)

# **Compute Jaccard Similarity Between Two Result Sets**
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# **Fetch Indexed Data with Strict Boolean Filtering**
def fetch_power_boolean_data(query_location, spatial_range, query_keywords, negative_keywords):
    print("\nFetching indexed data with spatial and strict Boolean filtering...")
    
    # PostgreSQL Query with PostGIS Spatial Filtering
    pg_cursor.execute("""
        SELECT tweet_id, latitude, longitude, keywords 
        FROM tweets 
        WHERE ST_DistanceSphere(ST_MakePoint(longitude, latitude), ST_MakePoint(%s, %s)) <= %s;
    """, (query_location[1], query_location[0], spatial_range))
    
    data = pg_cursor.fetchall()
    
    indexed_data = []
    for row in data:
        keywords = row[3]
        if keywords is None:
            keywords = []
        elif isinstance(keywords, str):
            keywords = [word.strip() for word in keywords.split(",") if word.strip()]
        
        # **Strict Boolean Filtering**
        if not any(kw in keywords for kw in query_keywords):
            continue  # Skip if no required keyword is found
        
        if any(re.search(rf"\b{re.escape(phrase)}\b", " ".join(keywords), re.IGNORECASE) for phrase in negative_keywords):
            continue  # Skip if any negative keyword is found
        
        indexed_data.append({
            "tweet_id": row[0],
            "latitude": row[1],
            "longitude": row[2],
            "keywords": keywords
        })
    
    print(f"Retrieved {len(indexed_data)} documents after strict Boolean filtering.\n")
    return indexed_data

# **Power Boolean Range Query with Evaluation**
def power_boolean_range_query(query_location, query_keywords, negative_keywords, spatial_range):
    """
    Implements Power Boolean Range Query with evaluation.
    """
    indexed_data = fetch_power_boolean_data(query_location, spatial_range, query_keywords, negative_keywords)
    print(f"Running Power Boolean Range Query on {len(indexed_data)} documents...")
    start_time = time.time()
    
    filtered_results = []
    processed_docs = 0
    
    def process_document(doc):
        nonlocal processed_docs
        doc_location = (doc["latitude"], doc["longitude"])
        spatial_distance = geodesic(query_location, doc_location).kilometers
        
        processed_docs += 1
        if processed_docs % 5000 == 0:
            print(f"Processed {processed_docs} / {len(indexed_data)} documents...")
        
        return {
            "tweet_id": doc["tweet_id"],
            "latitude": doc["latitude"],
            "longitude": doc["longitude"],
            "distance_km": spatial_distance
        }
    
    # **Parallel Processing**
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_document, indexed_data))
    
    filtered_results.extend(results)
    
    end_time = time.time()
    execution_time = round(end_time - start_time, 3)

    # **Final Sorted Results (Optional: Sort by Distance)**
    sorted_results = sorted(filtered_results, key=lambda x: x["distance_km"])[:10]
    retrieved_tweet_ids = {doc["tweet_id"] for doc in sorted_results}

    # **Jaccard Similarity (Optional: Compare with RCA or POWER Results)**
    # Example: If you have another set of results (e.g., `power_results`)
    # jaccard_rca_boolean = jaccard_similarity(set(retrieved_tweet_ids), set(power_results))

    # **Prepare Evaluation Summary**
    evaluation_summary = pd.DataFrame({
        "Metric": ["Execution Time (s)", "Processed Documents"],
        "Value": [execution_time, processed_docs]
    })

    # **Print Evaluation Results**
    print("\n🔹 Power Boolean Range Query Execution Metrics:")
    print(evaluation_summary.to_string(index=False))

    return sorted_results

# **Execute Query**
if __name__ == "__main__":
    query_location = (34.0522, -118.2437)  # Los Angeles, CA
    query_keywords = ["beauty"]  # Required keywords
    negative_keywords = ["boring"]  # Exclude these keywords
    spatial_range = 5000  # 5km radius

    results = power_boolean_range_query(query_location, query_keywords, negative_keywords, spatial_range)
    
    print("\n🔹 **Power Boolean Range Query Results:**")
    for doc in results:
        print(f"Tweet ID: {doc['tweet_id']}, Location: ({doc['latitude']}, {doc['longitude']}), Distance: {doc['distance_km']:.2f} km")
