import json
import psycopg2
import logging

# **Database Connection**
try:
    conn = psycopg2.connect(
        dbname="tweet_data",
        user="postgres",
        password="Ranjitha@14",  # Replace with your actual password
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    print("Connected to PostgreSQL successfully!")
except psycopg2.OperationalError as e:
    print("PostgreSQL connection failed. Check credentials and try again.")
    print(f"Error: {e}")
    exit(1)

# **Load TEQ Indexed Data**
try:
    with open("quadtree_index.json", "r") as f:
        location_data = json.load(f)  # { tweet_id: (longitude, latitude) }

    with open("textual_index.json", "r") as f:
        textual_data = json.load(f)  # { keyword: [(tweet_id, weight), ...] }

    print(f"Loaded TEQ Indexed Data - {len(location_data)} tweets")
except Exception as e:
    print(f"Error loading TEQ index: {e}")
    exit(1)

# **Transform Textual Data for Faster Lookup**
tweet_keyword_map = {}

for keyword, entries in textual_data.items():
    for tweet_id, weight in entries:  # Extract tweet_id and weight
        if tweet_id not in tweet_keyword_map:
            tweet_keyword_map[tweet_id] = {"keywords": [], "weights": []}
        tweet_keyword_map[tweet_id]["keywords"].append(keyword.strip())
        tweet_keyword_map[tweet_id]["weights"].append(weight)  # Store the weight

print(f"Transformed Textual Index for {len(tweet_keyword_map)} tweets.")

# **Prepare Data for Insertion**
tweet_records = []

for tweet_id, (longitude, latitude) in location_data.items():
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        logging.warning(f"Skipping invalid location: {latitude}, {longitude} (Tweet ID: {tweet_id})")
        continue  # Ignore invalid coordinates

    # Get keywords and weights
    keywords = tweet_keyword_map.get(int(tweet_id), {"keywords": [], "weights": []})
    keywords_array = "{" + ",".join(keywords["keywords"]) + "}"  # Format as PostgreSQL TEXT array
    weights_array = "{" + ",".join(map(str, keywords["weights"])) + "}"  # Format as PostgreSQL FLOAT array

    tweet_records.append((tweet_id, latitude, longitude, keywords_array, weights_array))

print(f"Prepared {len(tweet_records)} tweets for PostgreSQL insertion.")

# **Update SQL Query to Store Keywords & Keyword Weights as Arrays**
insert_query = """
    INSERT INTO tweets (tweet_id, latitude, longitude, keywords, keyword_weights)
    VALUES (%s, %s, %s, %s::TEXT[], %s::FLOAT[])
    ON CONFLICT (tweet_id) DO UPDATE 
    SET latitude = EXCLUDED.latitude,
        longitude = EXCLUDED.longitude,
        keywords = EXCLUDED.keywords,
        keyword_weights = EXCLUDED.keyword_weights;
"""

BATCH_SIZE = 5000  # Increased batch size for faster insertion
count = 0

for i in range(0, len(tweet_records), BATCH_SIZE):
    batch = tweet_records[i:i+BATCH_SIZE]
    cursor.executemany(insert_query, batch)
    conn.commit()
    count += len(batch)
    print(f"Inserted {count} tweets...")

print(f"**Successfully Inserted {count} tweets into PostgreSQL!**")

# **Close Connection**
cursor.close()
conn.close()
