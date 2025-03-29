import csv
import json
import logging
import sys

# Increase recursion limit for large dataset handling
sys.setrecursionlimit(5000)

# Configure logging
logging.basicConfig(level=logging.INFO)

### ** Define Spatial Indexing Components (Quadtree + BoundingBox)**
class BoundingBox:
    """Defines the spatial boundary for Quadtree nodes."""
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max

    def contains(self, x, y):
        """Checks if a point is within the boundary."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def intersects(self, other):
        """Checks if another bounding box intersects this one."""
        return not (self.x_max < other.x_min or self.x_min > other.x_max or
                    self.y_max < other.y_min or self.y_min > other.y_max)

class Quadtree:
    """Quadtree for spatial indexing with neighboring list and location table."""
    MAX_CAPACITY = 2000  # Adjusted for better performance
    MIN_REGION_SIZE = 0.0001  # Avoid excessive subdivision

    def __init__(self, boundary):
        self.boundary = boundary
        self.points = []
        self.divided = False
        self.northeast = self.northwest = self.southeast = self.southwest = None
        self.neighbors = []  # Neighboring leaf cells
        self.location_table_pointer = None  # Pointer to stored locations

    def subdivide(self):
        """Splits a full Quadtree node into four child nodes."""
        x_mid = (self.boundary.x_min + self.boundary.x_max) / 2
        y_mid = (self.boundary.y_min + self.boundary.y_max) / 2

        # Prevent excessive subdivision (tiny regions)
        if (self.boundary.x_max - self.boundary.x_min) < self.MIN_REGION_SIZE or \
           (self.boundary.y_max - self.boundary.y_min) < self.MIN_REGION_SIZE:
            return

        self.northeast = Quadtree(BoundingBox(x_mid, y_mid, self.boundary.x_max, self.boundary.y_max))
        self.northwest = Quadtree(BoundingBox(self.boundary.x_min, y_mid, x_mid, self.boundary.y_max))
        self.southeast = Quadtree(BoundingBox(x_mid, self.boundary.y_min, self.boundary.x_max, y_mid))
        self.southwest = Quadtree(BoundingBox(self.boundary.x_min, self.boundary.y_min, x_mid, y_mid))

        self.divided = True

        # Assign neighbors dynamically
        for child in [self.northeast, self.northwest, self.southeast, self.southwest]:
            self.neighbors.append(child)

        # Move existing points to child nodes
        for point in self.points:
            self._insert_into_child(point)
        self.points = []

    def _insert_into_child(self, point):
        """Assigns point to the appropriate quadrant after subdivision."""
        if self.northeast and self.northeast.boundary.contains(point[0], point[1]):
            return self.northeast.insert(point)
        elif self.northwest and self.northwest.boundary.contains(point[0], point[1]):
            return self.northwest.insert(point)
        elif self.southeast and self.southeast.boundary.contains(point[0], point[1]):
            return self.southeast.insert(point)
        elif self.southwest and self.southwest.boundary.contains(point[0], point[1]):
            return self.southwest.insert(point)
        return False

    def insert(self, point):
        """Inserts a point into the Quadtree."""
        if not self.boundary.contains(point[0], point[1]):
            return False

        if len(self.points) < self.MAX_CAPACITY:
            self.points.append(point)
            return True

        if not self.divided:
            self.subdivide()

        return self._insert_into_child(point)

# Load tweets and construct the Quadtree (First Pass)
def load_tweets_and_index(file_path):
    """First Pass: Builds the spatial Quadtree and initializes location tables."""
    root_boundary = BoundingBox(-180, -90, 180, 90)
    quadtree = Quadtree(root_boundary)

    location_table = {}
    count = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                tweet_id = int(row['tweet_id'])
                lat = float(row['latitude'])
                lon = float(row['longitude'])

                if lat == 0.0 and lon == 0.0:
                    continue  # Ignore invalid points

                quadtree.insert((lon, lat, tweet_id))

                # Store tweet location in location table
                location_table[tweet_id] = (lon, lat)

                count += 1
                if count % 100000 == 0:
                    logging.info(f"Processed {count} tweets...")

            except Exception as e:
                logging.error(f"Error processing tweet {row}: {e}")

    logging.info(f" Completed indexing {count} tweets.")
    return quadtree, location_table

# Second Pass: Build the textual index
def build_textual_index(file_path, location_table):
    """Second Pass: Builds textual index and inverted lists."""
    keyword_index = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                tweet_id = int(row['tweet_id'])
                if tweet_id not in location_table:
                    continue  # Skip tweets not indexed in the first pass

                # Extract keywords and weights
                keywords = row['keywords'].strip('{}').split(',')
                weights = [float(w.strip('{}')) for w in row['keyword_weights'].split(',')]

                # Build inverted index
                for keyword, weight in zip(keywords, weights):
                    if keyword not in keyword_index:
                        keyword_index[keyword] = []
                    keyword_index[keyword].append((tweet_id, weight))

            except Exception as e:
                logging.error(f"Error processing tweet {row}: {e}")

    # Sort inverted index based on weight
    for keyword in keyword_index:
        keyword_index[keyword].sort(key=lambda x: x[1], reverse=True)

    return keyword_index

# Run indexing (Two-Pass TEQ Indexing)
quadtree, location_table = load_tweets_and_index('C:\\Users\\ranjitha\\OneDrive\\Desktop\\U-ASK\\tweets_cleaned.csv')
keyword_index = build_textual_index('C:\\Users\\ranjitha\\OneDrive\\Desktop\\U-ASK\\tweets_cleaned.csv', location_table)

# Save indexes to JSON
with open('quadtree_index.json', 'w') as f:
    json.dump(location_table, f)

with open('textual_index.json', 'w') as f:
    json.dump(keyword_index, f)

# Verify indexed tweets
with open('quadtree_index.json', 'r') as f:
    quadtree_data = json.load(f)
print(f" Total tweets indexed in Quadtree: {len(quadtree_data)}")

with open('textual_index.json', 'r') as f:
    textual_data = json.load(f)

unique_tweet_ids = set()
for keyword in textual_data:
    for tweet_id, _ in textual_data[keyword]:
        unique_tweet_ids.add(tweet_id)

print(f" Total unique tweets indexed in Textual Index: {len(unique_tweet_ids)}")

logging.info(f" TEQ Quadtree and Textual Indexing Completed!")
print(" TEQ Quadtree and Textual Indexing Completed!")
