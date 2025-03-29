import os
import csv

# Directory containing all folders with tweet data
base_directory = "C:\\Users\\ranjitha\\OneDrive\\Desktop\\U-ASK\\U-ask_data"

# Initialize variables
structured_data = []
problematic_files = {}
processed_count = 0  # Track processed tweets

print("Processing started...")

# Loop through tweet folders
for folder_name in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder_name)
    
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing folder: {folder_name}")

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

                for line in lines:
                    parts = line.strip().split()

                    if len(parts) < 4:
                        problematic_files.setdefault(filename, []).append(f"Malformed line: {line.strip()}")
                        continue

                    try:
                        tweet_id = int(parts[0])
                        latitude = float(parts[1])
                        longitude = float(parts[2])
                        text_size = int(parts[3])  

                        keywords_and_weights = parts[4:]

                        # Ensure we take the correct number of keyword-weight pairs
                        actual_pairs = len(keywords_and_weights) // 2
                        if actual_pairs != text_size:
                            text_size = actual_pairs  

                        keywords = []
                        weights = []
                        for i in range(0, min(text_size * 2, len(keywords_and_weights)), 2):
                            keyword = keywords_and_weights[i]
                            try:
                                weight = float(keywords_and_weights[i + 1])
                                keywords.append(keyword)
                                weights.append(weight)
                            except ValueError:
                                continue  # Skip invalid pairs

                        if keywords and weights:
                            structured_data.append({
                                "tweet_id": tweet_id,
                                "latitude": latitude,
                                "longitude": longitude,
                                "keywords": "{ " + ",".join(keywords) + " }",  
                                "keyword_weights": "{ " + ",".join(map(str, weights)) + " }"
                            })
                            processed_count += 1  # Increment count

                            # Print progress every 10,000 tweets
                            if processed_count % 10000 == 0:
                                print(f"Processed {processed_count} tweets...")

                    except ValueError:
                        problematic_files.setdefault(filename, []).append(
                            f"Skipping malformed entry: {line.strip()}"
                        )

# Save processed data to CSV
csv_filename = "C:\\Users\\ranjitha\\OneDrive\\Desktop\\U-ASK\\processed_tweets.csv"

with open(csv_filename, mode='w', newline='', encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["tweet_id", "latitude", "longitude", "keywords", "keyword_weights"])
    
    writer.writeheader()
    writer.writerows(structured_data)  # Fast writing method

print(f"Data has been processed and saved to {csv_filename}")

# Save log file for problematic files
log_filename = "C:\\Users\\ranjitha\\OneDrive\\Desktop\\U-ASK\\problematic_files.log"

if problematic_files:
    with open(log_filename, "w", encoding="utf-8") as log_file:
        for file, issues in problematic_files.items():
            log_file.write(f"File: {file}\n")
            for issue in issues:
                log_file.write(f"  - {issue}\n")
            log_file.write("\n")

    print(f"Some files had issues. Check {log_filename} for details.")
else:
    print("No issues found in any files.")
