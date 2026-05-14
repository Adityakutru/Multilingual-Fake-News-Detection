import os
import pandas as pd

# Main dataset path
base_path = "Datasets/zenodo fake news"

# Empty list to store all articles
data = []

# Get all language folders
language_folders = os.listdir(base_path)

# Loop through each language folder
for language_folder in language_folders:

    language_path = os.path.join(base_path, language_folder)

    # Example:
    # Gujarati_F&R_News
    # Hindi_F&R_News

    subfolders = os.listdir(language_path)

    # Loop through fake/real folders
    for subfolder in subfolders:

        subfolder_path = os.path.join(language_path, subfolder)

        # Determine label
        if "fake" in subfolder.lower():
            label = "fake"
        else:
            label = "real"

        # Determine language
        language = subfolder.split("_")[0]

        # Get all text files
        files = os.listdir(subfolder_path)

        # Loop through each file
        for file_name in files:
            
            print(f"Processing: {file_name}")
            if file_name.endswith(".txt"):

                file_path = os.path.join(subfolder_path, file_name)

                try:

                    with open(file_path, "r", encoding="utf-8") as file:

                        text = file.read()

                    # Store article data
                    data.append({
                        "text": text,
                        "label": label,
                        "language": language,
                        "source": "Zenodo"
                    })

                except Exception as e:

                    print(f"Error reading {file_name}")
                    print(e)

# Convert to DataFrame
df = pd.DataFrame(data)

# Show dataset info
print(df.head())

print("\nTotal articles:", len(df))

# Save CSV
df.to_csv(
    "Datasets/zenodo_dataset.csv",
    index=False,
    encoding="utf-8"
)

print("\nCSV file created successfully!")