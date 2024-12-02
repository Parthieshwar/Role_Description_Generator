import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
try:
    df = pd.read_excel('/home/parthieshwar/Development/NP/Resume/Backend/Book.xlsx')
except FileNotFoundError:
    print("Error: File 'Book.xlsx' not found.")
    exit()

# Check if required columns are in the dataset
required_columns = ['Role', 'Description']
if not all(column in df.columns for column in required_columns):
    print(f"Error: Required columns {required_columns} not found in the file.")
    exit()

# Convert the 'Description' column to a list for embedding
dataset = df['Description'].tolist()

# Load the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute embeddings for each description
embeddings = model.encode(dataset)
print("Embeddings computed successfully.")

# Perform clustering using DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(embeddings)
df['cluster'] = clusters
print("Clustering complete.")

# Define a function to find related descriptions based on a user input role
def find_related_descriptions(user_input, data, model, embeddings):
    # Find exact matches based on the Role column
    exact_matches = data[data['Role'].str.lower() == user_input.lower()]
    
    # Embed the user input and calculate cosine similarities
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, embeddings).flatten()
    
    # Get top 30 most similar descriptions
    top_indices = np.argsort(similarities)[::-1][:30]
    related_descriptions = data.iloc[top_indices][['Role', 'Description', 'cluster']]
    
    # Combine exact matches and top related descriptions, removing duplicates
    results = pd.concat([exact_matches, related_descriptions]).drop_duplicates().reset_index(drop=True)
    
    # Return only the 'Description' column of results
    return results['Description']

# Get user input
user_input = input("Role: ").strip()

# Retrieve related descriptions based on input
related_descriptions = find_related_descriptions(user_input, df, model, embeddings)

# Display the results
if related_descriptions.empty:
    print("No related descriptions found.")
else:
    print("Related Descriptions:")
    print(related_descriptions.to_string(index=False))
