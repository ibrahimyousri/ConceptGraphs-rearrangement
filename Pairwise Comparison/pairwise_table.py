import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.figure_factory as ff
import plotly.graph_objs as go

scene_name = "FloorPlan206"
# Load the pickle files
with open(f'{scene_name}/initial.pkl', 'rb') as file:
    initial_data = pickle.load(file)

with open(f'{scene_name}/shuffled.pkl', 'rb') as file:
    final_data = pickle.load(file)

fobjects_data = final_data['objects']
initial_objects_data = initial_data['objects']

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Initialize an empty matrix for storing similarities
similarity_matrix = np.zeros((len(fobjects_data), len(initial_objects_data)))

# Calculate cosine similarity for each pair
for i, fobject in enumerate(fobjects_data):
    for j, iobject in enumerate(initial_objects_data):
        similarity_matrix[i, j] = cosine_similarity(fobject['clip_ft'], iobject['clip_ft'])


# Plotting the similarity matrix
if len(fobjects_data) <50:
    plt.figure(figsize=(20,20))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm')
    plt.title('CLIP Embeddings Similarity')
    plt.xlabel('Initial Scene Objects')
    plt.ylabel('Shuffled Scene Objects')
    plt.show()
else:
    plt.figure(figsize=(20, 20))
    sns.heatmap(similarity_matrix, cmap='coolwarm', cbar_kws={'label': 'Cosine Similarity'})
    plt.title('CLIP Embeddings Similarity')
    plt.xlabel('Initial Scene Objects')
    plt.ylabel('Shuffled Scene Objects')
    plt.tight_layout()  # Adjust layout to fit the larger plot
    plt.show()
