'''
import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.write(
    "test"
)
'''
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import faiss
import pickle
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from PIL import Image

# --- Streamlit UI ---

st.set_page_config(
    page_title="Visual Search System",
    # page_icon="🖼️",
    layout="wide"
)

st.title("Visual Search System")
st.markdown("---")
# st.markdown("""
# This app demonstrates the concept of searching images using text queries and image embeddings.
# Enter a query below, and the app will simulate finding relevant images.
# **Note:** This is a simulated environment. The embeddings and search results are random for demonstration purposes.
# """)

# --- Configuration ---
# Number of dummy images to simulate
# NUM_DUMMY_IMAGES = 50
# Dimension of the embedding vectors
EMBEDDING_DIM = 512
# Number of top results to display
k = 5

# --- Simulated Data Generation ---

@st.cache_resource
def load_clip_model(model_name: str):
    """
    Loads a pre-trained CLIP model and its processor.
    The processor handles image preprocessing.
    The model generates embeddings.
    """
    print(f"Loading CLIP processor for {model_name}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"Loading CLIP model for {model_name}...")
    model = CLIPModel.from_pretrained(model_name)
    model.eval() # Set model to evaluation mode
    print("CLIP model and processor loaded successfully.")
    return processor, model

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512
clip_processor, clip_model = load_clip_model(CLIP_MODEL_NAME)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def load_embeddings():
    """
    Simulates loading or generating image embeddings and image metadata.
    In a real application, you would load these from your vector database
    and image storage.
    """
    # st.write("Generating dummy image embeddings and metadata...")
    # image_embeddings = {}
    # image_metadata = {} # Stores information like image_id -> image_url

    # for i in range(NUM_DUMMY_IMAGES):
        # image_id = f"image_{i+1}"
        # # Generate a random vector as a dummy embedding
        # embedding = np.random.rand(EMBEDDING_DIM)
        # # Normalize the embedding (important for cosine similarity)
        # embedding = embedding / np.linalg.norm(embedding)
        # image_embeddings[image_id] = embedding

        # # Generate a placeholder image URL
        # # You can replace this with actual image URLs if you have them locally
        # # or from a public source (e.g., Unsplash, Lorem Picsum)
        # width = random.randint(300, 600)
        # height = random.randint(200, 400)
        # # Using placehold.co for dummy images
        # image_url = f"https://placehold.co/{width}x{height}/000000/FFFFFF?text=Image+{i+1}"
        # image_metadata[image_id] = {"url": image_url, "description": f"A placeholder image {i+1}"}
    faiss_index_path = 'new_faiss_index.bin'
    indexed_paths_path = 'new_indexed_image_paths.pkl'
    loaded_faiss_index = faiss.read_index(faiss_index_path)
    with open(indexed_paths_path, 'rb') as f:
        loaded_indexed_image_paths = pickle.load(f)

    # st.write("Dummy data generated.")
    # return image_embeddings, image_metadata
    return loaded_faiss_index, loaded_indexed_image_paths


# Load Embeddings
load_faiss_index, load_indexed_image_paths = load_embeddings()


# --- Simulated Embedding and Search Functions ---

# def get_text_embedding(query_text: str) -> np.ndarray:
    
    # Example using Sentence Transformers (install: pip install sentence-transformers)
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2') # Load once
# def get_text_embedding(query_text: str) -> np.ndarray:
#     return model.encode(query_text)


    # For demonstration, we'll generate a random vector.
    # In a real scenario, this vector should be semantically meaningful
    # and compatible with your image embeddings.
    # st.info(f"Simulating text embedding for: '{query_text}'")
    # embedding = np.random.rand(EMBEDDING_DIM)
    # return embedding / np.linalg.norm(embedding) # Normalize

# def search_image_embeddings(query_embedding: np.ndarray, top_n: int = TOP_N_RESULTS):

        
    # st.info("Simulating similarity search...")
    # similarities = []
    # for image_id, img_emb in dummy_image_embeddings.items():
    #     # Calculate cosine similarity between query and image embedding
    #     # Reshape for sklearn's cosine_similarity which expects 2D arrays
    #     similarity = cosine_similarity(query_embedding.reshape(1, -1), img_emb.reshape(1, -1))[0][0]
    #     similarities.append((image_id, similarity))

    # # Sort by similarity in descending order
    # similarities.sort(key=lambda x: x[1], reverse=True)

    # # Return top N results
    # return similarities[:top_n]

df = pd.read_csv('captions_for_testimages.csv')

# User input
user_query = st.text_input(
    "Enter your search query:",
    placeholder="e.g., 'A beautiful landscape', 'Dogs playing in a park', 'Abstract art'"
)

if st.button("Search Images"):
    if user_query:
        st.subheader("Search Results:")
        with st.spinner("Processing query and searching for images..."):
            # 1. Get text embedding for the query
            # query_embedding = get_text_embedding(user_query)
            with torch.no_grad():
                text_inputs = clip_processor(text=user_query, return_tensors="pt", padding=True, truncation=True)
                text_features = clip_model.get_text_features(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
            # 2. Search for similar images
            # Prepare the query embedding for FAISS search
            query_embedding = text_features.squeeze().numpy().astype('float32')
            query_embedding_for_faiss_search = query_embedding.reshape(1, -1)
            # top_results = search_image_embeddings(query_embedding)
            distances, indices = load_faiss_index.search(query_embedding_for_faiss_search, k)
            st.markdown("---")
            for i, idx in enumerate(indices[0]):
                # Check if the index is valid within the bounds of the loaded paths list
                if 0 <= idx < len(load_indexed_image_paths):
                    # st.markdown(f"- Result {i+1}: {load_indexed_image_paths[idx]} (Similarity: {distances[0][i]:.4f})")
                    # Display the image
                    st.markdown(f"Image {i+1}")
                    # st.markdown(load_indexed_image_paths[idx])
                    # st.image(load_indexed_image_paths[idx])
                    # image_path="/testimages/" + df.iloc[idx]['filename']
                    # st.markdown(image_path)
                    # files = os.listdir('testimages')
                    # for file in files:
                    #     st.markdown(file)
                    st.markdown(os.path.join('testimages', df.iloc[idx]['filename']))
                    st.image(Image.open(os.path.join('testimages', df.iloc[idx]['filename'])))
                    st.markdown(f"(Similarity: {distances[0][i]:.4f})")
                    # st.markdown(f"Caption: {df.iloc[idx]['filename']}")
                    st.markdown(f"Caption: {df.iloc[idx]['caption']} with a similarity score of {distances[0][i]:.4f} when comapred to the query {user_query})
                    st.markdown("---")
                    # st.image(image, caption='Enter any caption here')
                else:
                    st.markdown(f"- Result {i+1}: Invalid index {idx}")
            # if top_results: /workspaces/image_search/testimages/0001.jpg
            #     # Display results in columns for better layout
            #     cols = st.columns(TOP_N_RESULTS) # Create columns for each result

            #     for i, (image_id, similarity) in enumerate(top_results):
            #         with cols[i]:
            #             st.markdown(f"**Result {i+1}**")
            #             st.markdown(f"Similarity: {similarity:.2f}")
                        
            #             # Retrieve metadata for the image
            #             metadata = dummy_image_metadata.get(image_id, {})
            #             image_url = metadata.get("url", "https://placehold.co/300x200?text=Image+Not+Found")
            #             description = metadata.get("description", "No description available.")

            #             st.image(image_url, caption=f"{image_id} - {description}", use_column_width=True)
            #             st.markdown("---") # Separator

            # else:
            #     st.warning("No images found for your query. Try a different one!")
    else:
        st.warning("Please enter a search query.")

# st.markdown("---")
# st.markdown("### How this works (Conceptually):")
# st.markdown("""
# 1.  **Image Embedding:** Each image in a large database is processed by a neural network (e.g., a Vision Transformer or CNN) which converts it into a high-dimensional numerical vector (an "embedding"). Images with similar content will have similar embeddings.
# 2.  **Text Embedding:** When you enter a text query, a separate text-based neural network converts your query into a vector in the *same embedding space* as the images.
# 3.  **Vector Database:** These image embeddings are stored in a specialized database (a "vector database" or "vector index") optimized for fast similarity searches.
# 4.  **Similarity Search:** The app then calculates the "distance" (e.g., cosine similarity) between your query's text embedding and all the image embeddings in the database.
# 5.  **Retrieve Results:** The images with the closest (most similar) embeddings to your query are returned as results.
# """)

