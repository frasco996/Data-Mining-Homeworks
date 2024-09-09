import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def build_inverted_index(df):
    inverted_index = {}
    
    descriptions = df['Product Description']

    for i, description in enumerate(descriptions):
        terms = description.split()

        for term in terms:
            # Convert the term to lowercase
            term_lower = term.lower()
            
            if term_lower not in inverted_index:
                inverted_index[term_lower] = set()
            inverted_index[term_lower].add(i)
    sorted_inverted_index = dict(sorted(inverted_index.items()))
    return sorted_inverted_index

def save_inverted_index(inverted_index, filename):
    with open(filename, 'wb') as file:
        pickle.dump(inverted_index, file)

def load_inverted_index(filename):
    with open(filename, 'rb') as file:
        inverted_index = pickle.load(file)
    return inverted_index

def calculate_cosine_similarity(query_vector, tfidf_matrix):
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    return cosine_similarities

def search_products(query, inverted_index, tfidf_vectorizer, tfidf_matrix, df):
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = calculate_cosine_similarity(query_vector, tfidf_matrix)

    # Get indices of most similar products
    similar_product_indices = cosine_similarities.argsort()[:-6:-1]

    # Retrieve and return information of the most similar products
    result = []
    for index in similar_product_indices:
        result.append(df.iloc[index])  # Use iloc to access DataFrame by integer location

    return result

if __name__ == "__main__":
    df = pd.read_csv('./amazon_productsClean.tsv', sep='\t')
    
    # Build inverted index
    inverted_index = build_inverted_index(df)
    
    # Save inverted index to a file
    save_inverted_index(inverted_index, "inverted_index.pkl")

    # Load inverted index from a file
    loaded_inverted_index = load_inverted_index("inverted_index.pkl")
    descriptions = df["Product Description"]
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
    # Example query
    #print(tfidf_matrix)
    query = input("Enter your search query: ")

    # Search products based on the query
    results = search_products(query, loaded_inverted_index, tfidf_vectorizer, tfidf_matrix, df)

    # Display the results
    for result in results:
        print(result['Product Description'])
