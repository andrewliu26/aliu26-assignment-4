from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')

# Initialize vectorizer (TF-IDF) and apply LSA (TruncatedSVD)
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X_tfidf = vectorizer.fit_transform(newsgroups.data)

# Perform LSA using SVD
n_components = 100  # Number of topics (can be adjusted)
lsa = TruncatedSVD(n_components=n_components)
X_lsa = lsa.fit_transform(X_tfidf)

# Compute document similarities (cosine similarity between all documents)
cosine_similarities = cosine_similarity(X_lsa)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Convert query to vector
    query_vec = vectorizer.transform([query])
    query_lsa = lsa.transform(query_vec)

    # Compute cosine similarities between query and all documents
    similarities = cosine_similarity(query_lsa, X_lsa)[0]

    # Get top 5 most similar documents
    indices = similarities.argsort()[::-1][:5]
    top_similarities = similarities[indices]
    top_documents = [newsgroups.data[i] for i in indices]

    return top_documents, top_similarities, indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)

    # Convert ndarray to lists before returning JSON response
    return jsonify({
        'documents': documents,
        'similarities': similarities.tolist(),  # Convert to list
        'indices': indices.tolist()  # Convert to list
    })


if __name__ == '__main__':
    app.run(debug=True)
