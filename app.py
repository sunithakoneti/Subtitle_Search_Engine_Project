from flask import Flask, request, jsonify, render_template
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

app = Flask(__name__, static_folder='static')

# Load embeddings from SQLite database
def load_embeddings_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the 'chunks' table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY,
                        document_id INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        embedding_chunk TEXT NOT NULL
                    )''')
    conn.commit()

    cursor.execute("SELECT * FROM chunks")
    rows = cursor.fetchall()

    embeddings_dict = {}

    for row in rows:
        if len(row) != 4:
            continue

        document_id, chunk_index, _, embedding_chunk = row
        if document_id not in embeddings_dict:
            embeddings_dict[document_id] = []

        embeddings_dict[document_id].append(np.array(list(map(float, embedding_chunk.split(',')))))

    conn.close()

    for doc_id, chunks in embeddings_dict.items():
        embeddings_dict[doc_id] = np.concatenate(chunks)

    return embeddings_dict


embeddings_dict = load_embeddings_from_db("vectors.db")

# Preprocess query and create query embedding
def preprocess_query(query):
    return query.lower().strip()

def create_query_embedding(query):
    return np.random.rand(1, 300)  # Dummy function; replace with actual embedding method

def calculate_similarity(query_embedding, document_embeddings):
    similarities = {}

    for doc_id, doc_embedding in document_embeddings.items():
        similarity = cosine_similarity(query_embedding, doc_embedding.reshape(1, -1))
        similarities[doc_id] = similarity[0][0]

    return similarities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')

    if not query:
        return jsonify({'error': 'Query parameter is missing'}), 400

    preprocessed_query = preprocess_query(query)
    query_embedding = create_query_embedding(preprocessed_query)

    query_embedding = normalize(query_embedding)

    similarities = calculate_similarity(query_embedding, embeddings_dict)

    sorted_documents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    top_documents = sorted_documents[:10]

    response = [{'document_id': doc_id, 'similarity_score': score} for doc_id, score in top_documents]

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
