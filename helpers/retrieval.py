import numpy as np

# Collection name -> (embeddings array, documents), loaded once instead of per request
_collection_cache = {}

def _get_collection_data(collection):
    if collection.name not in _collection_cache:
        data = collection.get(include=['embeddings', 'documents'])
        _collection_cache[collection.name] = (
            np.asarray(data['embeddings'], dtype=np.float32),
            data['documents'],
        )
    return _collection_cache[collection.name]

def knn_search(prompt, embedding_model, collection, k=10):
    # Compute embedding for the input sentence
    prompt_embedding = embedding_model.encode([prompt], convert_to_numpy=True)[0]

    embeddings, documents = _get_collection_data(collection)

    # Exact L1 nearest neighbors (same result as the old ball tree, without rebuilding it)
    distances = np.abs(embeddings - prompt_embedding).sum(axis=1)
    k = min(k, len(documents))
    nearest = np.argpartition(distances, k - 1)[:k]
    nearest = nearest[np.argsort(distances[nearest])]

    # Retrieve words from indices
    retrieved_words = [documents[i] for i in nearest]
    print("Retrieved words: ", retrieved_words)
    return retrieved_words


def get_or_create_collection(client, collection_name, words, word_embeddings):
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
        
        # Retrieve the existing IDs (words) in the collection
        existing_ids = set(collection.get()['ids'])

        # Identify new words (IDs) to add
        new_words = []
        new_embeddings = []

        for i, word in enumerate(words):
            if word not in existing_ids:
                new_words.append(word)
                new_embeddings.append(word_embeddings[i])
            # else:
            #     print(f"Embedding ID '{word}' already exists. Skipping addition.")

        if new_words:
            print(f"Adding {len(new_words)} new embeddings to the collection.")
            collection.add(
                documents=new_words,
                embeddings=[embedding.tolist() for embedding in new_embeddings],
                ids=new_words
            )
        else:
            print("No new embeddings to add.")
    except Exception as e:
        print(f"Collection '{collection_name}' not found. Creating a new one.")
        collection = client.create_collection(name=collection_name)
        collection.add(
            documents=words,
            embeddings=word_embeddings.tolist(),
            ids=words
        )
    return collection
