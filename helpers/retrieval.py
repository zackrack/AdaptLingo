from sklearn.neighbors import NearestNeighbors

def knn_search(prompt, embedding_model, collection):
    # Compute embedding for the input sentence
    prompt_embedding = embedding_model.encode([prompt], convert_to_tensor=True)

    # Move the embedding to CPU if it's on CUDA
    if prompt_embedding.is_cuda:
        prompt_embedding = prompt_embedding.cpu()

    data = collection.get(include=['embeddings', 'documents'])
    embeddings = data['embeddings']
    documents = data['documents']

    # l1 ball tree might be good
    knn = NearestNeighbors(n_neighbors=10, metric='l1', algorithm='ball_tree')
    knn.fit(embeddings)

    # Compute nearest neighbors
    distances, indices = knn.kneighbors(prompt_embedding.numpy())

    # Retrieve words from indices
    retrieved_words = [documents[i] for i in indices[0]]
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
