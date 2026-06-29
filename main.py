from typing import Dict, List, Tuple

from embedder import Embedder
from gitsource import GithubRepositoryDataReader
from gitsource import chunk_documents
from minsearch import Index, VectorSearch


def get_question_vector():
    embedder = Embedder()
    vector = embedder.encode("How does approximate nearest neighbor search work?")
    return vector


def first():
    vector = get_question_vector()
    print(f"First question: answer is: {vector[0]}, vector length: {len(vector)}")


def dot_product(first: List[float], second: List[float]) -> float:
    sum = 0
    for first_element, second_element in zip(first, second):
        sum += first_element * second_element

    return sum


def get_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    documents = [file.parse() for file in reader.read()]
    return documents


def second():
    documents = get_documents()

    embedder = Embedder()

    for document in documents:
        if document["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md":
            text_vector = embedder.encode(document["content"])
            question_vector = embedder.encode("How does approximate nearest neighbor search work?")
            print(f"Second question: answer is {dot_product(text_vector, question_vector)}")


def third():
    embedder = Embedder()
    documents = get_documents()
    chunks = chunk_documents(documents, size=2000, step=1000)
    max_similarity = 0.0
    max_similarity_name = None

    for chunk in chunks:
        if chunk["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md":
            text_vector = embedder.encode(chunk["content"])
            question_vector = embedder.encode("How does approximate nearest neighbor search work?")
            similarity = dot_product(text_vector, question_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_name = chunk["filename"]

    print(f"Third answer is {max_similarity_name}")

def get_chunks_and_vectors() -> Tuple[List[Dict], List[List[float]]]:
    embedder = Embedder()
    documents = get_documents()
    chunks = chunk_documents(documents, size=2000, step=1000)

    vectors = []
    for chunk in chunks:
        text_vector = embedder.encode(chunk["content"])
        vectors.append(text_vector)

    return chunks, vectors


def fourth():
    chunks, vectors = get_chunks_and_vectors()

    index = VectorSearch(keyword_fields=["filename"])
    index.fit(vectors, chunks)

    embedder = Embedder()
    query_vector = embedder.encode("What metric do we use to evaluate a search engine?")

    results = index.search(query_vector)
    print(f"Fourth answwer is: {results[0]["filename"]}")


def fifth():
    chunks, vectors = get_chunks_and_vectors()

    index = VectorSearch(keyword_fields=["filename"])
    index.fit(vectors, chunks)

    embedder = Embedder()
    query_vector = embedder.encode("How do I store vectors in PostgreSQL?")

    results_embedding = index.search(query_vector, num_results=5)

    index_text = Index(text_fields=["content"])
    index_text.fit(chunks)

    results_text = index_text.search("How do I store vectors in PostgreSQL?", num_results=5)

    text_results = [result["filename"] for result in results_text]
    for result in results_embedding:
        if result not in text_results:
            print(f"Fifth answer: {result["filename"]}")

    import pdb; pdb.set_trace()


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def sixth():
    chunks, vectors = get_chunks_and_vectors()

    index = VectorSearch(keyword_fields=["filename"])
    index.fit(vectors, chunks)

    embedder = Embedder()
    query_vector = embedder.encode("How do I give the model access to tools?")

    results_embedding = index.search(query_vector, num_results=295)

    index_text = Index(text_fields=["content"])
    index_text.fit(chunks)

    results_text = index_text.search("How do I give the model access to tools?", num_results=295)

    final_results = rrf([results_embedding, results_text])

    print(f"Sixth answer: {final_results[0]["filename"]}")


def main():
    print("Hello from llm-zoomcamp-hw2!")
    first()
    second()
    third()
    fourth()
    fifth()
    sixth()


if __name__ == "__main__":
    main()
