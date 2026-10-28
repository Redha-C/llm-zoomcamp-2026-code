from gitsource import GithubRepositoryDataReader
from minsearch import Index


def retrieve_repo(repo: str) -> List:
    documents = []
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name=repo,
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    # Reading the entire repo
    files = reader.read()

    # Appending each page to documents
    for file in files:
        doc = file.parse()
        documents.append(doc)

    return documents

def index_and_search(documents: List, query: str) -> List:
    # Function which indexes documents and search the answer to a given query inside the indexed documents
    # 1. Index
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )

    indexed_documents = index.fit(documents)

    # 2. Search
    results = indexed_documents.search(query)

    return results

# Question 1: How many lesson pages are there in the dataset ? 
print("=" * 25 + " Question 1 " + "=" * 25)
print("\n")
print(f"The dataset has {len(retrieve_repo(repo="llm-zoomcamp"))} pages.")
print("\n")
print("=" * 62)
print("\n")

# Question 2: Indexing and search
print("=" * 25 + " Question 2 " + "=" * 25)
print("\n")
query = "How does the agentic loop keep calling the model until it stops?"
documents = retrieve_repo(repo="llm-zoomcamp")
results = index_and_search(documents, query)
print(f"The filename of the first result coming from the indexed documents is: {results[0]['filename']}")
print("\n")
print("=" * 62)
print("\n")
