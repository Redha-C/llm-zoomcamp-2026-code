from gitsource import GithubRepositoryDataReader



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


# Question 1: How many lesson pages are there in the dataset ? 
print("=" * 25 + " Question 1 " + "=" * 25)
print("\n")
print(f"The dataset has {len(retrieve_repo(repo="llm-zoomcamp"))} pages.")
print("\n")
print("=" * 62)
