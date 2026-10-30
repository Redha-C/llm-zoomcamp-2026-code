from gitsource import GithubRepositoryDataReader
from minsearch import Index
from openai import OpenAI


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
    search_results = indexed_documents.search(query)

    return search_results

# RAG: 
# 1. Retrieve data -> retrieve_repo(repo: str) 
# 2. Built the prompt with context -> build_prompt(context, query)
# 3. Generate a new answer from the model 

def build_prompt(context: str, query: str) -> str:
    USER_PROMPT_TEMPLATE = """
    Context:
    {context}

    Question:
    {query}
    """
    # Prompt
    prompt = USER_PROMPT_TEMPLATE.format(
        context = context,
        query = query
    )

    return prompt

def llm(prompt, model='gpt-5.4-mini'):
    openai_client = OpenAI()
    response = openai_client.responses.create(
        model=model,
        input=prompt
    )

    llm_output = response.output_text
    llm_input_tokens = response.usage.input_tokens

    return llm_output, llm_input_tokens

def rag(query, model='gpt-5.4-mini'):
    # Retrieve data
    documents = retrieve_repo(repo='llm-zoomcamp')

    # Build prompt with enriched context
    context = index_and_search(documents, query)
    prompt = build_prompt(context, query)

    # Generate new answer
    llm_outputs = llm(prompt, model)

    return llm_outputs


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

# Question 3: RAG
print("=" * 25 + " Question 3 " + "=" * 25)
print("\n")
query = "How does the agentic loop keep calling the model until it stops?"
llm_output, llm_input_tokens = rag(query)
print(f"The llm output enriched with RAG is: {llm_output}") 
print("\n")
print(f"The number of tokens send to the model is: {llm_input_tokens}")
print("=" * 62)
print("\n")
