from gitsource import GithubRepositoryDataReader
from dotenv import load_dotenv
from openai import OpenAI
from minsearch import Index
from rag_helper import RAGBase

### RAG step 1: Retrieve data 
# Function to read files from repo
def read_files_from_repo(): 
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    
    files = reader.read()

    return files

# Function to parse files retrieved from repo
def parse_files(files):
    documents = []
    for file in files:
        doc = file.parse()
        documents.append(doc)
    return documents

### RAG step 2: Parse, index data and augment the context to the LLM
def create_index_and_search(documents):
    index = Index(
        text_fields=['filename', 'content'],
        keyword_fields=['lesson']
    )

    fit = index.fit(documents)

    search_results = index.search("lesson")

    return search_results

def build_prompt(question, search_results):
    USER_PROMPT_TEMPLATE = """
    Question:
    {question}

    Context:
    {context}
    """
    context = ""
    for result in search_results:
        context += f"Filename: {result['filename']}\nContent: {result['content']}\n\n"

    prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
    return prompt.strip()

# Function to set up GPT model
def openai_llm(prompt, model='gpt-5.4-mini'):
    openai_client = OpenAI()

    instructions = """
    Your task is to answer questions from the course participants
    based on the provided context.

    Use the context to find relevant information and provide accurate
    answers. If the answer is not found in the context,
    respond with "I don't know."
    """

    input_messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": prompt}
    ]

    response = openai_client.responses.create(
        model='gpt-5.4-mini',
        input=input_messages
    )

    return response.output_text

def rag(query):
    files = read_files_from_repo()
    documents = parse_files(files)
    search_results = create_index_and_search(documents)
    prompt = build_prompt(query, search_results)
    answer = openai_llm(prompt)
    return answer
    
query = "How many lesson pages?"

print(rag(query))
