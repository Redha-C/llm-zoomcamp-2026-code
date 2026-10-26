from gitsource import GithubRepositoryDataReader
from dotenv import load_dotenv
from openai import OpenAI
from minsearch import Index
from gitsource import chunk_documents

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
def create_index_and_search(documents, query):
    index = Index(
        text_fields=['content'],
        keyword_fields=['filename']
    )

    fit = index.fit(documents)

    search_results = index.search(query=query)

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
def openai_llm(instructions, prompt, model='gpt-5.4-mini', use_tools=False):
    openai_client = OpenAI()

    input_messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": prompt}
    ]

    if use_tools==True:
        response = openai_client.responses.create(
            model='gpt-5.4-mini',
            input=input_messages,
            tools=[search_tool]
        )
    else:
        response = openai_client.responses.create(
            model='gpt-5.4-mini',
            input=input_messages
        )

    return response

def rag(query, documents, make_agent=False):
    # Step 1: Retrive data
    files = read_files_from_repo()
    documents = parse_files(files)

    # Step 2: Create index and search + augment the context to the LLM
    search_results = create_index_and_search(documents, query)
    prompt = build_prompt(query, search_results)

    if make_agent==True:
        # Step 3: Generate response using the augmented LLM
        response = openai_llm(instructions, prompt, use_tools=True)
    else:
        response = openai_llm(instructions, prompt)

    # Retrieve the number of input tokens
    input_tokens = response.usage.input_tokens

    return response.output_text, input_tokens

def chunk_pages(documents, size=2000, step=1000):
    chunks = chunk_documents(documents, size, step)
    return chunks

# Define base instructions and prompt for the LLM
instructions = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

query = "How does the agentic loop keep calling the model until it stops?"

# Homework question 1: How many lesson pages are in the dataset? 
files = read_files_from_repo()
documents = parse_files(files)
print(f"Number of lesson pages in the dataset: {len(documents)}")
print("\n")

# Homework question 2: Indexing and searching
search_results = create_index_and_search(documents, query)
print(f"For this query '{query}', the folder the LLM should look into is: {search_results[0]['filename']}")
print("\n")

# Homework question 3: RAG
answer, input_tokens = rag(query, documents)
print(f"Answer for the query '{query}': {answer}")
print("\n")

print(f"Number of input tokens for the RAG response: {input_tokens}")
print("\n")
      
# Homework question 4: Chunking lesson pages
chunks = chunk_pages(documents)
print(f"Number of chunks created from lesson pages: {len(chunks)}")
print("\n")

# Homework question 5: RAG with chunking
answer_w_chunks, input_tokens_w_chunks = rag(query, chunks)
print(f"Answer for the query '{query}' with chunking: {answer_w_chunks}")
print(f"Number of input tokens for the RAG response with chunking: {input_tokens_w_chunks}")

# Homework question 6: Agent
instructions = """
You're a course teaching assistant. Answer the student's question 
using the search tool. 
Make multiple searches with different keywords before answering.
"""

query = "How does the agentic loop work, and how is it different from plain RAG?"

search_tool = {
    "type": "function",
    "name": "create_index_and_search",
    "description": "Search the repo files for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the repo file."
            }
        },
        "required": ["documents","query"],
        "additionalProperties": False
    }
}

answer_w_agent, input_tokens_w_agent = rag(query, chunks, make_agent=True)
print(f"Answer for the query '{query}' using an agent: {answer_w_agent}")

