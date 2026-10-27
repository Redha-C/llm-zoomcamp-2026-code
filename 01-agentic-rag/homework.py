from gitsource import GithubRepositoryDataReader
from dotenv import load_dotenv
from openai import OpenAI
from minsearch import Index
from gitsource import chunk_documents

from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

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
def openai_llm(instructions, prompt, model='gpt-5.4-mini'):
    openai_client = OpenAI()

    input_messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": prompt}
    ]

    response = openai_client.responses.create(
        model='gpt-5.4-mini',
        input=input_messages
    )

    return response

def rag(query, documents):
    # Step 1: Retrive data
    files = read_files_from_repo()
    documents = parse_files(files)

    # Step 2: Create index and search + augment the context to the LLM
    search_results = create_index_and_search(documents, query)
    prompt = build_prompt(query, search_results)

    response = openai_llm(instructions, prompt)

    # Retrieve the number of input tokens
    input_tokens = response.usage.input_tokens

    return response.output_text, input_tokens

def chunk_pages(documents, size=2000, step=1000):
    chunks = chunk_documents(documents, size, step)
    return chunks

def search_in_chuncks(query: str) -> str:
    # Use the search function
    results = create_index_and_search(documents=chunks, query=query)

    # Format the result
    context = ""
    for result in results:
        context += (
            f"Filename: {result['filename']}\nContent: {result['content']}\n\n"
        )
    return context

def agent(function, function_schema, prompt, model="gpt-5.4-mini"):
    agent_tools = Tools()
    agent_tools.add_tool(function, function_schema)
    chat_interface = IPythonChatInterface()
    callback = DisplayingRunnerCallback(chat_interface)

    runner = OpenAIResponsesRunner(
        tools=agent_tools,
        developer_prompt=instructions,
        chat_interface=chat_interface,
        llm_client=OpenAIClient(model="gpt-5.4-mini")
    )

    result = runner.loop(
        prompt=prompt,
        callback=callback,
    )
    
    return result

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
search_tool = {
    "type": "function",
    "name": "search_in_chuncks",
    "description": "Search the repository lesson chunks for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the repo chunks."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

prompt = "How does the agentic loop work, and how is it different from plain RAG?"
result = agent(function=search_in_chuncks, function_schema=search_tool, prompt=prompt)

# Nombre de fois où l'agent a appelé l'outil 'search_in_chuncks'
search_call_count = sum(
    1
    for msg in result.all_messages
    if getattr(msg, "type", None) == "function_call"
    or hasattr(msg, "arguments")
)

print("\n" + "=" * 50)
print("AGENT FINAL ANSWER:")
print("=" * 50)
print(result.last_message)

print("\n" + "=" * 50)
print("METRICS:")
print("=" * 50)
print(f"Total Cost: ${result.cost.total_cost:.6f}")
print(f"Number of times the agent called search: {search_call_count}")
