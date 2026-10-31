from gitsource import GithubRepositoryDataReader
from minsearch import Index
from openai import OpenAI
from gitsource import chunk_documents
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

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

def index_and_search(documents: list, query: str, use_chunks=True) -> list:
    # Function which indexes documents and search the answer to a given query inside the indexed documents
    # 1. Index
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )

    if use_chunks:
        chunks = chunk_documents(documents, size=2000, step=1000)
        indexed_documents = index.fit(chunks)
    else:
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

def rag(query, model='gpt-5.4-mini', use_chunk_context=True):
    # Retrieve data
    documents = retrieve_repo(repo='llm-zoomcamp')

    # Build prompt with enriched context
    if use_chunk_context:
        chunks = chunk_documents(documents, size=2000, step=1000)
        context_w_chunks = index_and_search(chunks, query)
        prompt = build_prompt(context_w_chunks, query)
    else:
        context = index_and_search(documents, query)
        prompt = build_prompt(context, query)

    # Generate new answer
    llm_outputs = llm(prompt, model)

    return llm_outputs

def agent(instructions: str, prompt: str, func: function, model="gpt-5.4-mini") -> str:
    agent_tools = Tools()
    agent_tools.add_tool(func)
    chat_interface = IPythonChatInterface()
    callback = DisplayingRunnerCallback(chat_interface)

    runner = OpenAIResponsesRunner(
        tools=agent_tools,
        developer_prompt=instructions,
        chat_interface=chat_interface,
        llm_client=OpenAIClient(model)
    )

    result = runner.loop(
        prompt=prompt,
        callback=callback,
    )

    all_messages = result.all_messages
    last_message = result.last_message
    cost = result.cost.total_cost
    tokens = result.tokens.output_tokens

    return all_messages, last_message, cost, tokens

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

# Question 4: Chunking
chunks = chunk_documents(documents=documents, size=2000, step=1000)
print("=" * 25 + " Question 4 " + "=" * 25)
print("\n")
print(f"The number of chunks we get is: {len(chunks)}") 
print("=" * 62)
print("\n")

# Question 5: RAG with chunking
llm_output_w_chunks, llm_input_tokens_w_chunks = rag(query, use_chunk_context=True)
print("=" * 25 + " Question 5 " + "=" * 25)
print("\n")
print(f"The number of tokens send to the model with chunks is: {llm_input_tokens_w_chunks}")
print("=" * 62)
print("\n")

# Question 6: Turning it into an agent 
instructions = "You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering."
prompt = "How does the agentic loop work, and how is it different from plain RAG?"
all_messages, last_message, cost, tokens = agent(instructions, prompt, index_and_search)
search_call_count = sum(
    1
    for msg in all_messages
    if getattr(msg, "type", None) == "function_call"
    or hasattr(msg, "arguments")
)

print("=" * 25 + " Question 6 " + "=" * 25)
print("\n")
print(f"The final answer is: {last_message}")
print("\n")
print(f"The final answer costs ${cost} and used {tokens} tokens.")
print("\n")
print(f"The agent called the function index_and_search {search_call_count} times.")
print("=" * 62)
print("\n")
