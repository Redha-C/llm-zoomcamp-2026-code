INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-5.4-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

class ElasticsearchIndex:

    def __init__(self, es_client, index_name="faq-llm-zoomcamp"):
        self.es_client = es_client
        self.index_name = index_name

    def search(
        self,
        query: str,
        num_results: int = 5,
        boost_dict: dict = None,
        filter_dict: dict = None,
    ):
        # 1. Construction dynamique des champs avec leurs poids (boosts)
        if boost_dict:
            fields = []
            for field, boost in boost_dict.items():
                fields.append(f"{field}^{boost}")
            # Si 'answer' n'est pas dans boost_dict, on l'ajoute sans boost explicite (poids = 1.0)
            if "answer" not in boost_dict:
                fields.append("answer")
        else:
            fields = ["question^3", "answer", "section^0.5"]

        must_clause = [
            {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields",
                }
            }
        ]

        # 2. Construction des filtres à partir du dictionnaire filter_dict
        filter_clause = []
        if filter_dict:
            for key, value in filter_dict.items():
                if value is not None:
                    filter_clause.append({"term": {key: value}})

        search_query = {
            "size": num_results,
            "query": {"bool": {"must": must_clause, "filter": filter_clause}},
        }

        response = self.es_client.search(
            index=self.index_name, body=search_query
        )

        return [hit["_source"] for hit in response["hits"]["hits"]]
