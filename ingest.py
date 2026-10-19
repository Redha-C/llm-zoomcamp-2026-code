import requests
from minsearch import Index

def load_faq_data():
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"""{url_prefix}{course["path"]}"""
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    return documents

def parse_github_readme():
    raw_url = "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/refs/heads/main/README.md"
    response = requests.get(raw_url)
    response.raise_for_status()
    content = response.text

    documents = []
    current_section = "Overview"
    current_text = []

    # Parse the README content line by line
    for line in content.split("\n"):
        # Detection of a Markdown header of level 1, 2, or 3
        if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
            # If we have accumulated text for the previous section
            if current_text:
                body = "\n".join(current_text).strip()
                if body:  # Avoid empty sections
                    documents.append(
                        {
                            "question": current_section,  # Use the section title as the "question"
                            "section": "GitHub README",
                            "answer": body,
                            "course": "llm-zoomcamp",
                        }
                    )

            # Update the title of the new section (removing the '#')
            current_section = line.lstrip("#").strip()
            current_text = []
        else:
            current_text.append(line)

    # Add the very last section of the file
    if current_text:
        body = "\n".join(current_text).strip()
        if body:
            documents.append(
                {
                    "question": current_section,
                    "section": "GitHub README",
                    "answer": body,
                    "course": "llm-zoomcamp",
                }
            )

    return documents

def load_all_data():
    # 1. Collect the classic FAQ data
    faq_docs = load_faq_data()

    # 2. Load the GitHub README
    readme_docs = parse_github_readme()

    # 3. Combine the two lists of documents
    all_documents = faq_docs + readme_docs
    return all_documents

def build_index(all_documents):
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"]
    )
    index.fit(all_documents)
    return index