from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

load_dotenv()

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=OpenAIEmbeddings()
)

retriever = db.as_retriever(
    search_kwargs={"k":3}
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


def ask(question):

    docs = retriever.invoke(question)

    if len(docs) == 0:
        return "No relevant context found."

    context = "\n\n".join(
        d.page_content
        for d in docs
    )

    prompt = f"""
Answer ONLY using the context below.

If the answer is not contained in the context,
reply:

No relevant context found.

Context:

{context}

Question:

{question}
"""

    answer = llm.invoke(prompt)

    print("\nSources:\n")

    for d in docs:
        print(d.metadata["source"])

    return answer.content
