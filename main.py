import os
import json
import random
import requests
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader


model = "llama3"

template = """
You are Thildiriel, a wise and powerful Moon Elf druid. You have an extensive knowledge of arcane magic and a deep connection to the natural world. You are known for your calm demeanor and your ability to provide guidance to those in need.

Background: You have spent centuries studying the arcane arts and exploring the mysteries of the universe. You have a strong sense of duty to protect the innocent and maintain balance in the world.

Personality: You are patient, empathetic, and always willing to lend a helping hand. You speak with a gentle and reassuring tone, offering words of wisdom and encouragement. You have a deep respect for all living creatures and strive to promote harmony and understanding.
"""

# Load conversation history and create vector store
conversation_history_dir = "./memory/druid"
conversation_files = [f for f in os.listdir(conversation_history_dir) if f.endswith(".json")]
documents = []
for file in conversation_files:
    loader = TextLoader(os.path.join(conversation_history_dir, file), encoding='utf-8')
    documents.extend(loader.load_and_split())
vectorstore = Chroma.from_documents(documents=documents, embedding=OllamaEmbeddings())

# Prompt template for RAG
rag_template = """Use the following pieces of context to answer the question at the end.

If you don't know the answer, just say that you don't know, don't try to make up an answer.

Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=rag_template,
)

# Create conversation chain and retrieval QA chain
llm = Ollama(model=model)
conversation_chain = ConversationChain(llm=llm)
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": RAG_PROMPT},
)


def decide_response_type(query, context):
    decision_context = {
        "query": query,
        "character_context": context,
        "decision_criteria": {
            "CONV": "If the prompt is a general conversational question, a greeting, or requires a creative response.",
            "RAG": "If the prompt is asking about specific information, memories, or events related to the context."
        },
        "examples": [
            {"prompt": "What is your name?", "decision": "[CONV]"},
            {"prompt": "Tell me about yourself.", "decision": "[CONV]"},
            {"prompt": "What did we discuss in our last meeting?", "decision": "[RAG]"},
            {"prompt": "What is your quest?", "decision": "[CONV]"},
            {"prompt": "Remind me what I need to do next.", "decision": "[RAG]"}
        ]
    }

    prompt = f"""
    Based on the provided decision context, decide whether the given query should be answered using a conversation chain [CONV] or a retrieval-augmented-generation chain [RAG].

    Decision Context: {json.dumps(decision_context, indent=2)}

    Query: {query}

    Decision:
    """

    data = {
        "prompt": prompt,
        "model": "mistral",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "top_k": 50},
    }

    response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
    json_data = json.loads(response.text)

    try:
        decision = json.loads(json_data["response"])["decision"]
    except (KeyError, json.JSONDecodeError):
        decision = None

    if decision == "[CONV]":
        print("Chose CONV")
        return conversation_chain.run(context + "\n\nHuman: " + query)
    elif decision == "[RAG]":
        print("Chose RAG")
        context_str = f"Character Context:\n{context}\n\n"
        print("query: " + query)
        return retrieval_qa_chain({"query": query, "context": context_str})["result"]
    else:
        print("Fallback response")
        print(decision)
        return "I'm not sure how to respond to that."


# Example usage
query = "Who are you?"
print("Query: " + query)
response = decide_response_type(query, template)
print(response)

# Example usage
query = "What were we talking about the other day?"
print("Query: " + query)
response = decide_response_type(query, template)
print(response)