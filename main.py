import os
import json
import requests
import wikipedia
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain

model = "llama3"

template_str = """
You are Alex, an intelligent and versatile AI assistant. You have extensive knowledge across various domains and are always ready to assist with any query or task. You are known for your efficiency, accuracy, and friendly demeanor.

Background: You have been trained on a wide range of topics and have access to a variety of tools to help you provide accurate and helpful responses. You strive to assist users in the best possible way.

Personality: You are friendly, helpful, and always willing to assist. You communicate clearly and effectively, ensuring that users receive the information they need.

Conversation history:
{history}

Human: {input}
AI: """

# Define conversation history directory
conversation_history_dir = "./memory/assistant"

# Create directory if it doesn't exist
os.makedirs(conversation_history_dir, exist_ok=True)

# Load conversation history and create vector store
def load_conversation_history(conversation_history_dir):
    conversation_files = [f for f in os.listdir(conversation_history_dir) if f.endswith(".json")]
    documents = []
    for file in conversation_files:
        file_path = os.path.join(conversation_history_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            summary = data.get("summary")
            if summary:
                documents.append(Document(page_content=summary))
    return documents


documents = load_conversation_history(conversation_history_dir)
vectorstore = Chroma.from_documents(documents=documents, embedding=OllamaEmbeddings())

# Prompt template for RAG
rag_template = """Use the following pieces of context to answer the question at the end.

If you don't know the answer, just say that you don't know, don't try to make up an answer.

Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}"""

# Initialize chains
conversation_prompt = PromptTemplate(input_variables=["history", "input"], template=template_str)
conversation_chain = ConversationChain(prompt=conversation_prompt, llm=Ollama(model=model))

qa_chain = load_qa_chain(llm=Ollama(model=model), chain_type="stuff")
retrieval_qa_chain = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectorstore.as_retriever())

def update_memory(context, conversation_history_dir, vectorstore):
    summary_prompt = f"Summarize the following conversation context:\n\n{context}\n\nSummary:"
    
    data = {
        "prompt": summary_prompt,
        "model": "mistral",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "top_k": 50},
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
        response.raise_for_status()
        json_data = response.json()
        summary = json_data["response"].strip()
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        print(f"Error during summarization: {e}")
        summary = None
    
    if summary:
        summary_file = os.path.join(conversation_history_dir, f"summary_{len(os.listdir(conversation_history_dir)) + 1}.json")
        with open(summary_file, 'w', encoding='utf-8') as file:
            json.dump({"summary": summary}, file, ensure_ascii=False, indent=2)
        
        new_documents = load_conversation_history(conversation_history_dir)
        vectorstore.add_documents(new_documents)

def decide_response_type(query, context):
    decision_context = {
        "query": query,
        "agent_context": context,
        "decision_criteria": {
        "CONV": "If the prompt is a general conversational question, a greeting, or requires a creative response.",
        "RAG": "If the prompt is asking about specific information or facts that can be retrieved from the knowledge base."
        "TOOL": "If the prompt is asking about a specific task or tool that can be performed by the agent.",
        "UPDATE": "If it's time to update the conversation history
        },
        "examples": [
        {"prompt": "Hi, how are you doing today?", "decision": "[CONV]"},
        {"prompt": "Tell me a little bit about yourself.", "decision": "[CONV]"},
        {"prompt": "What is the capital of France?", "decision": "[RAG]"},
        {"prompt": "How can I improve my public speaking skills?", "decision": "[RAG]"},
        {"prompt": "What's your favorite hobby?", "decision": "[CONV]"},
        {"prompt": "What's the best way to learn Python?", "decision": "[TOOL]"},
        {"prompt": "I'm having trouble with my computer. Can you help me?", "decision": "[TOOL]"}
        ]
    }

    prompt = f"""
Based on the following context, decide whether to continue the conversation [CONV], retrieve augmented generation [RAG], use a tool chain [TOOL], or update memory [UPDATE].

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

    try:
        response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
        response.raise_for_status()
        json_data = response.json()
        decision_response = json.loads(json_data["response"].strip())
        decision = decision_response.get("decision")
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        print(f"Error during decision-making: {e}")
        decision = None

    if decision == "CONV":
        return conversation_chain.invoke({"history": context, "input": query})["response"]
    elif decision == "RAG":
        context_str = f"Character Context:\n{context}\n\n"
        return retrieval_qa_chain.invoke({"query": query, "context": context_str})["result"]
    elif decision == "TOOL":
        if "calculate" in query.lower():
            return calculator_tool(query)
        elif "find" in query.lower() or "information" in query.lower():
            return wikipedia_tool(query)
        elif "weather" in query.lower():
            return weather_tool(query)
        else:
            return "Tool not recognized."
    elif decision == "UPDATE":
        update_memory(context, conversation_history_dir, vectorstore)
        new_context = load_conversation_history(conversation_history_dir)
        return "Memory has been updated with the current conversation context. How can I assist you further?"
    else:
        return "I'm not sure how to respond to that."

def calculator_tool(query):
    import re
    match = re.search(r'\d+ \+ \d+', query)
    if match:
        numbers = list(map(int, match.group().split(' + ')))
        return str(sum(numbers))
    return "Calculation not recognized."

def wikipedia_tool(query):
    search_query = query.replace("find information about", "").strip()
    results = wikipedia.search(search_query)
    if results:
        page = wikipedia.page(results[0])
        return page.summary
    return "No information found."

def weather_tool(query):
    return "The weather in New York is partly cloudy with a high of 68°F and low of 55°F."

# Example usage
queries = [
    "What is your name?", 
    "What were we talking about the other day?",
    "Can you calculate 5+7 for me?",
    "Find information about the latest AI trends.",
    "What's the weather like in New York?"
]

for query in queries:
    print(f"Query: {query}")
    response = decide_response_type(query, template_str)
    print(f"Response: {response}")
