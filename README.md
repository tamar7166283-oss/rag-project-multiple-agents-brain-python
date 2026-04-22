# Agentic RAG System with Event-Driven Workflows

An advanced Retrieval-Augmented Generation (RAG) system built with **LlamaIndex Workflows**. This project implements an autonomous agent capable of intelligent query routing, self-correction, and context-aware conversation.

## 🚀 Key Features

* **Event-Driven Architecture**: Utilizes a non-linear workflow to manage complex AI states, moving away from rigid sequential chains.
* **Intelligent Hybrid Routing**: Dynamically routes queries between **Semantic Search** (Vector DB) and **Structured Data Extraction** (JSON) based on intent analysis.
* **Autonomous Self-Correction**: Implements a validation loop that triggers a **Query Reformulation** step if retrieval confidence falls below a specific threshold.
* **Contextual Memory**: Features a **Query Rewriting** engine that transforms follow-up questions into standalone search terms using conversation history.

## 🛠️ Architecture & Workflow

The system operates through a series of asynchronous steps:
1. **Router**: Analyzes the query and chat history to decide the optimal retrieval path.
2. **Retriever**: Interacts with **Pinecone** for vector-based semantic retrieval.
3. **Validator**: Checks the relevance of retrieved nodes.
4. **Reformulator**: (Optional) If no relevant data is found, the LLM attempts to broaden the search term.
5. **Synthesizer**: Generates the final answer based strictly on the validated context.

## 💻 Tech Stack

* **Orchestration**: LlamaIndex (Workflows)
* **Vector Database**: Pinecone
* **Async Processing**: Python asyncio
* **UI/Interface**: Gradio

## 🚦 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3.Configure your environment variables (API Keys) and run:
```bash
python main.py
```
## 🧠 Design Decisions
* **Why LlamaIndex Workflows?*
*Standard RAG pipelines are often too brittle for production. By using an Event-Driven approach, the system can handle edge cases (like empty search results) gracefully. This architecture allows for better observability and fine-grained control over the AI's decision-making process.

* **Hybrid Search Strategy**
*Integrated both vector-based search and structured data parsing. This ensures the agent can handle both "fuzzy" natural language questions and precise requests for structured information (like lists or rules) from JSON sources.

* **The Self-Correction Loop**
*To reduce hallucinations, the system includes a mandatory validation step. If the retrieved information doesn't meet the similarity threshold, the agent proactively "rethinks" its query instead of providing a wrong answer, significantly increasing system reliability.

* **Asynchronous Execution**
*The entire workflow is built using asyncio to ensure non-blocking operations, which is essential for maintaining a responsive UI and handling multiple RAG steps efficiently.
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)

   
