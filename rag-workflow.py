import os
import json
import urllib3
import asyncio
import gradio as gr
from llama_index.core.workflow import (
    StartEvent, StopEvent, Workflow, step, Event, Context
)
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from pinecone import Pinecone
from llama_index.core.llms import ChatMessage

from config import embed_model, llm, PINECONE_API_KEY, PINECONE_INDEX_NAME

os.environ['CURL_CA_BUNDLE'] = ''
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
Settings.llm = llm
Settings.embed_model = embed_model

class SemanticSearchEvent(Event):
    query: str

class ExtractionQueryEvent(Event):
    query: str

class RetrieveEvent(Event):
    nodes: list

class ValidationEvent(Event):
    relevant_nodes: list

class QueryReformulationEvent(Event):
    original_query: str
    reason: str

class RAGWorkflow(Workflow):
    
    @step
    async def router(self, ctx: Context, ev: StartEvent) -> ExtractionQueryEvent | SemanticSearchEvent | StopEvent:
        query = ev.get("query")
        chat_history = ev.get("chat_history", "")
        if not query: return StopEvent(result="No query provided.")
        
        # 1. שכתוב השאילתה לפי ההיסטוריה (Query Rewriting)
        rewrite_msg = ChatMessage(rewrite_msg = ChatMessage(role="system", content=f"""
            You are a query rewriter for a RAG system.
            1. Look at the Chat History and the New Question.
            2. If the user asks a question about the project (using words like "it", "this", "the app"), rewrite it as a standalone search query.
            3. If the user gives you a command (like "search for X" or "tell me more"), extract the core topic they want to find.
            4. Do not answer the question yourself. Just return the search query.
            
            History:
            {chat_history}
            
            New Question: {query}
            Standalone Query:"""))
        
        rewritten_res = await llm.achat([rewrite_msg])
        final_query = str(rewritten_res.message.content).strip()
        
        print(f"🔄 Original: {query} -> Rewritten: {final_query}")
        await ctx.store.set("user_query", final_query)
        
        router_msg = ChatMessage(role="user", content=f"""
            Analyze: "{final_query}"
            If it asks for a LIST, ALL items, RULES, or DECISIONS, return 'STRUCTURED'.
            Otherwise, return 'SEMANTIC'. One word only.""")
        
        response = await llm.achat([router_msg])
        choice = str(response.message.content).strip().upper()
        
        if "STRUCTURED" in choice:
            return ExtractionQueryEvent(query=final_query)
        return SemanticSearchEvent(query=final_query)

    @step
    async def extract_from_json(self, ctx: Context, ev: ExtractionQueryEvent) -> StopEvent:
        print(f"📊 Extraction: {ev.query}")
        try:
            with open("structured_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            msg = ChatMessage(role="user", content=f"Data: {json.dumps(data['items'], ensure_ascii=False)}\nQuery: {ev.query}")
            res = await llm.achat([msg])
            return StopEvent(result=str(res.message.content))
        except: return StopEvent(result="Error reading structured data.")

    @step
    async def retrieve(self, ctx: Context, ev: SemanticSearchEvent) -> RetrieveEvent:
        print(f"🔍 Searching: {ev.query}")
        pc = Pinecone(api_key=PINECONE_API_KEY, ssl_verify=False)
        index = VectorStoreIndex.from_vector_store(PineconeVectorStore(pinecone_index=pc.Index(PINECONE_INDEX_NAME)))
        nodes = index.as_retriever(similarity_top_k=5).retrieve(ev.query)
        return RetrieveEvent(nodes=nodes)

    @step
    async def validate(self, ctx: Context, ev: RetrieveEvent) -> ValidationEvent | QueryReformulationEvent | StopEvent:
        filtered_nodes = SimilarityPostprocessor(similarity_cutoff=0.25).postprocess_nodes(ev.nodes)
        
        if not filtered_nodes:
            retries = await ctx.data.get("retry_count", 0)
            if retries < 1:
                await ctx.data.set("retry_count", retries + 1)
                return QueryReformulationEvent(original_query=await ctx.store.get("user_query"), reason="No nodes")
            return StopEvent(result="I couldn't find relevant information for your request.")
        
        return ValidationEvent(relevant_nodes=filtered_nodes)

    @step
    async def reformulate_query(self, ev: QueryReformulationEvent) -> SemanticSearchEvent | StopEvent:
        print(f"🔄 Retrying with better query...")
        msg = ChatMessage(role="user", content=f"The query '{ev.original_query}' failed. Provide a broader search term or return 'IRRELEVANT'.")
        res = await llm.achat([msg])
        new_q = str(res.message.content).strip()
        return StopEvent(result="Topic not found.") if "IRRELEVANT" in new_q.upper() else SemanticSearchEvent(query=new_q)

    @step
    async def synthesize(self, ctx: Context, ev: ValidationEvent) -> StopEvent:
        query = await ctx.store.get("user_query")
        
        qa_template = PromptTemplate(
            "Context: {context_str}\n"
            "Question: {query_str}\n"
            "Instructions: Answer based ONLY on context. If not explicit, you can infer if it's related to the app's core purpose. "
            "If unrelated, say you don't know.\nAnswer:"
        )

        synthesizer = get_response_synthesizer(llm=llm, response_mode="tree_summarize", text_qa_template=qa_template)
        response = synthesizer.synthesize(query=query, nodes=ev.relevant_nodes)
        return StopEvent(result=str(response))

async def chat(message, history):
    formatted_history = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in history[-3:]])
    wf = RAGWorkflow(timeout=60)
    result = await wf.run(query=message, chat_history=formatted_history)
    return str(result)

demo = gr.ChatInterface(fn=chat, title="Agentic RAG with History & Self-Correction")

if __name__ == "__main__":
    demo.launch()