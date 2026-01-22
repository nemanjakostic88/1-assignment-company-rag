import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retrieval import retrieve_documents

load_dotenv()

JUDGE_MODEL = "gpt-4o-mini"

# LLM-as-a-Judge Prompt
RELEVANCE_PROMPT = """You are a relevance judge.
User Question: {question}
User Context/Intent: The user is strictly looking for information related to: {topics}.
Retrieved Document: {document}

Is this document relevant to the question GIVEN the specific context? 
If the document is about a different topic (e.g., 'Email Groups' when context is 'Scheduling'), answer NO.

Respond ONLY with 'YES' or 'NO'."""

# Test Cases derived from help_rag_ready_published_only.json
TEST_CASES = [
    {
        "id": "archive_ambiguity",
        "question": "How do I archive?",
        "filter_topics": ["Certification", "Certificates"],
        "description": "Target: Certificates."
    },
    {
        "id": "groups_ambiguity",
        "question": "How do I add a new group?",
        "filter_topics": ["Scheduling"], 
        "description": "Target: Scheduling Groups."
    },
    {
        "id": "export_ambiguity",
        "question": "How do I export the data?",
        "filter_topics": ["Inventory"],
        "description": "Target: Inventory Export."
    },
    {
        "id": "profile_ambiguity",
        "question": "How do I edit the profile?",
        "filter_topics": ["Employees"],
        "description": "Target: Employee Profile."
    },
    {
        "id": "schedule_verify",
        "question": "How do I verify the schedule?",
        "filter_topics": ["Scheduling"],
        "description": "Target: Scheduling Verification."
    }
]

def evaluate_relevance(question: str, doc_content: str, topics: list) -> bool:
    """Use LLM to judge if a doc is relevant given the topic context."""
    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(RELEVANCE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    # If no topics provided (Naive), context is "General FieldFlo usage"
    context = ", ".join(topics) if topics else "General FieldFlo software usage"
    
    result = chain.invoke({
        "question": question, 
        "topics": context,
        "document": doc_content
    })
    return "YES" in result.strip().upper()

def calculate_precision(question: str, docs: list, topics: list = None) -> float:
    if not docs:
        return 0.0
    
    relevant_count = 0
    print(f"   Judging {len(docs)} documents against context: {topics if topics else 'General'}...")
    for doc in docs:
        is_relevant = evaluate_relevance(question, doc.page_content, topics)
        if is_relevant:
            relevant_count += 1
            
    return relevant_count / len(docs)

def run_precision_delta_eval():
    print("="*60)
    print("Running Precision Delta Eval: Naive vs. Filtered RAG")
    print("="*60)
    
    results = []
    
    for case in TEST_CASES:
        print(f"\n🧪 Test Case: {case['question']}")
        
        # 1. Naive Retrieval
        # We pass the filter_topics to the calculator so the Judge knows what the user WANTED
        naive_docs = retrieve_documents(case['question'], topics=None, k=5)
        naive_prec = calculate_precision(case['question'], naive_docs, topics=case['filter_topics'])
        
        # 2. Filtered Retrieval
        filtered_docs = retrieve_documents(case['question'], topics=case['filter_topics'], k=5)
        filtered_prec = calculate_precision(case['question'], filtered_docs, topics=case['filter_topics'])
        
        # 3. Calculate Delta
        delta = filtered_prec - naive_prec
        
        print(f"   Naive Precision:    {naive_prec:.2f}")
        print(f"   Filtered Precision: {filtered_prec:.2f}")
        print(f"   📈 Precision Delta: {delta:+.2f}")
        
        results.append(delta)

    avg_delta = sum(results) / len(results)
    print("\n" + "="*60)
    print(f"Average Precision Improvement: {avg_delta:+.2%}")
    print("="*60)

if __name__ == "__main__":
    run_precision_delta_eval()