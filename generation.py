import os
import sys
from dotenv import load_dotenv
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retrieval import retrieve_documents

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

# RAG Prompt
RAG_PROMPT = """You are a FieldFlo Support Assistant. 
Answer the question based ONLY on the context provided.

Context:
{context}

Question: {question}

If the context doesn't answer the question, say so.
Answer:"""

def format_context(docs):
    return "\n\n".join([f"Source: {d.metadata.get('title', 'Unknown')}\nContent: {d.page_content}" for d in docs])

def generate_answer(
    question: str,
    topics: Optional[List[str]] = None,
    verbose: bool = True
):
    """
    Generate answer using RAG pipeline with optional filters.
    """
    # 1. Retrieve (Prints are handled inside retrieval.py)
    docs = retrieve_documents(question, topics=topics, k=5)
    
    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the help center.",
            "sources": []
        }

    # 2. Format
    context_text = format_context(docs)

    # 3. Generate
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    print("🤖 Generating answer...\n")
    answer = chain.invoke({
        "context": context_text,
        "question": question
    })

    return {
        "answer": answer,
        "sources": docs
    }

def print_deduplicated_sources(docs):
    """Helper to print unique sources."""
    seen_titles = set()
    unique_sources = []
    
    for doc in docs:
        title = doc.metadata.get('title', 'Unknown')
        if title not in seen_titles:
            unique_sources.append(doc)
            seen_titles.add(title)
    
    print(f"\n📚 Sources ({len(unique_sources)} unique articles):")
    for doc in unique_sources:
        title = doc.metadata.get('title', 'Unknown')
        topics = doc.metadata.get('topics', [])
        print(f"  • {title} | Topics: {topics}")

def interactive_mode():
    """Run interactive Q&A session with filter commands."""
    print("\n" + "="*60)
    print("Metadata-Filtered RAG - Interactive Q&A")
    print("="*60)
    print("\nFilter commands:")
    print("  topic:CRM           - Filter by topic (e.g., CRM, Payroll, Projects)")
    print("  clear               - Reset all filters")
    print("  filters             - Show active filters")
    print("  quit                - Exit")
    print("-" * 60)

    active_topics = []

    while True:
        try:
            user_input = input("\n>> ").strip()
            
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Handle Commands
            if user_input.lower() == 'clear':
                active_topics = []
                print("✅ All filters cleared")
                continue

            if user_input.lower() == 'filters':
                print(f"🔍 Active Topics: {active_topics if active_topics else 'None'}")
                continue

            if user_input.lower().startswith('topic:'):
                parts = user_input.split(':', 1)
                if len(parts) > 1:
                    new_topic = parts[1].strip()
                    active_topics = [new_topic] 
                    print(f"✅ Topic filter set to: {active_topics}")
                continue

            # Handle Question
            print("-" * 50)
            result = generate_answer(
                user_input, 
                topics=active_topics if active_topics else None
            )
            
            print("Answer:")
            print("-" * 50)
            print(result['answer'])
            
            print_deduplicated_sources(result['sources'])

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("=" * 60)
    print("Metadata-Filtered RAG - Generation Pipeline")
    print("=" * 60)

    example_queries = [
        {
            "q": "How do I add a new lead?",
            "topics": ["CRM"],
            "desc": "CRM-focused query"
        },
        {
            "q": "How do I run a payroll report?",
            "topics": ["Payroll"],
            "desc": "Payroll-focused query"
        },
        {
            "q": "How do I fill out a waste load out form?",
            "topics": ["Projects"],
            "desc": "Project form query"
        },
        {
            "q": "How do I create a new employee?",
            "topics": ["Employees"],
            "desc": "Employee management query"
        }
    ]

    print("\n📋 Example Queries with Filters:")
    for i, ex in enumerate(example_queries, 1):
        print(f"  {i}. {ex['q']}... ({ex['desc']})")

    print("\n" + "-" * 50)
    choice = input("Enter 1-4 for examples, 'i' for interactive mode, or your question: ").strip()

    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3', '4']:
        ex = example_queries[int(choice) - 1]
        print(f"\n📝 Question: {ex['q']}")
        print(f"📁 Context: Filter by topics={ex['topics']}")
        
        result = generate_answer(ex['q'], topics=ex['topics'])
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result['answer'])
        
        print_deduplicated_sources(result['sources'])
            
    elif choice:
        # Standard question without filters
        print(f"\n📝 Question: {choice}")
        result = generate_answer(choice)
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result['answer'])
        
        print_deduplicated_sources(result['sources'])

if __name__ == "__main__":
    main()