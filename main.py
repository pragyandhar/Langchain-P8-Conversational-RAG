from loader import load_file
from splitter import splitter
from vectorstore import build_vectorstore, build_retriever
from chain import build_chain
import os
import uuid

def load_multiple_documents(file_path):
    all_chunks = []
    for path in file_path:
        try:
            # Step-1: Load the document
            document = load_file(path)
            # Step-2: Split into chunks
            chunks = splitter(document)
            all_chunks.extend(chunks)

            print(f"  '{path}' → {len(chunks)} chunks")
        except (TypeError, ValueError) as e:
            print(f"  ERROR processing '{path}': {e}")
            continue
        except Exception as e:
            print(f"  ERROR processing '{path}': {type(e).__name__}: {e}")
            continue
    
    return all_chunks

def diplay_sources(context_docs):
    seen = set()
    sources = []

    for doc in context_docs:
        # Handle both Document objects and tuples
        if isinstance(doc, tuple):
            doc = doc[0]  # Extract Document from tuple if needed
        
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "")
        label = f"{source} (Page {page})" if page else source
        if label not in seen:
            seen.add(label)
            sources.append(label)
    
    if sources:
        print(f"[Sources: {' | '.join(sources)}]")
    

def main():
    print("\n--- Conversational RAG ---\n")

    print("Enter file paths one by one. Press Enter with no input when done.")
    file_paths = []
    while True:
        path = input(f"File {len(file_paths) + 1}: ").strip()
        if not path:
            if not file_paths:
                print("Please enter at least one file.")
                continue
            break
        if not os.path.exists(path):
            print(f"File not found: '{path}' — skipping")
            continue
        file_paths.append(path)
    
    # Step-1: Load and split documents
    print("\nLoading and splitting documents...")
    all_chunks = load_multiple_documents(file_paths)
    print(f"Total chunks created: {len(all_chunks)}")
    # Step-2: Build Vectorstore and Retriever
    vectorstore = build_vectorstore(all_chunks)
    retriever = build_retriever(vectorstore)
    # Step-3: Build the RAG Chain
    rag_chain = build_chain(retriever)
    # Step-4: Unique session per run
    session_id = str(uuid.uuid4())

    print("\nReady! Ask questions about your documents.")
    print("Commands: 'history' to see chat history | 'quit' to exit\n")

    from chain import store

    while True:
        question = input("You: ").strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if question.lower() == "history":
            messages = store.get(session_id, None)
            if not messages or not messages.messages:
                print("No history yet.\n")
            else:
                print("\n--- Chat History ---")
                for msg in messages.messages:
                    role = "You" if msg.type == "human" else "Assistant"
                    print(f"{role}: {msg.content}")
                print()
            continue

        if not question:
            continue

        result = rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )

        print(f"\nAssistant: {result['answer']}")
        diplay_sources(result["context"])
        print()

if __name__ == "__main__":
    main()
