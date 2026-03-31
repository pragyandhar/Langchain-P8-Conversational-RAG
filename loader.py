from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

def load_file(file_path: str):
    # Validate input type
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be a string, not {type(file_path).__name__}")
    
    # Extract extension
    ext = os.path.splitext(file_path)[-1].lower()

    # If that extension is .pdf or .txt
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Only .pdf and .txt files are supported.")

    # Create the document object
    documents = loader.load()

    # Return the document object
    print(f"\nLoaded {len(documents)} document(s) from '{file_path}'")
    return documents
