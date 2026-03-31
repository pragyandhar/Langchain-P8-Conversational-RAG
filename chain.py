from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

# Session store
store = {}


def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Prompt-1: Question Condenser - Condenses a follow-up question into a standalone question.
condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a Question Condenser. Your sole responsibility is to rewrite a user's latest question into a single, self-contained question that can be fully understood without any prior conversation history. You have been given with chat history and the latest user question.

    ## Core Rules
    1. RESOLVE all pronouns, demonstratives, and implicit references 
    (it, this, that, they, the above, the previous, etc.) by replacing 
    them with the exact entity they refer to from the conversation history.

    2. PRESERVE the user's original intent without adding, inferring, or 
    assuming information that is not present in the conversation history 
    or the question itself.

    3. If the latest question is already fully self-contained and requires 
    no context to understand, return it UNCHANGED. Do not rephrase for 
    style or variety.

    4. If the latest question represents a clear topic shift — i.e., it has 
    no meaningful connection to the prior conversation — treat it as a 
    fully standalone question and return it as-is without merging prior 
    context into it.

    5. NEVER answer the question. NEVER explain your reasoning. 
    Output ONLY the rewritten standalone question, nothing else.

    6. Keep the output as concise as possible while remaining complete. 
    Do not pad, summarize, or expand the scope of the question.

    ## Output Format
    A single question. No preamble. No explanation. No punctuation beyond 
    what the question itself requires.
    """),
    #-------------------------------------------------#
    MessagesPlaceholder(variable_name="chat_history"),
    #-------------------------------------------------#
    ("human", "{input}")
])

# Prompt-2: QA Chain
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        You are a helpful assistant that answers questions
        strictly based on the provided context below.

        Rules:
        - If the answer exists in the context, answer clearly and concisely.
        - If the answer is NOT in the context, say: "I could not find this in the provided documents."
        - Never use outside knowledge.

        Context:
        {context}
        """),
        #-------------------------------------------------#
        MessagesPlaceholder(variable_name="chat_history"),
        #-------------------------------------------------#
        ("human", "{input}")
    ]
)


def build_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Step-1: Create history-aware retriever chain
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        condense_prompt
    )
    # Step-2: Combine retrieved documents into a single context string
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt
    )
    # Step-3: Create a full RAG Chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        combine_docs_chain,
    )
    # Step-4: Wrap with Message History
    convo_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history"
    )

    return convo_chain
