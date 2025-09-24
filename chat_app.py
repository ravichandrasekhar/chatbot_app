import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
 
import config
from db import get_retriever
 
st.set_page_config(page_title="RAG Chatbot (LangChain + FAISS)", page_icon="🤖")
 
st.title("RAG Chatbot — LangChain + FAISS")
st.write("Ask questions about your documents. (Index your docs first with `python ingest.py`)")
 
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("LLM model", options=[config.OPENAI_MODEL, "gpt-4", "gpt-3.5-turbo"], index=0)
    temp = st.slider("Temperature", 0.0, 1.0, config.OPENAI_TEMPERATURE)
    top_k = st.number_input("Retriever top_k", min_value=1, max_value=10, value=config.TOP_K)
    if st.button("Re-load index"):
        st.session_state.pop("chain", None)
        st.success("Index reload requested (done on next query).")
 
retriever = None
try:
    retriever = get_retriever(search_kwargs={"k": top_k})
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.info("Run `python ingest.py` to build the index from files in ./data/")
    st.stop()
 
if config.USE_CHAT_OPENAI:
    if not config.OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set in environment. Please set it to use ChatOpenAI.")
        st.stop()
    llm = ChatOpenAI(model_name=model, temperature=temp, openai_api_key=config.OPENAI_API_KEY)
else:
    # fallback to OpenAI text LLM
    llm = OpenAI(temperature=temp, openai_api_key=config.OPENAI_API_KEY, model_name=model)
 
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer" )
 
chain = ConversationalRetrievalChain.from_llm(
    # llm=llm,
    # retriever=retriever,
    # memory=memory,
    # return_source_documents=True,
        llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)
 
# UI state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []
 
# chat input
query = st.text_input("You:", key="input")
if query:
    with st.spinner("Thinking..."):
        result = chain({"question": query})
    answer = result["answer"]
    # Show answer and sources
    st.session_state.messages.append({"role": "user", "text": query})
    st.session_state.messages.append({"role": "assistant", "text": answer})
 
    st.markdown("**Answer:**")
    st.write(answer)
 
    docs = result.get("source_documents", [])
    if docs:
        st.markdown("**Source documents (top results):**")
        for i, doc in enumerate(docs):
            meta = doc.metadata or {}
            source = meta.get("source", "unknown")
            st.markdown(f"- **Result {i+1}** — source: `{source}`")
            # present a short excerpt
            excerpt = doc.page_content[:1000].replace("\n", " ")
            st.write(excerpt + ("..." if len(doc.page_content) > 1000 else ""))
 
# show conversation history
if st.session_state["messages"]:
    st.markdown("---")
    st.markdown("### Conversation history")
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['text']}")
        else:
            st.markdown(f"**Bot:** {m['text']}")
 