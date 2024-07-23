import os
import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if os.getenv("OPENAI_API_KEY") is None:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Constants
OPENAI_SUMMARY_MODEL = "gpt-4o-mini"
OPENAI_QA_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs")

# Streamlit app title
st.title("Indian Budget 2024 Q&A 🇮🇳")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to load and prepare the agent
@st.cache_resource
def load_agent():
    # pdf_path = os.path.join(PDF_DIR, "budget_2024.pdf")
    documents = SimpleDirectoryReader(input_dir=PDF_DIR).load_data()

    node_parser = SentenceSplitter()
    nodes = node_parser.get_nodes_from_documents(documents)

    if not os.path.exists(DATA_DIR):
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=DATA_DIR)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=DATA_DIR),
        )

    summary_index = SummaryIndex(nodes)

    vector_query_engine = vector_index.as_query_engine(
        llm=OpenAI(model=OPENAI_QA_MODEL),
        embedding_model=OpenAI(model=OPENAI_EMBEDDING_MODEL),
    )
    summary_query_engine = summary_index.as_query_engine(
        llm=OpenAI(model=OPENAI_SUMMARY_MODEL),
    )

    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description="Useful for questions related to specific aspects of Indian Budget 2024.",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description="Useful for any requests that require a holistic summary of the Indian Budget 2024. "
                "For questions that require more specific sections, please use the vector_tool.",
            ),
        ),
    ]

    function_llm = OpenAI(model=OPENAI_QA_MODEL)
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt="""
        You are a specialized agent designed to answer queries about Indian Budget for year 2024.
        You have access to 2023 budget speech as well thus you can compare and answer questions using that context.
        You must use at least one of the tools provided when answering a question about indian budget for year 2024.
        If you're asked anything else other than this topic, PLEASE DO NOT ANSWER the same and tell user who you are for better clarity.
        Be as helpful as you can to human in case of readiness with headings, bullet points, tables, etc.
        e.g. Whenever a comparison is asked, try to answer in a tabular points.
        You do not have any tie ups with indian government, you're just an AI agent who can help answer the questions related to Indian Budget.
        respect user, respect government, respet budget, respect everyone.
        """,
    )
    return agent


# Load the agent
agent = load_agent()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about Indian Budget 2024"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in agent.stream_chat(prompt).response_gen:
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
