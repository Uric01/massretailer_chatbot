import streamlit as st
import time
import random
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
import os
from api_keys import MISTRAL_API_KEY, LANGSMITH_API_KEY
import pickle
from langchain_core.messages import AIMessage

if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "sales_chatbot"
    
    
    
def get_llm():
    """Load llm"""
    llm= ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.1,
        max_retries=2
    )
    return llm

def get_vectorstore():    
    file_path = os.path.join(os.getcwd(), 'vectordb.pickle')

    if os.path.exists(file_path):
        print(f"File path: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                vector_store = pickle.load(f)
                if hasattr(vector_store, 'similarity_search'):
                    results = vector_store.similarity_search("makro", k=2)
                    return vector_store
                for result in results:
                    print(getattr(result, 'page_content', 'No page content'))
                    print(getattr(result, 'metadata', 'No metadata'))
               
                else:
                    print("The loaded object does not have a 'similarity_search' method.")
        except Exception as e:
            print(f"Try error occurred: {e}")
    else:
        print("File path does not exist.")
        

from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
    

from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState) 


from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = get_vectorstore().similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
    
    
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = get_llm().bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = get_llm().invoke(prompt)
    return {"messages": [response]}
    
    
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()    

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "def234"}}
    
    
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(get_llm(), [retrieve], checkpointer=memory)

config = {"configurable": {"thread_id": "def234"}}


###############################################################################################################################################



# Page Configuration
st.set_page_config(page_title="Ecommerce Salesperson", page_icon="💬")
st.title("💬 Wholesaler Ecommerce Salesperson")

# Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_run" not in st.session_state:
    st.session_state.first_run = True

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Simulated Bot Response (Replace with your logic)
def get_bot_response(user_input):
    """Simple echo bot with typing simulation"""
    responses = [
        "Interesting! Can you tell me more about that?",
        "I understand. How can I assist you further?",
        "Thank you for sharing. What would you like to discuss?",
        "Let me think about that...",
    ]
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Return random response
    return random.choice(responses)

# Chat Interface
if input_message := st.chat_input("Type your message here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": input_message})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(input_message)

    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        if "messages" in event and event["messages"]:
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage):
                bot_response = last_message.content



    # Display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Simulate typing animation
        for chunk in bot_response.split():
            full_response += chunk + " "
            time.sleep(0.1)
            response_placeholder.markdown(full_response + "▌")
        
        # Final response
        response_placeholder.markdown(full_response)
    
    
    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})