from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(
    api_key = google_api_key,
    model = "models/gemini-2.5-flash"
)

def chat_node(state: ChatState):
    message = state['messages']
    response = llm.invoke(message)
    return {
        'messages':[response]
    }

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile()

initial_state = {
    'messages':[HumanMessage(content="What is the capital of India")]
}

response = chatbot.invoke(initial_state)
print(response)