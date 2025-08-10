from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import ToolNode
import os


load_dotenv()


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    
    
@tool
def add(a: int, b: int):
    """This is an Addition function that adds two numbers (int) together."""
    return a+b

@tool
def subtract(a: int, b: int):
    """This is a Subtraction function that subtracts two numbers (int) from each other."""
    return a-b

@tool
def multiply(a: int, b: int):
    """This is a Multiplication function that multiplies two numbers (int) together."""
    return a*b

@tool
def devide(a: int, b: int):
    """This is a Division function that divides two numbers (int) by each other."""
    return a/b


tools = [add, subtract, multiply, devide]


llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash",
                             temperature=0,
                             api_key=os.getenv("GOOGLE_API_KEY")).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    
    prompt = SystemMessage(content= 'You are my AI assistant, you need to answer all my queries and before answering them, make sure to call me : O Lord Top G')
    response = llm.invoke([prompt] + state['messages'])
    return {'messages': [response]} 
    
    
def should_continue(state: AgentState) -> str: 
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return 'end'
    else:
        return 'continue'
    
graph = StateGraph(AgentState)
graph.add_node('agent', model_call)

tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

graph.set_entry_point('agent')
graph.add_conditional_edges('agent', should_continue, 
                      {
                          'continue' : 'tools',
                          'end' : END
                      })

graph.add_edge('tools', 'agent')

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {'messages' : [('user', 'Add 3 and 9, minus 9 from 10, multiply 8 by 5, divide 6 by 2')]}
print_stream(app.stream(inputs, stream_mode='values'))