from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os


load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


@tool
def update_document(content : str) -> str:
    """updates the document with provided content"""
    global document_content
    document_content =  content
    
    return f'document update successful!, the content now is: \n{content}'

@tool
def save_document(filename : str) -> str:
    """Save the current document to a textfile and finish the process
    
    Args:
        filename: Name of the file  
    """
    global document_content
    
    if not filename.endswith('.txt'):
        filename =  f'{filename}.txt'
        
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f'\n Document has been saved to {filename}')
        return f'Document has been saved successfully to {filename}.'
        
    except Exception as e:
        return f'Error saving document: {str(e)}'
        
        
tools = [update_document, save_document]

llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash",
                             temperature=0,
                             api_key=os.getenv("GOOGLE_API_KEY")).bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)
    
    if not state["messages"]:
        user_input = "Hey there i need help!"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
        
    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n  TOOL RESULT: {message.content}")
            
            
graph = StateGraph(AgentState)

graph.add_node('agent', agent)
graph.add_node('tools', ToolNode(tools))

graph.set_entry_point('agent')
graph.add_edge('agent', 'tools')

graph.add_conditional_edges('tools', should_continue,{
    'continue' : 'agent',
    'end' : END
})

app= graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")
    

if __name__ == "__main__":
    run_document_agent()
    
