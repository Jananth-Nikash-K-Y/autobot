import gradio as gr
import json

from langchain.llms import Ollama
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

llm = Ollama(model="llama3.1:8b")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True, 
    handle_parse_errors=True
)

initial_json = {
    "containers": [
        {
            "id": 1,
            "weight": 1000,
            "type": "20GP",
            "destination": "New York"           
        },
        {
            "id": 2,
            "weight": 2000,
            "type": "40GP",
            "destination": "Los Angeles"
        }
    ],
    "trucks": [ 
        {
            "id": 1,
            "capacity": 1000,
            "allowed_types": ["refrigerated", "dry"],
            "route": "New York"
        },
        {
            "id": 2,
            "capacity": 2000,
            "allowed_types": ["dry"],
            "route": "Los Angeles"
        }
    ]
}

agent.memory.chat_memory.add_user_message(f"Here is the current truck/container data: {json.dumps(initial_json)}")

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    You are a container-truck assistant.
    Use the follwoing data to answer the user's question.

    Data:
    {context}

    Question: {question}
    """
)

chain = LLMChain(llm=llm, prompt=prompt)

context = json.dumps(initial_json, indent=2)

def chat_with_agent(input):
    try:
        response = agent.run(input)
        return response
    except Exception as e:
        return f"An error occurred: {e}"
    
app = gr.Interface(
    fn=chat_with_agent,
    inputs=gr.Textbox(lines=2, placeholder="Start chatting with the agent..."),
    outputs=gr.Textbox(lines=10, placeholder="Response will appear here..."),
    title="Logistics Agent",
    description="Ask the agent questions about container and truck assignments."
)

if __name__ == "__main__":
    app.launch()
