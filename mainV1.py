import json
import re
import gradio as gr

from langchain.llms import Ollama
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool

llm = Ollama(model="llama3.1:8b")

@tool
def assign_container(input_json: str) -> str:
    input_json = re.sub(r"^```json|```$", "", input_json.strip(), flags=re.MULTILINE)

    try:
        data = json.loads(input_json)
    except json.JSONDecodeError:
        return "Invalid JSON input. Please provide a valid JSON object."

    containers = data.get("containers", [])
    trucks = data.get("trucks", [])
    assignments = []

    for container in containers:
        for truck in trucks:
            if(
                container["weight"] == truck["capacity"] and
                container["type"] == truck["allowed_types"] and
                container["destination"] == truck["route"]
            ):
                assignments.append({
                    "container_id": container["id"],
                    "truck_id": truck["id"]
                })
                break
    
    return json.dumps({"assignments": assignments}, indent=2)

tools = [assign_container]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def process_json(input_text: str) -> str:
    try:
        response = agent.run(f"Assign containers to trucks based on the following input: {input_text}")
        return response
    except Exception as e:
        return f"An error occurred: {e}"
    
app = gr.Interface(
    fn=process_json,
    inputs=gr.Textbox(label="Input JSON", lines=20, placeholder="Enter your JSON input here..."),
    outputs=gr.Textbox(label="Output JSON", lines=10),
    title="Container Assignment Agent",
    description="Assign containers to trucks based on the provided input JSON."
)

if __name__ == "__main__":
    app.launch()



    
