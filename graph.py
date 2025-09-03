from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from states import *
from my_tools import write_file, read_file, get_current_directory, list_files

_ = load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

def planner_prompt(user_prompt: str) -> str:
    return f"Create a detailed plan for: {user_prompt}"

def architect_prompt(plan: str) -> str:
    return f"Create implementation tasks for plan: {plan}"

def coder_system_prompt() -> str:
    return "You are a coding assistant. Generate complete, working code based on the task description."


def planner_agent(state: dict) -> dict:
    """Converts user prompt into a structured Plan."""
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    return {"plan": resp}


def architect_agent(state: dict) -> dict:
    """Creates TaskPlan from Plan."""
    plan: Plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json())
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")

    resp.plan = plan
    print(resp.model_dump_json())
    return {"task_plan": resp}


def coder_agent(state: dict) -> dict:
    """Generate code based on task plan."""
    task_plan = state["task_plan"]
    
    if not task_plan.implementation_steps:
        return {"code": "No implementation steps provided", "status": "DONE"}
    
    # Get first task for simplicity
    current_task = task_plan.implementation_steps[0]
    
    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        "Generate complete, working code for this task."
    )
    
    # Generate code using LLM
    response = llm.invoke(system_prompt + "\n" + user_prompt)
    
    # Write the generated code to file
    try:
        write_file.invoke({"path": current_task.filepath, "content": response.content})
        return {"code": response.content, "status": "DONE"}
    except Exception as e:
        return {"code": f"Error: {str(e)}", "status": "ERROR"}


graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")
graph.add_edge("coder", END)

graph.set_entry_point("planner")
agent = graph.compile()
if __name__ == "__main__":
    result = agent.invoke({"user_prompt": "Create a simple calculator web app with HTML, CSS and JavaScript"},
                          {"recursion_limit": 100})
    print("Final State:", result)