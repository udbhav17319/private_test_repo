import asyncio
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelFunction

# -----------------------------------------------------------
# 1️⃣ Define your dummy agents (replace these with real ones)
# -----------------------------------------------------------
async def code_writer_agent(user_message, history):
    print(f"[CODE WRITER] Handling request: {user_message}")

async def code_executor_agent(user_message, history):
    print(f"[CODE EXECUTOR] Executing code for: {user_message}")

async def code_reviewer_agent(user_message, history):
    print(f"[CODE REVIEWER] Reviewing code for: {user_message}")

async def api_builder_agent(user_message, history):
    print(f"[API BUILDER] Building API for: {user_message}")

# Register your agents in a dictionary for dynamic calling
registered_agents = {
    "CODEWRITER_NAME": code_writer_agent,
    "CODEEXECUTOR_NAME": code_executor_agent,
    "CODE_REVIEWER_NAME": code_reviewer_agent,
    "APIBUILDER_NAME": api_builder_agent,
}

# -----------------------------------------------------------
# 2️⃣ Initialize Kernel
# -----------------------------------------------------------
kernel = Kernel()

# -----------------------------------------------------------
# 3️⃣ Define the prompt for agent selection
# -----------------------------------------------------------
prompt_text = """
You are a decision function.
You can select one or more agents depending on user intent.

Valid agent names:
- CODEWRITER_NAME
- CODEEXECUTOR_NAME
- CODE_REVIEWER_NAME
- APIBUILDER_NAME

Rules:
- If the user asks for code → CODEWRITER_NAME
- If the user asks to execute code → CODEEXECUTOR_NAME
- If the user asks for review → CODE_REVIEWER_NAME
- If the user asks to build an API → APIBUILDER_NAME
- If multiple actions are needed, return names separated by commas.
- Return ONLY the agent name(s), no explanation.

User message: {{user_message}}
Conversation history: {{history}}
"""

config = PromptTemplateConfig(
    input_variables=["user_message", "history"]
)

# Create the selection function
select_agent_func = kernel.create_function_from_prompt(prompt_text, config=config)

# -----------------------------------------------------------
# 4️⃣ Define function to select and invoke agents dynamically
# -----------------------------------------------------------
async def select_and_invoke_agents(user_message: str, history: str = ""):
    result = await select_agent_func.invoke(
        input_context={"user_message": user_message, "history": history}
    )

    selected_text = result.text.strip()
    print(f"\n[SELECTOR OUTPUT]: {selected_text}\n")

    agent_names = [name.strip() for name in selected_text.split(",") if name.strip()]

    if not agent_names:
        print("[ERROR] No agent selected.")
        return

    # Call selected agents sequentially
    for agent_name in agent_names:
        agent_func = registered_agents.get(agent_name)
        if agent_func:
            await agent_func(user_message, history)
        else:
            print(f"[WARNING] Unknown agent: {agent_name}")

# -----------------------------------------------------------
# 5️⃣ Run example
# -----------------------------------------------------------
async def main():
    await select_and_invoke_agents("Please build an API and execute the code", "Earlier we discussed Azure Functions.")
    await select_and_invoke_agents("Write a Python function and then review it", "We worked on this before.")

if __name__ == "__main__":
    asyncio.run(main())
