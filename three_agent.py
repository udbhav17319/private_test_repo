def safe_result_parser(result, agents):
    """
    Convert LLM output into actual agent objects to call.
    Supports multiple agents in sequence.
    """
    if not result.value:
        return []
    val = str(result.value).strip()
    selected_agents = []

    # Split by comma and normalize
    for name in val.split(","):
        name = name.strip()
        for agent in agents:
            if agent.name.lower() == name.lower():
                selected_agents.append(agent)
                break

    return selected_agents

selection_strategy = KernelFunctionSelectionStrategy(
    function=selection,
    kernel=_create_kernel("selector"),
    result_parser=lambda r: safe_result_parser(r, chat.agents),  # Pass actual agents
    agent_variable_name="agents",
    history_variable_name="history",
)



import asyncio
import dotenv
import logging
import os
import uuid
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import KernelFunctionTerminationStrategy
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt

from local_python_plugin3 import LocalPythonPlugin  # Your code execution plugin

dotenv.load_dotenv()

# Azure OpenAI Config
azure_openai_endpoint = os.getenv("OPENAI_ENDPOINT", "")
azure_openai_api_key = os.getenv("OPENAI_KEY", "")
azure_openai_deployment = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o")

# Agent names
CODEWRITER_NAME = "CodeWriter"
CODEEXECUTOR_NAME = "CodeExecutor"
CODE_REVIEWER_NAME = "CodeReviewer"
TERMINATION_KEYWORD = "done"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create kernel per agent
def _create_kernel(service_id: str) -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment,
            api_key=azure_openai_api_key,
        )
    )
    kernel.add_plugin(plugin_name="LocalCodeExecutionTool", plugin=LocalPythonPlugin())
    return kernel

# Result parser for selecting next agent
def parse_agent_result(result):
    if not result.value:
        return None
    val = result.value
    if isinstance(val, list) and val:
        val = val[0]
    name = str(val).strip()
    if name.lower() in ["codewriter", "writer"]:
        return CODEWRITER_NAME
    if name.lower() in ["codeexecutor", "executor"]:
        return CODEEXECUTOR_NAME
    if name.lower() in ["codereviewer", "reviewer"]:
        return CODE_REVIEWER_NAME
    if TERMINATION_KEYWORD in name.lower():
        return TERMINATION_KEYWORD
    return None

async def multi_agent_loop(user_query: str, max_iterations: int = 5):
    # --- Agents ---
    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=_create_kernel(CODEWRITER_NAME),
        name=CODEWRITER_NAME,
        instructions=f"You are a skilled Python developer. Write clean Python code based on user requests. Return only code, no explanations.",
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=CODEWRITER_NAME,
            temperature=0.2,
            max_tokens=1500,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    executor = ChatCompletionAgent(
        service_id=CODEEXECUTOR_NAME,
        kernel=_create_kernel(CODEEXECUTOR_NAME),
        name=CODEEXECUTOR_NAME,
        instructions=f"You are an execution agent. Run Python code and return output/errors.",
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=CODEEXECUTOR_NAME,
            temperature=0.2,
            max_tokens=1500,
            function_choice_behavior=FunctionChoiceBehavior.Required(
                filters={"included_plugins": ["LocalCodeExecutionTool"]}
            ),
        ),
    )

    reviewer = ChatCompletionAgent(
        service_id=CODE_REVIEWER_NAME,
        kernel=_create_kernel(CODE_REVIEWER_NAME),
        name=CODE_REVIEWER_NAME,
        instructions=f"You are a senior Python code reviewer. Review code for correctness, readability, performance, and best practices. Suggest improvements concisely.",
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=CODE_REVIEWER_NAME,
            temperature=0.2,
            max_tokens=1000,
            function_choice_behavior=FunctionChoiceBehavior.Required(),
        ),
    )

    agents = {
        CODEWRITER_NAME: writer,
        CODEEXECUTOR_NAME: executor,
        CODE_REVIEWER_NAME: reviewer
    }

    # --- LLM selection function ---
    select_next_agent_fn = KernelFunctionFromPrompt(
        function_name="select_next_agent",
        prompt=f"""
You are an agent selector. Based on the user's query and conversation history, decide which agent should run next.
Agents available:
- CodeWriter: writes Python code
- CodeExecutor: executes Python code
- CodeReviewer: reviews Python code

Rules:
- Only choose the agent(s) required for this query.
- If the task is fully completed, reply "{TERMINATION_KEYWORD}".
- Return only the agent name or "{TERMINATION_KEYWORD}" with no extra text.

User Query:
{{{{user_query}}}}

Conversation History:
{{{{history}}}}
        """
    )

    # --- Initialize AgentGroupChat ---
    chat = AgentGroupChat(
        agents=list(agents.values()),
        selection_strategy=KernelFunctionSelectionStrategy(
            function=select_next_agent_fn,
            kernel=_create_kernel("selector"),
            result_parser=parse_agent_result,
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=list(agents.values()),
            function=select_next_agent_fn,
            kernel=_create_kernel("terminator"),
            result_parser=lambda r: TERMINATION_KEYWORD in str(r.value).lower(),
            history_variable_name="history",
            maximum_iterations=max_iterations,
        ),
    )

    await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_query))

    final_responses = []
    async for response in chat.invoke():
        final_responses.append({"agent": response.name, "content": response.content})
        logging.info(f"\n🤖 {response.name}:\n{response.content}\n")

    return final_responses

# ------------------ Interactive loop ------------------
async def main():
    print("🎯 Multi-Agent Assistant Ready!")
    print("Type `exit` to quit.\n")

    while True:
        user_input = input("🧠 User:> ")
        if user_input.lower() == "exit":
            break

        responses = await multi_agent_loop(user_input)
        print("✅ Task complete.\n")
        for r in responses:
            print(f"[{r['agent']}] Output:\n{r['content']}\n")

if __name__ == "__main__":
    asyncio.run(main())
