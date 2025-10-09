 line 126, in _render_function_call
    raise CodeBlockRenderException(error_msg) from exc
semantic_kernel.exceptions.template_engine_exceptions.CodeBlockRenderException: Function `user_message` not found

import asyncio
import dotenv
import logging
import json
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt

from local_python_plugin3 import LocalPythonPlugin  # Your local code execution plugin

# Load .env
dotenv.load_dotenv()

# Azure OpenAI Config
azure_openai_endpoint = "https://etiasandboxaifoundry.openai.azure.com/"
azure_openai_api_key = ""
azure_openai_deployment = "gpt-4o"

CODEWRITER_NAME = "CodeWriter"
CODEEXECUTOR_NAME = "CodeExecutor"
CODE_REVIEWER_NAME = "CodeReviewer"
APIBUILDER_NAME = "APIBUILDER"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Kernel setup ---
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


# --- Multi-agent invocation logic ---
async def invoke_selected_agents(kernel: Kernel, agents_dict: dict, user_message: str):
    """
    Selects one or more agents using the selection function, then executes them concurrently.
    """
    # 1️⃣ Ask the selection agent which agents to run
    selection_prompt = KernelFunctionFromPrompt(
        function_name="select_agents",
        prompt=f"""
            You are a decision-making system responsible for selecting which agents should handle the user's latest request.

            You can select one or more agents depending on the complexity of the user's message.
            Valid agents:
            - {CODEWRITER_NAME}
            - {CODEEXECUTOR_NAME}
            - {CODE_REVIEWER_NAME}
            - {APIBUILDER_NAME}

            Guidelines:
            - If the user asks to write or generate code → include {CODEWRITER_NAME}.
            - If the user asks to execute or test code → include {CODEEXECUTOR_NAME}.
            - If the user asks to review or optimize code → include {CODE_REVIEWER_NAME}.
            - If the user asks to build or deploy an API → include {APIBUILDER_NAME}.
            - If multiple actions are implied (e.g., "write and run code", or "generate an API and test it") → include all relevant agents separated by commas.
            - Return ONLY the agent names, separated by commas — no extra words, no explanations.

            User message:
            {{{{user_message}}}}
        """,
    )

    result = await selection_prompt.invoke(_create_kernel("selector"), user_message=user_message)
    selected_raw = str(result).strip()
    selected_agents = [a.strip() for a in selected_raw.split(",") if a.strip()]

    print(f"\n🎯 Selected Agents: {selected_agents}")

    if not selected_agents:
        return {"error": "No agents selected"}

    # 2️⃣ Run selected agents concurrently
    async def run_agent(agent_name):
        agent = agents_dict.get(agent_name)
        if not agent:
            return {agent_name: "Agent not found"}
        try:
            response = await agent.complete_chat(
                [ChatMessageContent(role=AuthorRole.USER, content=user_message)]
            )
            return {agent_name: str(response[0].content) if response else ""}
        except Exception as e:
            return {agent_name: f"Error: {e}"}

    results = await asyncio.gather(*(run_agent(a) for a in selected_agents))
    merged = {}
    for r in results:
        merged.update(r)
    return merged


# --- Agent creation ---
def create_agents():
    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=_create_kernel(CODEWRITER_NAME),
        name=CODEWRITER_NAME,
        instructions=f"""
            You are a highly skilled Python developer named {CODEWRITER_NAME}.
            - Write clean Python code based on user requests.
            - Return only code, no explanations.
            - Let the executor handle running the code.
        """,
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
        instructions=f"""
            You are an execution agent named {CODEEXECUTOR_NAME}.
            - Run Python code and return output/errors.
            - If a library is missing, install it.
            - Respond in plain English summarizing results.
        """,
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
        instructions=f"""
            You are a senior Python code reviewer named {CODE_REVIEWER_NAME}.
            - Review code for correctness, readability, performance, and best practices.
            - Suggest improvements concisely.
            - Do not execute code unless explicitly asked.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=CODE_REVIEWER_NAME,
            temperature=0.3,
            max_tokens=1000,
            function_choice_behavior=FunctionChoiceBehavior.Required(),
        ),
    )

    apibuilder = ChatCompletionAgent(
        service_id=APIBUILDER_NAME,
        kernel=_create_kernel(APIBUILDER_NAME),
        name=APIBUILDER_NAME,
        instructions=f"""
            You are {APIBUILDER_NAME}, an expert in building REST APIs as Azure Functions in Node.js.
            - Generate complete deployable Azure Function apps.
            - Output only `index.js` and `function.json`.
            - Code must be ready for Azure Function deployment.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=APIBUILDER_NAME,
            temperature=0.1,
            max_tokens=2000,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    return {
        CODEWRITER_NAME: writer,
        CODEEXECUTOR_NAME: executor,
        CODE_REVIEWER_NAME: reviewer,
        APIBUILDER_NAME: apibuilder,
    }


# --- Main loop ---
async def main():
    agents = create_agents()
    kernel = _create_kernel("main")

    print("🤖 Multi-Agent Chat (Dynamic Selection Enabled)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("🧠 User:> ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        result = await invoke_selected_agents(kernel, agents, user_input)
        print("\n=== Combined Output ===")
        print(json.dumps(result, indent=2))
        print("=======================\n")


if __name__ == "__main__":
    asyncio.run(main())
