Prompt #1:
I’m working in a large enterprise and need to demonstrate how Semantic Kernel can orchestrate multiple agents. Write a Python script that identifies the most financially valuable opportunities for the enterprise, then execute the code to verify it runs without errors.
Agents involved: 1 → 2

Prompt #2:
I’m working in a large enterprise and need Python code for a ping pong game. Once the code is ready, review it to suggest improvements and optimizations.
Agents involved: 1 → 3

Prompt #3:
I’m working in a large enterprise hackathon. Write Python code for a ping pong game, execute it to confirm it works correctly, and then publish the completed app.
Agents involved: 1 → 2 → 3


import asyncio
import dotenv
import logging
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
TERMINATION_KEYWORD = "yes"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def termination_parser(result):
    if not result.value:
        return False
    val = str(result.value).strip()
    return TERMINATION_KEYWORD.lower() in val.lower()

async def main():
    # --- Agents ---
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
            - Generate full deployable Azure Function apps.
            - Accept text in JSON body or uploaded text files.
            - Handle target language, LLM integration, and environment variables.
            - Return only code files (`index.js` and `function.json`).
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=APIBUILDER_NAME,
            temperature=0.1,
            max_tokens=2000,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    agents = [writer, executor, reviewer, apibuilder]

    # --- Selection strategy ---
    selection = KernelFunctionFromPrompt(
        function_name="select_next",
        prompt=f"""
            You are a decision function.
            Pick the required agent(s) based ONLY on the user's last message.
            Valid names: {', '.join([a.name for a in agents])}
            - If user asks for code → {CODEWRITER_NAME}.
            - If user asks to execute code → {CODEEXECUTOR_NAME}.
            - If user asks for review → {CODE_REVIEWER_NAME}.
            - If user asks to build an API → {APIBUILDER_NAME}.
            Return agent names comma-separated if multiple. No extra text.
            User message: {{{{user_message}}}}
        """
    )

    # --- Termination strategy ---
    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=f"""
            Determine if the user's request has been fully completed.
            Say only "{TERMINATION_KEYWORD}" if:
            - The correct agent(s) have responded once with output/code.
            Otherwise, respond with anything else.
            Conversation history: {{{{history}}}}
        """
    )

    # --- Multi-agent chat ---
    chat = AgentGroupChat(
        agents=agents,
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection,
            kernel=_create_kernel("selector"),
            result_parser=lambda r: safe_result_parser(r, chat.agents),
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=agents,
            function=termination,
            kernel=_create_kernel("terminator"),
            result_parser=termination_parser,
            history_variable_name="history",
            maximum_iterations=10,
        ),
    )

    print("🎯 Multi-Agent Assistant Ready. Type your request below:")
    print("Type `exit` to quit or `reset` to restart.\n")

    while True:
        user_input = input("🧠 User:> ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            await chat.reset()
            print("🔁 Conversation reset.\n")
            continue

        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))

        async for response in chat.invoke():
            print(f"\n🤖 {response.name}:\n{response.content}\n")

        if chat.is_complete:
            print("✅ Task complete.\n")

if __name__ == "__main__":
    asyncio.run(main())
