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

from local_python_plugin3 import LocalPythonPlugin

dotenv.load_dotenv()

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


def safe_result_parser(result, history_agents):
    """Return the next agent to call based on history and user request"""
    if not result.value:
        return None
    val = result.value
    if isinstance(val, list) and val:
        val = val[0]
    name = str(val).strip().lower()
    mapping = {
        "codeexecutor": CODEEXECUTOR_NAME,
        "codewriter": CODEWRITER_NAME,
        "codereviewer": CODE_REVIEWER_NAME,
        "apibuilder": APIBUILDER_NAME
    }
    chosen = mapping.get(name)
    # Skip agent if already called in this conversation
    if chosen and chosen not in history_agents:
        return chosen
    return None


def termination_parser(result):
    if not result.value:
        return False
    val = result.value
    if isinstance(val, list) and val:
        val = val[0]
    return TERMINATION_KEYWORD.lower() in str(val).lower()


async def main():
    # --- Agents ---
    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=_create_kernel(CODEWRITER_NAME),
        name=CODEWRITER_NAME,
        instructions=f"""
            You are {CODEWRITER_NAME}, write Python code based on user request.
            Return only code. Let executor handle execution.
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
            You are {CODEEXECUTOR_NAME}, run Python code and return output/errors.
            Respond in plain English summarizing results.
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
            You are {CODE_REVIEWER_NAME}, review code for correctness, readability, and best practices.
            Do not execute code unless asked.
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
            You are {APIBUILDER_NAME}, build full deployable Node.js Azure Functions APIs.
            Return only index.js and function.json, ready to deploy.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=APIBUILDER_NAME,
            temperature=0.1,
            max_tokens=2000,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    agents_list = [writer, executor, reviewer, apibuilder]

    # --- Selection function ---
    selection = KernelFunctionFromPrompt(
        function_name="select_next",
        prompt=f"""
            You are a decision function.
            Pick the next agent that should respond based on full conversation history.
            Skip agents that already responded.
            Valid names: {CODEWRITER_NAME}, {CODEEXECUTOR_NAME}, {CODE_REVIEWER_NAME}, {APIBUILDER_NAME}.
            Return ONLY the agent name.
            Conversation history:
            {{{{$history}}}}
        """
    )

    # --- Termination function ---
    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=f"""
            Determine if user's request is fully complete.
            Say only "{TERMINATION_KEYWORD}" if completed, else respond anything else.
            Conversation history:
            {{{{$history}}}}
        """
    )

    chat = AgentGroupChat(
        agents=agents_list,
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection,
            kernel=_create_kernel("selector"),
            result_parser=lambda res: safe_result_parser(res, chat.history_agents if 'chat' in locals() else []),
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=agents_list,
            function=termination,
            kernel=_create_kernel("terminator"),
            result_parser=termination_parser,
            history_variable_name="history",
            maximum_iterations=10,
        ),
    )

    print("🎯 Multi-Agent Chained Assistant Ready. Type your request:")
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
