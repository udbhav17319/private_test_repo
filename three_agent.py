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

def safe_result_parser(result):
    if not result.value:
        return None
    val = result.value
    if isinstance(val, list) and val:
        val = val[0]
    name = str(val).strip().lower()
    if "codeexecutor" in name:
        return CODEEXECUTOR_NAME
    if "codewriter" in name:
        return CODEWRITER_NAME
    if "codereviewer" in name:
        return CODE_REVIEWER_NAME
    if "apibuilder" in name:
        return APIBUILDER_NAME
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

            Your goal is to generate **complete deployable Azure Function apps** based on user requests. 

            Requirements:

            1. **Azure LLM Integration**
            - Use Azure OpenAI / LLM for processing.
            - Read **API key, endpoint, and deployment** from environment variables:
                - OPENAI_KEY
                - OPENAI_ENDPOINT
                - OPENAI_DEPLOYMENT
            - The API should perform translations or other LLM tasks as requested.

            2. **Input Handling**
            - Accept **plain text** in JSON body.
            - Accept **text files** uploaded via `multipart/form-data` (in-memory).
            - Allow optional **target language** in body or query, default to English.
            - Handle everything **on the fly**, no blob storage.

            3. **Node.js Azure Functions v4+ style**
            - `exports.default = async function(context, req) { ... }`
            - Include **function.json** with proper HTTP trigger + response bindings.

            4. **Output Requirements**
            - Return **only the code files**: `index.js` and `function.json`.
            - The code must be **ready to deploy to Azure Functions**.
            - Do not include explanations, comments, or extra text.

            5. **Example API to Implement**
            - Translation API:
                - Accepts text or file.
                - Uses Azure LLM for translation.
                - Returns translated text as JSON.

            Always ensure:
            - **Environment variables are used** for sensitive info.
            - Code handles **both text and file inputs in-memory**.
            - Output is **directly usable** in Azure Functions.
            """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=APIBUILDER_NAME,
            temperature=0.1,
            max_tokens=2000,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    # --- Selection strategy ---
    selection_prompt = f"""
        You are a decision function.
        Pick exactly one agent based ONLY on the user's last message in history.
        Valid names:
        - {CODEWRITER_NAME}
        - {CODEEXECUTOR_NAME}
        - {CODE_REVIEWER_NAME}
        - {APIBUILDER_NAME}

        Rules:
        - If the user asks for code → {CODEWRITER_NAME}.
        - If the user asks to execute code → {CODEEXECUTOR_NAME}.
        - If the user asks for review → {CODE_REVIEWER_NAME}.
        - If the user asks to build an API → {APIBUILDER_NAME}.
        - Return ONLY the agent name, no extra text.

        User message: {{user_message}}
    """
    selection = KernelFunctionFromPrompt(
        function_name="select_next",
        prompt=selection_prompt
    )

    # --- Termination strategy ---
    termination_prompt = f"""
        Determine if the user's request has been fully completed.
        Say only "{TERMINATION_KEYWORD}" if:
        - The correct agent has responded once with output/code.
        Otherwise, respond with anything else.

        User message: {{user_message}}
    """
    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=termination_prompt
    )

    # --- Multi-agent chat ---
    chat = AgentGroupChat(
        agents=[writer, executor, reviewer, apibuilder],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection,
            kernel=_create_kernel("selector"),
            result_parser=safe_result_parser,
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[writer, executor, reviewer, apibuilder],
            function=termination,
            kernel=_create_kernel("terminator"),
            result_parser=termination_parser,
            history_variable_name="history",
            maximum_iterations=10,
        ),
    )

    print("🎯 Multi-Agent Assistant Ready with API Builder. Type your request below:")
    print("Type `exit` to quit or `reset` to restart.\n")

    while True:
        user_input = input("🧠 User:> ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            await chat.reset()
            print("🔁 Conversation reset.\n")
            continue

        # Add user message
        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))

        # Invoke agents
        async for response in chat.invoke():
            print(f"\n🤖 {response.name}:\n{response.content}\n")

        if chat.is_complete:
            print("✅ Task complete.\n")

if __name__ == "__main__":
    asyncio.run(main())
