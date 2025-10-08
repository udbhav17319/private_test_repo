import asyncio
import dotenv
import logging
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import KernelFunctionTerminationStrategy
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt

from local_python_plugin3 import LocalPythonPlugin
import httpx

# Load environment
dotenv.load_dotenv()

# --- Config ---
CUSTOM_ENDPOINT = "https://etiasandboxapp.azurewebsites.net/engine/api/chat/generate_ai_response"
BEARER_TOKEN = "YOUR_BEARER_TOKEN_HERE"  # replace with your token

CODEWRITER_NAME = "CodeWriter"
CODE_REVIEWER_NAME = "CodeReviewer"
TERMINATION_KEYWORD = "yes"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Custom Chat Completion Service ---
class CustomChatCompletion:
    def __init__(self, service_id: str, endpoint: str, bearer_token: str):
        self.service_id = service_id
        self.endpoint = endpoint
        self.bearer_token = bearer_token

    async def get_chat_response_async(self, request):
        prompt_text = request.messages[-1].content if request.messages else ""

        payload = {
            "user_id": "user_1",
            "prompt_text": prompt_text,
            "chat_type": "New-Chat",
            "conversation_id": "",
            "current_msg_parent_id": "",
            "current_msg_id": "",
            "conversation_type": "default",
            "ai_config_key": "AI_GPT4o_Config",
            "files": [],
            "image": False,
            "bing_search": False
        }

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            msg_list = data.get("data", {}).get("msg_list", [])
            if len(msg_list) > 1:
                text = msg_list[1].get("message", "")
            else:
                text = ""
            return type("ChatCompletionResponse", (), {"content": text})


# --- Kernel Creation ---
def _create_kernel(service_id: str) -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        CustomChatCompletion(service_id=service_id, endpoint=CUSTOM_ENDPOINT, bearer_token=BEARER_TOKEN)
    )
    kernel.add_plugin(plugin_name="LocalCodeExecutionTool", plugin=LocalPythonPlugin())
    return kernel


# --- Result Parsers ---
def safe_result_parser(result):
    if not result.value:
        return CODEWRITER_NAME  # fallback
    val = result.value
    if isinstance(val, list) and val:
        val = val[0]
    name = str(val).strip().lower().replace("\n", "").replace(" ", "")
    if "codewriter" in name:
        return CODEWRITER_NAME
    if "codereviewer" in name:
        return CODE_REVIEWER_NAME
    return CODEWRITER_NAME  # fallback


def termination_parser(result):
    if not result.value:
        return False
    val = result.value
    if isinstance(val, list) and val:
        val = val[0]
    return TERMINATION_KEYWORD.lower() in str(val).lower()


# --- Main Async Function ---
async def main():
    # --- Kernels ---
    writer_kernel = _create_kernel(CODEWRITER_NAME)
    reviewer_kernel = _create_kernel(CODE_REVIEWER_NAME)
    selector_kernel = _create_kernel("selector")
    terminator_kernel = _create_kernel("terminator")

    # --- Agents ---
    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=writer_kernel,
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

    reviewer = ChatCompletionAgent(
        service_id=CODE_REVIEWER_NAME,
        kernel=reviewer_kernel,
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

    # --- Selection strategy ---
    selection = KernelFunctionFromPrompt(
        function_name="select_next",
        prompt=f"""
            You are a decision function.
            Pick exactly one agent based ONLY on the user's last message in history.
            Valid names:
            - {CODEWRITER_NAME}
            - {CODE_REVIEWER_NAME}

            Rules:
            - If the user asks for code → {CODEWRITER_NAME}.
            - If the user asks for review → {CODE_REVIEWER_NAME}.
            - Return ONLY the agent name as plain text, no extra text.

            Conversation history:
            {{{{$history}}}}
        """,
    )

    # --- Termination strategy ---
    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=f"""
            Determine if the user's request has been fully completed.
            Say only "{TERMINATION_KEYWORD}" if:
            - The correct agent has responded once with output/code.
            Otherwise, respond with anything else.

            Conversation history:
            {{{{$history}}}}
        """,
    )

    # --- Multi-agent chat ---
    chat = AgentGroupChat(
        agents=[writer, reviewer],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection,
            kernel=selector_kernel,
            result_parser=safe_result_parser,
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[writer, reviewer],
            function=termination,
            kernel=terminator_kernel,
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
