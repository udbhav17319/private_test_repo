import asyncio
import dotenv
import logging
import httpx
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import KernelFunctionTerminationStrategy
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from local_python_plugin3 import LocalPythonPlugin


# ---------------- CONFIGURATION ----------------
dotenv.load_dotenv()
CUSTOM_ENDPOINT = "https://etiasandboxapp.azurewebsites.net/engine/api/chat/generate_ai_response"
BEARER_TOKEN = "YOUR_BEARER_TOKEN"
AZURE_OPENAI_ENDPOINT = "https://YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_API_KEY = "YOUR_AZURE_KEY"

CODEWRITER_NAME = "CodeWriter"
CODEREVIEWER_NAME = "CodeReviewer"
TERMINATION_KEYWORD = "yes"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------- CUSTOM CHAT COMPLETION ----------------
class CustomChatCompletion(ChatCompletionClientBase):
    def __init__(self, service_id: str, endpoint: str, bearer_token: str):
        # include ai_model_id to satisfy SK validation
        super().__init__(service_id=service_id, ai_model_id="custom-model")
        self.endpoint = endpoint
        self.bearer_token = bearer_token

    async def get_chat_message_contents(self, request, settings=None, **kwargs):
        """Core completion call recognized by Semantic Kernel."""
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
            "bing_search": False,
        }
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }
        logging.debug(f"[{self.service_id}] Sending payload: {payload}")

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            logging.debug(f"[{self.service_id}] Raw API response: {data}")
            msg_list = data.get("data", {}).get("msg_list", [])
            text = msg_list[1].get("message", "") if len(msg_list) > 1 else ""
            logging.debug(f"[{self.service_id}] Extracted text: {text}")
            return [ChatMessageContent(role=AuthorRole.ASSISTANT, content=text)]


# ---------------- KERNEL CREATION ----------------
def create_custom_kernel(service_id: str) -> Kernel:
    kernel = Kernel()
    kernel.add_service(CustomChatCompletion(service_id=service_id, endpoint=CUSTOM_ENDPOINT, bearer_token=BEARER_TOKEN))
    kernel.add_plugin(plugin_name="LocalCodeExecutionTool", plugin=LocalPythonPlugin())
    return kernel


def create_selector_kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="selector_service",
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
        )
    )
    return kernel


# ---------------- PARSERS ----------------
def safe_result_parser(result):
    logging.debug(f"[Selector Parser] Raw selector result: {result.value}")
    if not result.value:
        return CODEWRITER_NAME
    val = str(result.value).strip().lower()
    if "review" in val:
        return CODEREVIEWER_NAME
    return CODEWRITER_NAME


def termination_parser(result):
    logging.debug(f"[Termination Parser] Raw result: {result.value}")
    return TERMINATION_KEYWORD.lower() in str(result.value).lower()


# ---------------- MAIN ----------------
async def main():
    writer_kernel = create_custom_kernel(CODEWRITER_NAME)
    reviewer_kernel = create_custom_kernel(CODEREVIEWER_NAME)
    selector_kernel = create_selector_kernel()

    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=writer_kernel,
        name=CODEWRITER_NAME,
        instructions="You are a skilled Python developer. Write Python code only.",
        execution_settings=AzureChatPromptExecutionSettings(service_id=CODEWRITER_NAME),
    )

    reviewer = ChatCompletionAgent(
        service_id=CODEREVIEWER_NAME,
        kernel=reviewer_kernel,
        name=CODEREVIEWER_NAME,
        instructions="You are a senior reviewer. Review Python code only.",
        execution_settings=AzureChatPromptExecutionSettings(service_id=CODEREVIEWER_NAME),
    )

    selection = KernelFunctionFromPrompt(
        function_name="select_next",
        prompt=f"""
        Pick exactly one agent based on the user's last message.
        Valid names:
        - {CODEWRITER_NAME}
        - {CODEREVIEWER_NAME}
        Conversation history:
        {{{{$history}}}}
        """,
    )

    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=f"""
        Determine if the user's request has been completed.
        Say "{TERMINATION_KEYWORD}" if done.
        Conversation history:
        {{{{$history}}}}
        """,
    )

    chat = AgentGroupChat(
        agents=[writer, reviewer],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection,
            kernel=selector_kernel,
            result_parser=safe_result_parser,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[writer, reviewer],
            function=termination,
            kernel=selector_kernel,
            result_parser=termination_parser,
            maximum_iterations=10,
        ),
    )

    print("🎯 Multi-Agent Assistant Ready. Type `exit` or `reset`.\n")

    while True:
        user_input = input("🧠 User:> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "reset":
            await chat.reset()
            print("🔁 Conversation reset.\n")
            continue

        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))
        try:
            async for response in chat.invoke():
                logging.debug(f"🤖 Agent {response.name} replied: {response.content}")
                print(f"\n🤖 {response.name}:\n{response.content}\n")
        except Exception as e:
            logging.exception("❌ Agent invocation failed")


if __name__ == "__main__":
    asyncio.run(main())
