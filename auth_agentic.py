vices\kernel_services_extension.py", line 88, in get_service
    raise KernelServiceNotFoundError(f"No services found of type {type}.")
semantic_kernel.exceptions.kernel_exceptions.KernelServiceNotFoundError: No services found of type <class 'semantic_kernel.connectors.ai.chat_completion_client_base.ChatCompletionClientBase'>.
❌ Failed to select or invoke agent. See logs above.


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
import httpx

# --- Config ---
dotenv.load_dotenv()
CUSTOM_ENDPOINT = "https://etiasandboxapp.azurewebsites.net/engine/api/chat/generate_ai_response"
BEARER_TOKEN = "YOUR_BEARER_TOKEN_HERE"  # replace
AZURE_OPENAI_ENDPOINT = "https://YOUR_AZURE_OPENAI_ENDPOINT"  # For selector kernel
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_API_KEY = "YOUR_API_KEY_HERE"

CODEWRITER_NAME = "CodeWriter"
CODE_REVIEWER_NAME = "CodeReviewer"
TERMINATION_KEYWORD = "yes"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


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
        logging.debug(f"[{self.service_id}] Sending request payload: {payload}")
        async with httpx.AsyncClient() as client:
            response = await client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            logging.debug(f"[{self.service_id}] Raw response: {data}")
            msg_list = data.get("data", {}).get("msg_list", [])
            if len(msg_list) > 1:
                text = msg_list[1].get("message", "")
            else:
                text = ""
            return type("ChatCompletionResponse", (), {"content": text})


# --- Kernel Creation ---
def _create_custom_kernel(service_id: str) -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        CustomChatCompletion(service_id=service_id, endpoint=CUSTOM_ENDPOINT, bearer_token=BEARER_TOKEN)
    )
    kernel.add_plugin(plugin_name="LocalCodeExecutionTool", plugin=LocalPythonPlugin())
    return kernel


def _create_selector_kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="selector_service",
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY
        )
    )
    return kernel


# --- Result Parsers ---
def safe_result_parser(result):
    logging.debug(f"[Selector Parser] Raw selector result: {result.value}")
    if not result.value:
        logging.debug("[Selector Parser] No value returned, defaulting to CodeWriter")
        return CODEWRITER_NAME
    val = result.value
    if isinstance(val, list) and val:
        val = val[0]
    name = str(val).strip().lower().replace("\n", "").replace(" ", "")
    logging.debug(f"[Selector Parser] Parsed agent name: {name}")
    if "codewriter" in name:
        return CODEWRITER_NAME
    if "codereviewer" in name:
        return CODE_REVIEWER_NAME
    logging.debug("[Selector Parser] Unknown agent, defaulting to CodeWriter")
    return CODEWRITER_NAME


def termination_parser(result):
    logging.debug(f"[Termination Parser] Raw termination result: {result.value}")
    if not result.value:
        return False
    val = result.value
    if isinstance(val, list) and val:
        val = val[0]
    done = TERMINATION_KEYWORD.lower() in str(val).lower()
    logging.debug(f"[Termination Parser] Done: {done}")
    return done


# --- Main Async Function ---
async def main():
    # --- Kernels ---
    writer_kernel = _create_custom_kernel(CODEWRITER_NAME)
    reviewer_kernel = _create_custom_kernel(CODE_REVIEWER_NAME)
    selector_kernel = _create_selector_kernel()
    terminator_kernel = _create_custom_kernel("terminator")  # can use custom kernel for termination

    # --- Agents ---
    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=writer_kernel,
        name=CODEWRITER_NAME,
        instructions="Write Python code only.",
        execution_settings=AzureChatPromptExecutionSettings(service_id=CODEWRITER_NAME)
    )

    reviewer = ChatCompletionAgent(
        service_id=CODE_REVIEWER_NAME,
        kernel=reviewer_kernel,
        name=CODE_REVIEWER_NAME,
        instructions="Review Python code only.",
        execution_settings=AzureChatPromptExecutionSettings(service_id=CODE_REVIEWER_NAME)
    )

    # --- Selection & termination functions ---
    selection = KernelFunctionFromPrompt(
        function_name="select_next",
        prompt=f"""
        Pick exactly one agent based on the user's last message.
        Valid names:
        - {CODEWRITER_NAME}
        - {CODE_REVIEWER_NAME}
        Conversation history:
        {{{{$history}}}}
        """
    )

    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=f"""
        Determine if the user's request has been fully completed.
        Say "{TERMINATION_KEYWORD}" if done.
        Conversation history:
        {{{{$history}}}}
        """
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

    print("🎯 Multi-Agent Assistant Ready. Type `exit` to quit or `reset` to restart.\n")

    while True:
        user_input = input("🧠 User:> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "reset":
            await chat.reset()
            logging.debug("Conversation reset by user")
            print("🔁 Conversation reset.\n")
            continue

        logging.debug(f"Adding user message: {user_input}")
        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))

        try:
            async for response in chat.invoke():
                logging.debug(f"Agent selected: {response.name}")
                logging.debug(f"Agent response content: {response.content}")
                print(f"\n🤖 {response.name}:\n{response.content}\n")
        except Exception as ex:
            logging.exception("Error during agent invocation")
            print("❌ Failed to select or invoke agent. See logs above.")

        if chat.is_complete:
            logging.debug("Chat marked as complete.")
            print("✅ Task complete.\n")


if __name__ == "__main__":
    asyncio.run(main())
