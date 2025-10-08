import asyncio
import logging
import requests
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents.group_chat import AgentGroupChat
from semantic_kernel.exceptions import AgentChatException


# ---------------------------------------------------------------
# ✅ Logging Setup
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------
# ✅ Custom ChatCompletion that calls your REST API
# ---------------------------------------------------------------
class CustomChatCompletion(ChatCompletionClientBase):
    """
    Custom LLM connector for Semantic Kernel that uses an external REST API.
    """

    def __init__(self, service_id: str, api_url: str, bearer_token: str, ai_model_id: str = "Custom-API-LLM"):
        super().__init__()
        self.service_id = service_id
        self.api_url = api_url
        self.bearer_token = bearer_token
        self.ai_model_id = ai_model_id
        logging.info(f"✅ Initialized CustomChatCompletion (service_id={service_id}, api_url={api_url})")

    async def complete_chat(self, context, settings=None):
        """
        Called when the agent generates a message.
        Sends the last user message to your API endpoint.
        """
        try:
            # Extract message text
            if hasattr(context, "messages") and context.messages:
                user_message = context.messages[-1].content
            else:
                user_message = str(context)

            logging.info(f"[{self.service_id}] 🧠 Received user message: {user_message}")

            # Prepare body
            payload = {
                "user_id": "",
                "prompt_text": user_message,
                "chat_type": "New-Chat",
                "conversation_id": "",
                "current_msg_parent_id": "",
                "current_msg_id": "",
                "conversation_type": "default",
                "ai_config_key": "AI_GPT4o_Config",
                "files": ["string"],
                "image": False,
                "bing_search": False
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.bearer_token}"
            }

            logging.debug(f"[{self.service_id}] 🚀 Sending POST request to {self.api_url}")
            logging.debug(f"[{self.service_id}] Headers: {headers}")
            logging.debug(f"[{self.service_id}] Payload: {payload}")

            # Call custom API
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()

            # Log raw JSON
            json_response = response.json()
            logging.debug(f"[{self.service_id}] ✅ Raw API Response: {json_response}")

            # Extract message from nested structure
            msg_list = json_response.get("data", {}).get("msg_list", [])
            if len(msg_list) > 1 and "message" in msg_list[1]:
                reply_text = msg_list[1]["message"]
            elif msg_list and "message" in msg_list[0]:
                reply_text = msg_list[0]["message"]
            else:
                reply_text = "⚠️ Could not extract message from API response."

            logging.info(f"[{self.service_id}] 💬 Model Response: {reply_text}")

            return reply_text

        except requests.exceptions.RequestException as e:
            logging.error(f"[{self.service_id}] ❌ API Request failed: {e}", exc_info=True)
            return f"Error calling API: {str(e)}"

        except Exception as e:
            logging.exception(f"[{self.service_id}] ❌ Unexpected error in complete_chat: {e}")
            return f"Unexpected error: {e}"


# ---------------------------------------------------------------
# ✅ Helper to Create Agents
# ---------------------------------------------------------------
def create_agent(name: str, service_id: str, api_url: str, bearer_token: str, kernel: Kernel):
    logging.info(f"🧠 Creating agent '{name}' with service_id '{service_id}'")
    chat_client = CustomChatCompletion(service_id=service_id, api_url=api_url, bearer_token=bearer_token)
    kernel.add_service(chat_client, service_id=service_id)

    agent = ChatCompletionAgent(
        service_id=service_id,
        kernel=kernel,
        name=name,
        description=f"{name} specialized in {service_id} tasks"
    )

    logging.info(f"✅ Agent '{name}' created successfully")
    return agent


# ---------------------------------------------------------------
# ✅ Main Execution Function
# ---------------------------------------------------------------
async def main():
    logging.info("🚀 Starting Multi-Agent Chat using Custom REST API")

    # Custom API endpoint and token
    CUSTOM_API_URL = "https://etiasandboxapp.azurewebsites.net/engine/api/chat/generate_ai_response"
    BEARER_TOKEN = "YOUR_BEARER_TOKEN_HERE"

    # Initialize kernel
    kernel = Kernel()
    logging.info("✅ Kernel initialized")

    # Create two agents using your API
    code_writer = create_agent("CodeWriter", "code-writer-service", CUSTOM_API_URL, BEARER_TOKEN, kernel)
    code_reviewer = create_agent("CodeReviewer", "code-reviewer-service", CUSTOM_API_URL, BEARER_TOKEN, kernel)

    # Group chat
    group_chat = AgentGroupChat(
        agents=[code_writer, code_reviewer],
        kernel=kernel
    )

    # Custom selection function
    async def select_agent(chat_history):
        try:
            last_msg = chat_history[-1]["content"].lower()
            if "review" in last_msg or "improve" in last_msg:
                logging.info("🎯 Selecting CodeReviewer")
                return code_reviewer
            else:
                logging.info("🎯 Selecting CodeWriter")
                return code_writer
        except Exception as e:
            logging.error(f"Error selecting agent: {e}", exc_info=True)
            return code_writer

    group_chat.select_agent_function = select_agent

    # Conversation
    messages = [
        {"role": "user", "content": "Write a Python program for a ping pong game."},
        {"role": "user", "content": "Review this code for best practices."}
    ]

    logging.info("💬 Starting conversation loop")

    for msg in messages:
        logging.info(f"🧍 User: {msg['content']}")
        try:
            async for response in group_chat.invoke(msg["content"]):
                if response and hasattr(response, "name"):
                    logging.info(f"✅ {response.name}: {response.content}")
                    print(f"\n🤖 {response.name}: {response.content}\n")
                else:
                    logging.warning("⚠️ Empty or invalid response structure received")

        except AgentChatException as ex:
            logging.error(f"❌ AgentChatException: {ex}", exc_info=True)
        except Exception as ex:
            logging.exception(f"❌ Unexpected error during chat: {ex}")

    logging.info("🏁 Conversation completed successfully")


# ---------------------------------------------------------------
# ✅ Run
# ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Execution interrupted by user.")
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
