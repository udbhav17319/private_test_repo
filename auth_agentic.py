import asyncio
import logging
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents.group_chat import AgentGroupChat
from semantic_kernel.exceptions import AgentChatException

# ---------------------------------------------------------------
# ✅ Setup Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------
# ✅ Custom Mock Chat Completion Class
# ---------------------------------------------------------------
class CustomChatCompletion(ChatCompletionClientBase):
    """
    A mock AI chat completion client that works with Semantic Kernel.
    This avoids any API call and just simulates responses.
    """

    def __init__(self, service_id: str, ai_model_id: str = "mock-model"):
        super().__init__()
        self.service_id = service_id
        self.ai_model_id = ai_model_id
        logging.info(f"✅ Initialized CustomChatCompletion (service_id={service_id}, ai_model_id={ai_model_id})")

    async def complete_chat(self, context, settings=None):
        """
        Called by SK when the agent generates a message.
        """
        user_message = ""

        try:
            if hasattr(context, "messages") and context.messages:
                user_message = context.messages[-1].content
            else:
                user_message = str(context)
        except Exception as e:
            logging.error(f"[{self.service_id}] Failed to extract message: {e}")
            user_message = "[unknown message]"

        logging.debug(f"[{self.service_id}] User message: {user_message}")

        # Simulate deterministic agent responses
        if "code" in user_message.lower():
            response = f"[{self.service_id}] Here is the Python code for your request."
        elif "review" in user_message.lower():
            response = f"[{self.service_id}] I have reviewed your code, and it looks clean and efficient!"
        else:
            response = f"[{self.service_id}] Echo: {user_message}"

        logging.debug(f"[{self.service_id}] Generated response: {response}")
        return response

# ---------------------------------------------------------------
# ✅ Helper Function to Create Agent
# ---------------------------------------------------------------
def create_agent(name: str, service_id: str, kernel: Kernel):
    logging.info(f"🧠 Creating agent '{name}' with service_id '{service_id}'")

    # Register mock completion as SK service
    chat_client = CustomChatCompletion(service_id=service_id)
    kernel.add_service(chat_client, service_id=service_id)

    # Create agent
    agent = ChatCompletionAgent(
        service_id=service_id,
        kernel=kernel,
        name=name,
        description=f"{name} specialized in {service_id} tasks"
    )

    logging.info(f"✅ Agent '{name}' created successfully")
    return agent

# ---------------------------------------------------------------
# ✅ Main Function
# ---------------------------------------------------------------
async def main():
    logging.info("🚀 Starting Semantic Kernel Multi-Agent Chat Demo")

    # Initialize kernel
    kernel = Kernel()
    logging.info("✅ Kernel initialized")

    # Create mock agents
    code_writer = create_agent("CodeWriter", "code-writer-service", kernel)
    code_reviewer = create_agent("CodeReviewer", "code-reviewer-service", kernel)

    # Create group chat with both agents
    group_chat = AgentGroupChat(
        agents=[code_writer, code_reviewer],
        kernel=kernel
    )

    # Define agent selection logic
    async def select_agent(chat_history):
        try:
            last_msg = chat_history[-1]["content"].lower()
            if "code" in last_msg:
                logging.info("🤖 Selecting CodeWriter agent")
                return code_writer
            else:
                logging.info("🤖 Selecting CodeReviewer agent")
                return code_reviewer
        except Exception as e:
            logging.error(f"Error in select_agent: {e}")
            return code_writer

    group_chat.select_agent_function = select_agent

    # Conversation flow simulation
    messages = [
        {"role": "user", "content": "ping pong game code"},
        {"role": "user", "content": "can you review this code?"}
    ]

    logging.info("💬 Beginning conversation loop")

    for msg in messages:
        try:
            logging.info(f"🗣️ User: {msg['content']}")
            async for response in group_chat.invoke(msg["content"]):
                if response and hasattr(response, "name"):
                    logging.info(f"✅ Response from {response.name}: {response.content}")
                    print(f"\n🤖 {response.name}: {response.content}\n")
                else:
                    logging.warning("⚠️ Empty response or invalid structure received")

        except AgentChatException as ex:
            logging.error(f"❌ AgentChatException: {ex}", exc_info=True)
        except Exception as ex:
            logging.exception(f"❌ Unexpected error: {ex}")

    logging.info("🏁 Conversation completed successfully")

# ---------------------------------------------------------------
# ✅ Entry Point
# ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Execution interrupted by user.")
    except Exception as e:
        logging.exception(f"Fatal error in main(): {e}")
