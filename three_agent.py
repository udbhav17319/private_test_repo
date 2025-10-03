import asyncio
import datetime
import os
import dotenv
import logging

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential
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

from local_python_plugin3 import LocalPythonPlugin  # Plugin for code execution

# Load .env
dotenv.load_dotenv()

# Azure OpenAI Config
azure_openai_endpoint = "https://eyq-incubator.america.fabric.ey.com/eyq/us/api/"
azure_openai_api_key = ""
#azure_openai_api_version = "gpt-4o-mini-2024-07-18"
azure_openai_deployment = "gpt-4o"

CODEWRITER_NAME = "CodeWriter"
CODEEXECUTOR_NAME = "CodeExecutor"
CODE_REVIEWER_NAME= "CodeReviewer"
TERMINATION_KEYWORD = "yes"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _create_kernel(service_id: str) -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment,
            api_key=azure_openai_api_key,
            #api_version=azure_openai_api_version,
        )
    )
    kernel.add_plugin(plugin_name="LocalCodeExecutionTool", plugin=LocalPythonPlugin())
    return kernel

def safe_result_parser(result):
    if not result.value:
        return None
    val = result.value
    if isinstance(val,list) and val:
        val=val[0]
    name=str(val).strip().lower()
    if "codeexecutor" in name:
        return CODEEXECUTOR_NAME
    if "codewriter" in name:
        return CODEWRITER_NAME  
    if "CodeReviewer" in name:
        return CODE_REVIEWER_NAME
    return None 

async def main():
    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=_create_kernel(CODEWRITER_NAME),
        name=CODEWRITER_NAME,
        instructions=f"""
            You are a highly skilled Python developer named {CODEWRITER_NAME}.
            Your job is to write clean, working Python code based on user requests.
            - Return only code, no explanations, no markdown, no extra text.
            - If external libraries are needed (like pygame), add pip install lines using subprocess.
            - Let the executor handle running the code. Do not run it yourself.
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
            - You run Python code and return output, errors, or results.
            - If a library is missing, install it using subprocess/pip.
            - If the code is GUI-based (pygame/tkinter), run it and wait for the window to close.
            - Respond in plain English summarizing the result. Do not invent outputs.
            - Do not explain code. Only report what actually happened.
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

    reviwer = ChatCompletionAgent(
        service_id=CODE_REVIEWER_NAME,
        kernel=_create_kernel(CODE_REVIEWER_NAME),
        name=CODE_REVIEWER_NAME,
        instructions=f"""
            You are an code reviewer agent named {CODE_REVIEWER_NAME}.
            - Your task is to review the code and give feedback based on that.
            - If the code is GUI-based (pygame/tkinter), run it and wait for the window to close.
            - Respond in plain English summarizing the result. Do not invent outputs.
            - Do not explain code. Only report what actually happened.
            """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=CODE_REVIEWER_NAME,
            temperature=0.2,
            max_tokens=1500,
            function_choice_behavior=FunctionChoiceBehavior.Required(),
        ),
    )
    # Selection strategy
    selection = KernelFunctionFromPrompt(
        function_name="select_next",
        prompt=f"""
            YOU are a decision function.
            Your job is to pick exactly one agent to respond next.
            Respond ONLY with one of the following exact names (no explationatio, no punctuation, no quotes):

            - {CODEWRITER_NAME}
            - {CODEEXECUTOR_NAME}
            - {CODE_REVIEWER_NAME}

            Rules:

            Return only the name. No other text.

            History:
            {{{{$history}}}}
            """
    )

    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=f"""
            Does the last message from {CODEEXECUTOR_NAME} indicate the task is complete?

            Say only "{TERMINATION_KEYWORD}" if:
            - The output shows the code has been executed.
            - Or, a GUI/game was launched (e.g., pygame).
            - Or, there are no follow-up steps mentioned.

            Say anything else otherwise.

            History:
            {{{{$history}}}}
            """

    )

    chat = AgentGroupChat(
        agents=[writer, executor,reviwer],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection,
            kernel=_create_kernel("selector"),
            #result_parser=lambda r: str(r.value[0]) if r.value else CODEWRITER_NAME,
            result_parser=safe_result_parser,
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[executor],
            function=termination,
            kernel=_create_kernel("terminator"),
            result_parser=lambda r: TERMINATION_KEYWORD in str(r.value[0]).lower(),
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
        
        # ✅ ADD this fix after the response is received
        if response.name == CODEEXECUTOR_NAME:
            print("✅ Task complete (executor finished).")
            break

        if chat.is_complete:
            print("✅ Task complete.")
            break

if __name__ == "__main__":
    asyncio.run(main())
