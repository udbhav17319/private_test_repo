import asyncio
import os
import uuid
import tempfile
import json
import logging
import requests

from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

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

logging.basicConfig(level=logging.INFO)

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
container_app_url = os.getenv("CONTAINER_APP_URL")

CODEWRITER_NAME = "CodeWriter"
CODEEXECUTOR_NAME = "CodeExecutor"
TERMINATION_KEYWORD = "yes"

default_credential = DefaultAzureCredential()
kernels = {}

def _create_kernel(service_id: str) -> Kernel:
    if service_id in kernels:
        return kernels[service_id]
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment,
            api_key=azure_openai_api_key,
        )
    )
    kernels[service_id] = kernel
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
    return None

def get_container_app_token(scope: str = None):
    token = default_credential.get_token(scope or "https://management.azure.com/.default")
    return token.token

def execute_code_in_container(code: str):
    token = get_container_app_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"code": code}
    resp = requests.post(container_app_url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()

async def run_multi_agent(prompt: str, max_iterations: int = 10):
    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=_create_kernel(CODEWRITER_NAME),
        name=CODEWRITER_NAME,
        instructions=f"""
You are a highly skilled Python developer named {CODEWRITER_NAME}.
Return only full runnable Python code.
""",
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=CODEWRITER_NAME,
            temperature=0.2,
            max_tokens=2000,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    executor = ChatCompletionAgent(
        service_id=CODEEXECUTOR_NAME,
        kernel=_create_kernel(CODEEXECUTOR_NAME),
        name=CODEEXECUTOR_NAME,
        instructions=f"""
You are {CODEEXECUTOR_NAME}. Send Python code to Azure Container App and return execution result.
""",
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=CODEEXECUTOR_NAME,
            temperature=0.2,
            max_tokens=1500,
            function_choice_behavior=FunctionChoiceBehavior.Required(),
        ),
    )

    selection = KernelFunctionFromPrompt(
        function_name="select_next",
        prompt=f"""
Pick one agent:

- {CODEWRITER_NAME}
- {CODEEXECUTOR_NAME}

After user -> {CODEWRITER_NAME}
After {CODEWRITER_NAME} -> {CODEEXECUTOR_NAME}
After {CODEEXECUTOR_NAME} stop.

History:
{{{{history}}}}
"""
    )

    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=f"""
Say "{TERMINATION_KEYWORD}" if {CODEEXECUTOR_NAME} is done.

History:
{{{{history}}}}
"""
    )

    chat = AgentGroupChat(
        agents=[writer, executor],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection,
            kernel=_create_kernel("selector"),
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
            maximum_iterations=max_iterations,
        ),
    )

    await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=prompt))

    code_output = None
    async for response in chat.invoke():
        if response.name == CODEWRITER_NAME:
            code = response.content
            file_name = f"generated_{uuid.uuid4().hex}.py"
            file_path = os.path.join(tempfile.gettempdir(), file_name)
            with open(file_path, 'w') as f:
                f.write(code)
            code_output = {"code_file": file_path, "code": code}
            exec_result = execute_code_in_container(code)
            code_output["execution_result"] = exec_result
    return code_output


prompt = "Write Python code to print the first 10 Fibonacci numbers"
result = asyncio.run(run_multi_agent(prompt))
print(result)
