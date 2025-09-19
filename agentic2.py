import asyncio
import datetime
import json
import logging
import os
import uuid
import tempfile

import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
import requests

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

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Azure OpenAI Config from environment
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Azure Container App Session Pool endpoint
container_app_url = os.getenv("CONTAINER_APP_URL")

CODEWRITER_NAME = "CodeWriter"
CODEEXECUTOR_NAME = "CodeExecutor"
TERMINATION_KEYWORD = "yes"

# Use DefaultAzureCredential for Managed Identity
default_credential = DefaultAzureCredential()

# Global cached kernels
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
    try:
        token = default_credential.get_token(scope or "https://management.azure.com/.default")
        return token.token
    except Exception as ex:
        logging.error(f"Failed to obtain managed identity token: {ex}")
        raise

def execute_code_in_container(code: str):
    """Send code to Azure Container App session pool for execution."""
    token = get_container_app_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"code": code}
    try:
        resp = requests.post(container_app_url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, HttpResponseError) as e:
        logging.error(f"Error executing code in container app: {e}")
        raise

async def run_multi_agent(prompt: str, max_iterations: int = 10):
    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=_create_kernel(CODEWRITER_NAME),
        name=CODEWRITER_NAME,
        instructions=f"""
You are a highly skilled Python developer named {CODEWRITER_NAME}.
Your job is to write clean, working Python code based on user requests.
- Return only code, no explanations, no markdown, no extra text.
- Always produce a full runnable script.
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
You are an execution agent named {CODEEXECUTOR_NAME}.
- You send Python code to the Azure Container App session pool for execution.
- Return only the actual execution result from the container.
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
YOU are a decision function.

Your job is to pick exactly one agent to respond next.

Respond ONLY with one of the following exact names (no punctuation, no quotes):

- {CODEWRITER_NAME}
- {CODEEXECUTOR_NAME}

Rules:
- After the user, it must be {CODEWRITER_NAME},
- After {CODEWRITER_NAME}, it must be {CODEEXECUTOR_NAME},
- After {CODEEXECUTOR_NAME}, stop.

History:
{{{{history}}}}
"""
    )

    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=f"""
Does the last message from {CODEEXECUTOR_NAME} indicate the task is complete?
Say only "{TERMINATION_KEYWORD}" if executed.
Anything else otherwise.

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
            # Save code to temp file for download
            code = response.content
            file_name = f"generated_{uuid.uuid4().hex}.py"
            file_path = os.path.join(tempfile.gettempdir(), file_name)
            with open(file_path, 'w') as f:
                f.write(code)
            code_output = {"code_file": file_path, "code": code}
            # Execute in container
            exec_result = execute_code_in_container(code)
            code_output["execution_result"] = exec_result

    return code_output

async def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        prompt = body.get("prompt")
        max_iterations = int(body.get("max_iterations", 10))
        if not prompt:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'prompt' in request body"}),
                status_code=400,
                mimetype="application/json"
            )
        result = await run_multi_agent(prompt, max_iterations)
        return func.HttpResponse(
            json.dumps(result, default=str),
            status_code=200,
            mimetype="application/json"
        )
    except ClientAuthenticationError as cae:
        return func.HttpResponse(json.dumps({"error": str(cae)}), status_code=401, mimetype="application/json")
    except Exception as e:
        logging.exception("Unhandled exception")
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500, mimetype="application/json")

# The Azure Functions entry point
app = func.FunctionApp()

@app.function_name(name="MultiAgentFunction")
@app.route(route="multiagent", methods=["POST"])
async def multiagent_function(req: func.HttpRequest) -> func.HttpResponse:
    return await main(req)
