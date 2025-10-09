    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        # Convert response JSON to a string to avoid unhashable dicts in SK internals
        resp_json = resp.json()
        # Return a string (JSON serialized). Kernel/plugin results should be primitives/strings.
        return json.dumps(resp_json, ensure_ascii=False)
    except requests.RequestException as e:
        logging.error(f"Error executing code in container app session pool: {e}")
        # Return a stringified error so it's safe to include in SK function result content
        return json.dumps({"error": str(e)})


import asyncio
import datetime
import json
import logging
import os
import uuid
import tempfile
import base64

from github import Github
from github.Auth import Token


import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
import requests

from semantic_kernel import Kernel
#from semantic_kernel.orchestration.function_tool import FunctionTool
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import KernelFunctionTerminationStrategy
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt


from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_plugin import KernelPlugin

import re
import random
import string
from datetime import datetime

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load .env
dotenv.load_dotenv()

# Azure OpenAI Config
azure_openai_endpoint = "https://eyq-incubator.america.fabric.ey.com/eyq/us/api/"
azure_openai_api_key = ""
azure_openai_deployment = "gpt-4o"

CODEWRITER_NAME = "CodeWriter"
CODEEXECUTOR_NAME = "CodeExecutor"
CODE_REVIEWER_NAME= "CodeReviewer"
APIBUILDER_NAME = "APIBUILDER"
TERMINATION_KEYWORD = "yes"

#final_code=None

# Use DefaultAzureCredential for Managed Identity
default_credential = DefaultAzureCredential()

# Global cached kernels
kernels = {}

def extract_trigger_name(js_code: str) -> str | None:
    # 1. Try direct trigger pattern
    direct_pattern = r"app\.http\s*\(\s*['\"]([\w\-]+)['\"]"
    match = re.search(direct_pattern, js_code)
    if match:
        return match.group(1)

    # 2. Try variable reference pattern
    var_ref_pattern = r"app\.http\s*\(\s*([\w\-]+)\s*,"
    match = re.search(var_ref_pattern, js_code)
    if match:
        var_name = match.group(1)
        # Find variable assignment for that variable name
        var_assign_pattern = rf"(?:const|let|var)\s+{re.escape(var_name)}\s*=\s*['\"]([\w\-]+)['\"]"
        assign_match = re.search(var_assign_pattern, js_code)
        if assign_match:
            return assign_match.group(1)

    return None


def gitpushfile(file_path,file_extension,repo_name):
    # Your GitHub personal access token (PAT)
    TOKEN = ""
    REPO_NAME = f""  # e.g. 
    #FILE_PATH = "index.html"        # path in repo
    #LOCAL_FILE = "local_test.py" # local file to upload/update
    LOCAL_FILE = file_path 
    COMMIT_MESSAGE = "Added html and javascript file"

    # Authenticate
    '''g = Github(TOKEN)
    repo = g.get_repo(REPO_NAME)'''
    g = Github(auth=Token(f"{TOKEN}"))
    repo = g.get_repo(REPO_NAME)

    # Read local file content
    with open(LOCAL_FILE, "rb") as f:
        content = f.read()
    random_name = str(uuid.uuid4())
    if file_extension=="js":
        git_LOCAL_FILE = f"src/functions/{random_name}.{file_extension}"
    else:
        git_LOCAL_FILE = f"{random_name}.{file_extension}"
    # Create new file if it doesn't exist
    repo.create_file(git_LOCAL_FILE, COMMIT_MESSAGE, content)
    print("File created!")
    logging.info("File created!")


    '''try:
        # Check if file exists
        contents = repo.get_contents(FILE_PATH)
        # Update existing file
        repo.update_file(contents.path, COMMIT_MESSAGE, content, contents.sha)
        print("File updated!")
        logging.info("File updated!")
    except:
        # Create new file if it doesn't exist
        repo.create_file(FILE_PATH, COMMIT_MESSAGE, content)
        print("File created!")
        logging.info("File created!")'''
    return random_name


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
    
    # Add the plugin
    #kernel.plugins.add(code_plugin)
    kernel.plugins["CodeExecutionPlugin"] = code_plugin

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

def get_container_app_token(scope: str = "https://dynamicsessions.io/.default"):
    """Get Managed Identity token for Azure Container Apps Session Pool API."""
    try:
        token = default_credential.get_token(scope)
        return token.token
    except Exception as ex:
        logging.error(f"Failed to obtain managed identity token for session pool: {ex}")
        raise



@kernel_function(name="ExecutePythonCode", description="Executes Python code in a secure container and returns output.")

def execute_code_in_container(code: str):

    # Build the full session pool URL
    #base_url = f"https://{session_pool_name}.{env_id}.{region}.azurecontainerapps.io"
    #url = f"{base_url}{execute_path}?identifier={session_id}"

    random_name = str(uuid.uuid4())
    url=f""

    # Get a token for the dynamic sessions audience
    token = get_container_app_token()
    #logging.info(f"{token}")
    logging.info("------------------------------------------------------------")
    #logging.info(f"{code}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    #payload = {"code": code}
    payload={
    "properties": {
        "codeInputType": "inline",
        "executionType": "synchronous",
        "code": f"{code}"
    }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logging.error(f"Error executing code in container app session pool: {e}")
        raise


code_plugin = KernelPlugin(
    name="CodeExecutionPlugin",
    functions=[execute_code_in_container]
)


async def run_multi_agent(prompt: str, max_iterations: int = 10):

    agents_involved=""

    suffix = ''.join(random.choices('abcdef0123456789', k=4))
    
    writer = ChatCompletionAgent(
        service_id=CODEWRITER_NAME,
        kernel=_create_kernel(CODEWRITER_NAME),
        name=CODEWRITER_NAME,
        instructions=f"""
            You are a highly skilled javascript and html developer named {CODEWRITER_NAME}.
            Your job is to write clean, working javascript and html code based on user requests.
            - Return only code, no explanations, no markdown, no extra text.
            - Always produce a full runnable script.
            - Generate a fully self-contained HTML file with inline CSS and JavaScript. 
            The code must not depend on any external files, CDNs, or frameworks. 
            It should include <html>, <head>, <body>, <style>, and <script> tags, 
            and be directly runnable by saving it as an .html file and opening it in a browser. 
            Use vanilla HTML, CSS, and JavaScript only. 
            Example: a playable Snake game with arrow key controls and score tracking.
            - Let the executor handle running the code. Do not run it yourself.
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
            - Run Python code and return output/errors.
            - If a library is missing, install it.
            - Respond in plain English summarizing results.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=CODEEXECUTOR_NAME,
            temperature=0.2,
            max_tokens=1500,
            function_choice_behavior=FunctionChoiceBehavior.Required(filters={"included_plugins": ["CodeExecutionPlugin"]}),
        )
    )

    reviewer = ChatCompletionAgent(
        service_id=CODE_REVIEWER_NAME,
        kernel=_create_kernel(CODE_REVIEWER_NAME),
        name=CODE_REVIEWER_NAME,
        instructions=f"""
            You are a senior code reviewer named {CODE_REVIEWER_NAME}.
            - Review code for correctness, readability, performance, and best practices.
            - Suggest improvements concisely.
            - Do not execute code unless explicitly asked.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=CODE_REVIEWER_NAME,
            temperature=0.2,
            max_tokens=1500,
            function_choice_behavior=FunctionChoiceBehavior.Required(),
        ),
    )

    apibuilder = ChatCompletionAgent(
        service_id=APIBUILDER_NAME,
        kernel=_create_kernel(APIBUILDER_NAME),
        name=APIBUILDER_NAME,
        instructions = f"""
            You are {APIBUILDER_NAME}, an expert in building REST APIs as Azure Functions in Node.js which
            is fully deplyable.
            Your goal is to generate **complete deployable Azure Function apps** based on user requests. 
            Requirements:
            1. **Azure LLM Integration**
            - Use Azure OpenAI / LLM for processing.
            - Read **API key, endpoint, and deployment** from environment variables:
                - OPENAI_KEY
                - OPENAI_ENDPOINT
                - OPENAI_DEPLOYMENT
            - The API should perform translations or other LLM tasks as requested.
            - use apiVersion "2023-05-15"
            - try to use axios library and chat completion as openai model is gpt-4o
            - provide code as per azure gpt-4o
            - use directly api endpoint and key for authentication like "baseUrl/openai/deployments/deploymentName/chat/completions?api-version=apiVersion"

            2. **Input Handling**
            - accept all parameter in json body only.
            - Handle everything **on the fly**, no blob storage.
            - Use getChatCompletions for chat models like gpt-4o.
            - do all the error handling

            3. **Node.js Azure Functions v4+ style**
            - use module.exports for azure function
            - Don't Include **function.json**.

            4. **Output Requirements**
            - Return **only the code files**: `index.js`
            - The code must be **ready to deploy to Azure Functions**.
            - Do not include explanations, comments, or extra text.

            5. **Example API to Implement**
            - Translation API:
                - Accepts text or file.
                - Uses Azure LLM for translation.
                - Returns translated text as JSON.

            User will provide:
            - The **type of API** (like translation) and any specific **details**.
            - The **agent must generate the full deployable Node.js Azure Function**.

            Always ensure:
            - **Environment variables are used** for sensitive info.
            - Code handles **both text and file inputs in-memory**.
            - Output is **directly usable** in Azure Functions.
            - don't add **```javascript** in start or end
            - do not include file name or javascript in the response. 
            - only give final code
            - accept all parameter in body only.
            - the code script should be handeling all errors and covers all cases.
            - Generate a JavaScript Azure Function using the Node.js v4 programming model with @azure/functions. 
            - The function should:
                Be an HTTP-triggered function
                Accept both GET and POST requests
                Return a greeting message using the name
                Use app.http() syntax from @azure/functions
                Include logging using context.log
                The trigger name in app.http() must be hardcoded as 'translationAPI_{suffix}'.
                
            """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=APIBUILDER_NAME,
            temperature=0.1,
            max_tokens=2000,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    selection = KernelFunctionFromPrompt(
        function_name="select_next",
        prompt=f"""
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
            - If the user asks to build an API (e.g., Azure Function, REST API) → {APIBUILDER_NAME}.
            - Return ONLY the agent name, no extra text.

            Conversation history:
            {{{{$history}}}}
        """
    )

    termination = KernelFunctionFromPrompt(
        function_name="check_done",
        prompt=f"""
            Determine if the user's request has been fully completed.
            Say only "{TERMINATION_KEYWORD}" if:
            - The correct agent has responded once with output/code.
            Otherwise, respond with anything else.

            Conversation history:
            {{{{$history}}}}
        """
    )

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
            result_parser=lambda r: TERMINATION_KEYWORD in str(r.value[0]).lower(),
            #result_parser=safe_result_parser,
            #agent_variable_name="agents",
            history_variable_name="history",
            maximum_iterations=max_iterations,
        ),
    )

    await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=prompt))

    code_output = None
    #final_code = None
    while True:
        async for response in chat.invoke():
            #code_output= response.content
            #log(response.name)
            agents_involved += f'{response.name}\n'
            logging.info(f"\n🤖 {response.name}:\n{response.content}\n")
            if response.name == CODEWRITER_NAME:
                # Save code to temp file for download
                code = response.content
                file_name = f"generated_{uuid.uuid4().hex}.py"
                file_path = os.path.join(tempfile.gettempdir(), file_name)
                
                with open(file_path, 'w') as f:
                    f.write(code)
                
                logging.info("----------------------------------------------")
                logging.info(code)
                git_name=gitpushfile(file_path,"html","test")
                logging.info(file_path)  #https://udbhavgupta.github.io/test
                code_output = {"code_file": f'https://udbhavgupta.github.io/test/{git_name}.html', "code": code}
                # Execute in container
                exec_result = execute_code_in_container(code)
                #code_output = exec_result
                #code_output = git_name
                #final_code = code
                #break
                return code_output
            
            if response.name == APIBUILDER_NAME:
                # Save code to temp file for download
                code = response.content
                
                file_name = f"generated_{uuid.uuid4().hex}.js"
                file_path = os.path.join(tempfile.gettempdir(), file_name)
                
                with open(file_path, 'w') as f:
                    f.write(code)

                logging.info("----------------------------------------------")
                logging.info(code)
                trigger_name = extract_trigger_name(code)
                git_name=gitpushfile(file_path,"js","agentic-api-azurepush2")
                logging.info(file_path)  #https://udbhavgupta.github.io/test
                #code_output = {"code_file": f'https://udbhavgupta.github.io/test/{git_name}', "code": code}
                code_output = {"code_file": f"https://agentic-api-testing2-d5gzh9h6c9cectfb.eastus-01.azurewebsites.net/api/{trigger_name}", "code": code}
                #code_output = f"https://agentic-api-testing2-d5gzh9h6c9cectfb.eastus-01.azurewebsites.net/api/{trigger_name}"
                #code_output=f"agentic-api-testing-hzadf4hfdbfkdbfb.eastus-01.azurewebsites.net/{git_name}"
                #print(code_output)
                return code_output


            '''if response.name == CODEEXECUTOR_NAME:
                code_output= response.content'''

        if chat.is_complete:
            print("Task complete.")
            logging.info("Task complete.")
            break
    code_output["logs"]= agents_involved
    #print(code_output)
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
        #result["logs"]=logs
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

@app.function_name(name="MultiAgentFunction1")
@app.route(route="multiagent1", methods=["POST"])
async def multiagent_function(req: func.HttpRequest) -> func.HttpResponse:
    return await main(req)

#if __name__ == "__main__":
#    asyncio.run(run_multi_agent("I am working for a large enterprise and I need to showcase Semantic Kernel is the way to go for orchestration. Write a python script that calculates the most financially valuable opportunities for the enterprise. Execute the generated code to check for error.",10))
#    #print(logs)
