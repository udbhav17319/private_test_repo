def get_container_app_token(scope: str = "https://dynamicsessions.io/.default"):
    """Get Managed Identity token for Azure Container Apps Session Pool API."""
    try:
        token = default_credential.get_token(scope)
        return token.token
    except Exception as ex:
        logging.error(f"Failed to obtain managed identity token for session pool: {ex}")
        raise


def execute_code_in_container(code: str):
    """
    Send code to Azure Container App Session Pool for execution.
    Assumes env variables:
      - SESSION_POOL_NAME
      - SESSION_POOL_ENV_ID
      - SESSION_POOL_REGION
      - SESSION_ID (or we generate one)
      - EXECUTE_PATH (e.g. '/execute')
    """
    session_pool_name = os.getenv("SESSION_POOL_NAME")
    env_id = os.getenv("SESSION_POOL_ENV_ID")
    region = os.getenv("SESSION_POOL_REGION", "eastus")
    session_id = os.getenv("SESSION_ID", str(uuid.uuid4()))
    execute_path = os.getenv("EXECUTE_PATH", "/execute")

    # Build the full session pool URL
    base_url = f"https://{session_pool_name}.{env_id}.{region}.azurecontainerapps.io"
    url = f"{base_url}{execute_path}?identifier={session_id}"

    # Get a token for the dynamic sessions audience
    token = get_container_app_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"code": code}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logging.error(f"Error executing code in container app session pool: {e}")
        raise
