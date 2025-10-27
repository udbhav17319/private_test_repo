# =========================================================
# 🧰 CodeDebuggerAgent (Executes + Reports back for fixes)
# =========================================================
class CodeDebuggerAgent(Agent):
    """Runs Python code locally and returns execution result or error report."""

    def __init__(self):
        super().__init__(
            name="CodeDebuggerAgent",
            description="Executes Python code locally and reports success or errors for automatic fixing.",
        )

    # --- required abstract methods (all implemented) ---
    async def get_response(self, task, **kwargs):
        """Synchronous single-shot call used by orchestration."""
        return await self._execute_code(task, **kwargs)

    async def invoke(self, task, **kwargs):
        """Primary call path for orchestration."""
        return await self._execute_code(task, **kwargs)

    async def invoke_stream(self, task, **kwargs):
        """Streaming variant (yields a single result for now)."""
        yield await self._execute_code(task, **kwargs)
    # -----------------------------------------------------

    async def _execute_code(self, task, **kwargs):
        """Extract and execute Python code safely, preserving thread context."""
        thread = kwargs.get("thread", None)

        if isinstance(task, ChatMessageContent):
            task_text = task.content
            thread = getattr(task, "thread", thread)
        else:
            task_text = str(task)

        # find Python code blocks
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", task_text, re.DOTALL)
        if not code_blocks:
            return ChatMessageContent(
                name=self.name, role="assistant",
                content="⚠️ No Python code block found to execute.",
                thread=thread,
            )

        code = code_blocks[0].strip()

        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tf:
                tf.write(code)
                tf.flush()
                temp_path = tf.name

            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=20,
            )

            if result.returncode == 0:
                output = result.stdout.strip() or "✅ Code executed successfully (no output)."
                summary = "Execution successful."
            else:
                output = result.stderr.strip() or result.stdout.strip()
                summary = "Execution failed. Please fix the code."

        except subprocess.TimeoutExpired:
            output = "⏱️ Code execution timed out (20 s limit)."
            summary = "Execution failed due to timeout."
        except Exception as e:
            output = f"❌ Runtime error: {e}"
            summary = "Execution failed due to runtime exception."
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return ChatMessageContent(
            name=self.name,
            role="assistant",
            content=(
                f"{summary}\n\n💻 **Execution Output:**\n```\n{output}\n```"
                "\nIf there was an error, please analyze it and fix the Python code."
            ),
            thread=thread,
        )
