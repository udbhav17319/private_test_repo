from semantic_kernel.contents import ChatMessageContent

# --- FIX: Add .message and .text attributes dynamically ---
if not hasattr(ChatMessageContent, "message"):
    def _get_message(self):
        return getattr(self, "content", None)
    def _set_message(self, value):
        setattr(self, "content", value)
    ChatMessageContent.message = property(_get_message, _set_message)

if not hasattr(ChatMessageContent, "text"):
    def _get_text(self):
        return getattr(self, "content", None)
    def _set_text(self, value):
        setattr(self, "content", value)
    ChatMessageContent.text = property(_get_text, _set_text)

# --- FIX: Ensure ChatMessageContent has .thread attribute ---
if not hasattr(ChatMessageContent, "thread"):
    @property
    def thread(self):
        return getattr(self, "_thread", None)
    @thread.setter
    def thread(self, value):
        setattr(self, "_thread", value)
    ChatMessageContent.thread = thread
# -------------------------------------------------------------------------------
