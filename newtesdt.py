from semantic_kernel.contents import ChatMessageContent

# Monkey patch
if not hasattr(ChatMessageContent, "message"):
    def _get_message(self): 
        return getattr(self, "content", None)
    def _set_message(self, value): 
        setattr(self, "content", value)
    ChatMessageContent.message = property(_get_message, _set_message)

# Test
msg = ChatMessageContent(role="assistant", content="Hello World!")
print("Message:", msg.message)   # should print Hello World!
msg.message = "Changed!"
print("Content:", msg.content)   # should print Changed!
