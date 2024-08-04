"""
To limit num. msgs in chat history
"""


class MessageQueue:
    """
    Queue to limit the max msgs in chat history
    """

    def __init__(self, max_pairs):
        self.max_pairs = max_pairs
        self.messages = []

    def add_message(self, message):
        """
        Add a message
        """
        self.messages.append(message)
        self._trim_old_messages()

    def _trim_old_messages(self):
        """
        Remove old message pair if exceed MAX_PAIRS_IN_CHAT
        """
        while len(self.messages) // 2 > self.max_pairs:
            self.messages.pop(0)
            self.messages.pop(0)

    def get_messages(self):
        """
        Return the list of messages in the Chat history
        """
        return self.messages
