from transformers import AsyncTextIteratorStreamer


class SmartAdaptStreamer(AsyncTextIteratorStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Remove "assistant\n\n" if it appears at the beginning
        if text.startswith("assistant\n\n"):
            text = text[len("assistant\n\n"):]  # Remove the prefix
        super().on_finalized_text(text, stream_end)
