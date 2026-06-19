"""ComfyUI-QwenEditPlusSlots - the Text Encode Qwen Edit Plus (Slots) custom node."""

from .qwen_edit_plus_slots import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
