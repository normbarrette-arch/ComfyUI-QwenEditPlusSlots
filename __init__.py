"""ComfyUI-QwenEditPlusSlots - Qwen-workflow custom nodes.

Nodes:
  - Text Encode Qwen Edit Plus (Slots): multi-slot prompt builder + Qwen
    Image Edit Plus encode (qwen_edit_plus_slots.py, with js/ frontend).
  - Skin Realism (De-Plastic): post-process that fixes plastic-looking
    AI-rendered skin (skin_realism.py).
"""

from .qwen_edit_plus_slots import (
    NODE_CLASS_MAPPINGS as _slots_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _slots_names,
)
from .skin_realism import (
    NODE_CLASS_MAPPINGS as _skin_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _skin_names,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _c, _n in ((_slots_classes, _slots_names), (_skin_classes, _skin_names)):
    NODE_CLASS_MAPPINGS.update(_c)
    NODE_DISPLAY_NAME_MAPPINGS.update(_n)

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
