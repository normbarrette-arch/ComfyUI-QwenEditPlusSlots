"""
Text Encode Qwen Edit Plus (Slots)

Combines a multi-slot prompt builder with the Qwen-Image-Edit "Plus" text
encode. Up to 10 toggleable text slots are assembled into one prompt (shown in a
live, editable 'final_prompt' field), which is then encoded exactly like
ComfyUI's core TextEncodeQwenImageEditPlus node (comfy_extras/nodes_qwen.py) -
same vision-token scaling, reference latents, and llama template - producing
identical CONDITIONING. The assembled/edited text is also returned as STRING.

By default the prompt comes from the slots; flip use_manual_edit to encode the
hand-edited final_prompt text instead. The slot UI (slot_count hiding, inline
ON/OFF toggles, live preview) is driven by js/qwen_edit_plus_slots.js.

comfy.utils / node_helpers are imported lazily inside encode() so the module
imports with no heavy dependencies (keeps load + unit-testing light); inside
ComfyUI both are already loaded, so the lazy import is effectively free.
"""

import math

MAX_SLOTS = 10

# Verbatim from comfy_extras/nodes_qwen.py (ComfyUI 0.24.1) so encoding matches.
QWEN_EDIT_LLAMA_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
    "size, texture, objects, background), then explain how the user's text instruction "
    "should alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


class QwenEditPlusPromptSlots:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP",),
            "mode": (["Combine Enabled", "Select One"],),
            "select": ("INT", {
                "default": 1, "min": 1, "max": MAX_SLOTS, "step": 1,
                "tooltip": "Which slot to output in 'Select One' mode",
            }),
            "separator": ("STRING", {"default": ", "}),
            "slot_count": ("INT", {
                "default": 2, "min": 1, "max": MAX_SLOTS, "step": 1,
                "tooltip": "Number of visible text slots",
            }),
            "use_manual_edit": ("BOOLEAN", {
                "default": False, "label_on": "MANUAL", "label_off": "AUTO",
                "tooltip": "AUTO: encode the combined slots. MANUAL: encode the edited final_prompt text instead.",
            }),
            "final_prompt": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "Live preview of the combined slots. Turn use_manual_edit ON to hand-edit and encode this text instead.",
            }),
        }
        optional = {
            "vae": ("VAE",),
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
            "image4": ("IMAGE",),
        }
        for i in range(1, MAX_SLOTS + 1):
            optional[f"text_{i}"] = ("STRING", {
                "multiline": True, "default": "", "dynamicPrompts": True,
            })
            optional[f"enable_{i}"] = ("BOOLEAN", {
                "default": i == 1, "label_on": "ON", "label_off": "OFF",
            })
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "prompt")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = (
        "Assemble a prompt from up to 10 toggleable text slots (with a live, editable "
        "preview), then encode it with the Qwen Image Edit Plus pipeline (clip + "
        "optional vae + up to 4 reference images) to CONDITIONING. Also outputs the "
        "final text as STRING. Output matches the core TextEncodeQwenImageEditPlus."
    )

    @staticmethod
    def assemble_prompt(mode, select, separator, slot_count, slots):
        """Build the prompt string from the slot widgets. Pure and unit-testable."""
        count = max(1, min(int(slot_count), MAX_SLOTS))
        texts = [slots.get(f"text_{i}", "") or "" for i in range(1, count + 1)]
        enables = [bool(slots.get(f"enable_{i}", i == 1)) for i in range(1, count + 1)]

        if mode == "Select One":
            idx = max(1, min(int(select), count)) - 1
            return texts[idx].strip()

        parts = [t.strip() for t, e in zip(texts, enables) if e and t.strip()]
        return separator.join(parts)

    @classmethod
    def resolve_prompt(cls, use_manual_edit, final_prompt, mode, select, separator, slot_count, slots):
        """The text actually encoded: manual override, else the assembled slots."""
        if use_manual_edit:
            return final_prompt
        return cls.assemble_prompt(mode, select, separator, slot_count, slots)

    def encode(self, clip, mode, select, separator, slot_count,
               use_manual_edit=False, final_prompt="",
               vae=None, image1=None, image2=None, image3=None, image4=None, **kwargs):
        import comfy.utils
        import node_helpers

        prompt = self.resolve_prompt(
            use_manual_edit, final_prompt, mode, select, separator, slot_count, kwargs
        )

        # --- Qwen Image Edit Plus encode (mirrors comfy_extras/nodes_qwen.py) ---
        # image4 extends the core's 3 inputs; the model is optimized for <=3, so a
        # 4th reference may reduce consistency. Unused (None) inputs are skipped, so
        # leaving image4 unconnected behaves identically to the 3-image core node.
        ref_latents = []
        images = [image1, image2, image3, image4]
        images_vl = []
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=QWEN_EDIT_LLAMA_TEMPLATE)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return (conditioning, prompt)


NODE_CLASS_MAPPINGS = {"RB_QwenEditPlusPromptSlots": QwenEditPlusPromptSlots}
NODE_DISPLAY_NAME_MAPPINGS = {"RB_QwenEditPlusPromptSlots": "Text Encode Qwen Edit Plus (Slots)"}
