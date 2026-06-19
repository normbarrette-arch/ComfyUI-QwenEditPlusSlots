# ComfyUI-QwenEditPlusSlots

A single ComfyUI custom node — **Text Encode Qwen Edit Plus (Slots)** — that
assembles a prompt from up to 10 toggleable, reorderable text slots (with a live,
editable preview) and encodes it through the **Qwen Image Edit Plus** pipeline,
outputting CONDITIONING plus the final prompt as STRING. It's a drop-in
replacement for the core `TextEncodeQwenImageEditPlus` node with a built-in
prompt builder.

## Install

**Via ComfyUI-Manager:** *Install via Git URL* → paste this repo's URL, then restart.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/ComfyUI-QwenEditPlusSlots.git
```
Restart ComfyUI (hard-refresh the browser too, so the slot UI JavaScript loads).
The node appears under **advanced/conditioning** as *Text Encode Qwen Edit Plus (Slots)*.

## Node: Text Encode Qwen Edit Plus (Slots)

**Inputs**
- `clip` (CLIP) — required.
- `vae` (VAE), `image1`..`image4` (IMAGE) — optional Qwen reference images. The
  core node stops at 3; `image4` is an opt-in extra (the model is optimized for
  ≤3, so a 4th may reduce consistency). Unused image inputs are skipped, so an
  unconnected `image4` behaves exactly like the 3-image core node.
- `mode` — `Combine Enabled` (join enabled, non-empty slots with `separator`) or
  `Select One` (output a single chosen slot).
- `select`, `separator`, `slot_count` (1–10) — combine/select controls and how
  many slots are visible.
- `use_manual_edit` — `AUTO`: encode the combined slots; `MANUAL`: encode the
  hand-edited `final_prompt` text instead.
- `final_prompt` — live preview of the combined slots while AUTO (read-only);
  becomes a free, encoded-verbatim edit field while MANUAL.
- `text_1..10` + `enable_1..10` — the slots, each with a native ON/OFF toggle and
  **▲ up / ▼ down** reorder buttons. Order persists with the workflow.

**Outputs**
- `conditioning` (CONDITIONING) — identical to what core `TextEncodeQwenImageEditPlus`
  produces for the same final prompt + images.
- `prompt` (STRING) — the final text that was encoded (assembled or manual).

## Credits & licensing

- The encode pipeline mirrors ComfyUI core's `TextEncodeQwenImageEditPlus`
  (`comfy_extras/nodes_qwen.py`), which is **GPL-3.0**; this project is therefore
  licensed **GPL-3.0** — see [LICENSE](LICENSE).
- The slot UI JavaScript is adapted from
  [kymeraj/comfyui-prompt-builder](https://github.com/kymeraj/comfyui-prompt-builder)
  (MIT) — see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
