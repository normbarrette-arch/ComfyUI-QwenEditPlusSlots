# ComfyUI-QwenEditPlusSlots

ComfyUI custom nodes for Qwen Image Edit workflows:

- **Text Encode Qwen Edit Plus (Slots)** — assembles a prompt from up to 10
  toggleable, reorderable text slots (with a live, editable preview) and encodes
  it through the **Qwen Image Edit Plus** pipeline, outputting CONDITIONING plus
  the final prompt as STRING. A drop-in replacement for the core
  `TextEncodeQwenImageEditPlus` node with a built-in prompt builder.
- **Skin Realism (De-Plastic)** — post-processing that fixes the plastic,
  waxy skin Qwen (and diffusion models generally) often render on people.

## Install

**Via ComfyUI-Manager:** *Install via Git URL* → paste this repo's URL, then restart.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/normbarrette-arch/ComfyUI-QwenEditPlusSlots.git
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

## Node: Skin Realism (De-Plastic)

`image/postprocessing` category. Fixes plastic-looking AI-rendered skin by
**synthesizing** what the render lacks — no reference photo needed:

- **pores** — band-passed micro-texture, midtone-weighted, automatically
  attenuated where the image already has texture (never double-textures).
- **grain** — fine luminance grain that unifies the treated region.
- **sheen_reduction** — soft-knee compression of skin highlights (kills the
  waxy specular look).
- **mottling** — very low-frequency color variation (subsurface redness).

**Targeting:** by default it detects skin-colored regions (approximate,
color-based, cleaned + feathered). Wire any **MASK** into `mask` (e.g. from a
person/face segmenter) to override precisely. `detect_skin` OFF + no mask =
treat the whole frame.

**Inputs:** `image`, master `strength`, `texture_scale` (pore size; ~1.0 for a
face filling a ~1024px frame), the four effect sliders, `detect_skin`, `seed`
(deterministic texture), optional `mask`.

**Outputs:** `image`, plus `treated_mask` so you can inspect exactly what was
touched.

Pure torch + numpy — no extra dependencies.
