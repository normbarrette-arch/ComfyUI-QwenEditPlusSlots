/*
 * Dynamic slot UI + live prompt preview + per-slot reorder for
 * "Text Encode Qwen Edit Plus (Slots)" (RB_QwenEditPlusPromptSlots).
 *
 * - Each slot uses a native ComfyUI BOOLEAN toggle (renders reliably on legacy
 *   and Node 2.0 frontends).
 * - slot_count hides unused slots.
 * - final_prompt live-previews the combined slots while use_manual_edit is OFF
 *   (read-only) and becomes a hand-edit field (encoded verbatim) while ON.
 * - Per-slot move up/down buttons reorder by swapping the text + enable VALUES
 *   with the neighbor (no Python change; the assembler reads slots top-to-bottom).
 *   Buttons are serialize:false and are interleaved without disturbing the order
 *   of serializable widgets, so saved workflows stay aligned.
 *
 * assemblePrompt() mirrors the Python resolve/assemble logic so the preview
 * matches what gets encoded. The slot_count dynamic-hide approach is inspired by
 * kymeraj/comfyui-prompt-builder (MIT): https://github.com/kymeraj/comfyui-prompt-builder
 */
import { app } from "../../scripts/app.js";

const MAX_SLOTS = 10;
const NODE_CLASS = "RB_QwenEditPlusPromptSlots";

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function toggleWidget(node, widget, show) {
    if (!widget) return;
    widget.hidden = !show;
    if (widget.inputEl) {
        widget.inputEl.style.display = show ? "" : "none";
    }
}

// Mirror of QwenEditPlusPromptSlots.assemble_prompt (Python).
function assemblePrompt(node) {
    const mode = getWidget(node, "mode")?.value;
    const select = parseInt(getWidget(node, "select")?.value ?? 1, 10) || 1;
    const separator = (getWidget(node, "separator")?.value ?? ", ").toString();
    let count = parseInt(getWidget(node, "slot_count")?.value ?? 1, 10) || 1;
    count = Math.max(1, Math.min(count, MAX_SLOTS));

    const texts = [];
    const enables = [];
    for (let i = 1; i <= count; i++) {
        texts.push((getWidget(node, `text_${i}`)?.value ?? "").toString());
        enables.push(!!getWidget(node, `enable_${i}`)?.value);
    }

    if (mode === "Select One") {
        const idx = Math.max(1, Math.min(select, count)) - 1;
        return (texts[idx] ?? "").trim();
    }

    const parts = [];
    for (let i = 0; i < count; i++) {
        const t = texts[i].trim();
        if (enables[i] && t) parts.push(t);
    }
    return parts.join(separator);
}

// Keep final_prompt in sync: read-only live preview when AUTO, editable when MANUAL.
function updatePreview(node) {
    const manual = !!getWidget(node, "use_manual_edit")?.value;
    const finalW = getWidget(node, "final_prompt");
    if (!finalW) return;

    if (finalW.inputEl) {
        finalW.inputEl.readOnly = !manual;
        finalW.inputEl.style.opacity = manual ? "1" : "0.6";
        finalW.inputEl.title = manual
            ? "MANUAL: this text is encoded"
            : "AUTO preview of combined slots - turn use_manual_edit ON to edit";
    }

    if (!manual) {
        const assembled = assemblePrompt(node);
        finalW.value = assembled;
        if (finalW.inputEl && finalW.inputEl.value !== assembled) {
            finalW.inputEl.value = assembled;
        }
    }
}

// Swap the text + enable VALUES of two slots (the whole slot moves).
function swapSlotValues(node, a, b) {
    for (const base of ["text", "enable"]) {
        const wa = getWidget(node, `${base}_${a}`);
        const wb = getWidget(node, `${base}_${b}`);
        if (!wa || !wb) continue;
        const tmp = wa.value;
        wa.value = wb.value;
        wb.value = tmp;
        if (wa.inputEl) wa.inputEl.value = wa.value;
        if (wb.inputEl) wb.inputEl.value = wb.value;
    }
}

function moveSlot(node, i, dir) {
    const count = Math.max(1, Math.min(getWidget(node, "slot_count")?.value ?? 1, MAX_SLOTS));
    const j = i + dir;
    if (i < 1 || j < 1 || i > count || j > count) return;
    swapSlotValues(node, i, j);
    updatePreview(node);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "RB.QwenEditPlusPromptSlots.ui",

    nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;

        const slotCountWidget = getWidget(node, "slot_count");
        if (!slotCountWidget) return;

        // --- Per-slot move buttons (value-swap reorder) -----------------------
        // serialize:false keeps them out of widgets_values; interleaving them
        // does not change the relative order of the serializable widgets.
        node._rbUp = [];
        node._rbDown = [];
        for (let i = 1; i <= MAX_SLOTS; i++) {
            const up = node.addWidget("button", "▲ up", null, () => moveSlot(node, i, -1), { serialize: false });
            const down = node.addWidget("button", "▼ down", null, () => moveSlot(node, i, 1), { serialize: false });
            up.serialize = false;
            down.serialize = false;
            node._rbUp[i] = up;
            node._rbDown[i] = down;
        }

        // Interleave: head widgets first, then [text_i, enable_i, up_i, down_i].
        const btnSet = new Set();
        for (let i = 1; i <= MAX_SLOTS; i++) {
            btnSet.add(node._rbUp[i]);
            btnSet.add(node._rbDown[i]);
        }
        const isSlotName = (nm) => /^(text|enable)_\d+$/.test(nm || "");
        const head = node.widgets.filter((w) => !btnSet.has(w) && !isSlotName(w.name));
        const ordered = [...head];
        for (let i = 1; i <= MAX_SLOTS; i++) {
            const t = getWidget(node, `text_${i}`);
            const e = getWidget(node, `enable_${i}`);
            if (t) ordered.push(t);
            if (e) ordered.push(e);
            ordered.push(node._rbUp[i], node._rbDown[i]);
        }
        node.widgets = ordered;

        function updateSlotVisibility() {
            const count = Math.max(1, Math.min(slotCountWidget.value, MAX_SLOTS));
            for (let i = 1; i <= MAX_SLOTS; i++) {
                const vis = i <= count;
                toggleWidget(node, getWidget(node, `text_${i}`), vis);
                toggleWidget(node, getWidget(node, `enable_${i}`), vis);
                toggleWidget(node, node._rbUp[i], vis && i > 1);
                toggleWidget(node, node._rbDown[i], vis && i < count);
            }

            const selectW = getWidget(node, "select");
            if (selectW && selectW.options) {
                selectW.options.max = count;
                if (selectW.value > count) selectW.value = count;
            }

            updatePreview(node);

            const computed = node.computeSize();
            node.setSize([Math.max(computed[0], node.size[0]), computed[1]]);
            node.setDirtyCanvas(true, true);
        }

        // Re-run visibility when slot_count changes.
        const origSlotCb = slotCountWidget.callback;
        slotCountWidget.callback = function (value) {
            if (origSlotCb) origSlotCb.call(this, value);
            updateSlotVisibility();
        };

        // Any widget that affects the assembled prompt refreshes the preview.
        const previewTriggers = ["mode", "select", "separator", "use_manual_edit"];
        for (let i = 1; i <= MAX_SLOTS; i++) {
            previewTriggers.push(`text_${i}`, `enable_${i}`);
        }
        for (const name of previewTriggers) {
            const w = getWidget(node, name);
            if (!w) continue;
            const orig = w.callback;
            w.callback = function (value) {
                if (orig) orig.call(this, value);
                updatePreview(node);
            };
            if (w.inputEl) {
                w.inputEl.addEventListener("input", () => updatePreview(node));
            }
        }

        // --- Robust persistence -------------------------------------------------
        // Dynamic widgets (slot_count hide/show + the move buttons) can shuffle or
        // clear ComfyUI's positional `widgets_values` on reload. To survive that,
        // also store every slot value BY NAME and restore by name after the default
        // positional restore (the approach VideoHelperSuite uses).
        const PERSIST_NAMES = ["mode", "select", "separator", "slot_count", "use_manual_edit", "final_prompt"];
        for (let i = 1; i <= MAX_SLOTS; i++) PERSIST_NAMES.push(`text_${i}`, `enable_${i}`);

        const origOnSerialize = node.onSerialize;
        node.onSerialize = function (o) {
            origOnSerialize?.apply(this, arguments);
            const state = {};
            for (const name of PERSIST_NAMES) {
                const w = getWidget(this, name);
                if (w) state[name] = w.value;
            }
            o.rb_slot_state = state;
        };

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (o) {
            origOnConfigure?.apply(this, arguments);
            const state = o && o.rb_slot_state;
            if (state) {
                for (const name of PERSIST_NAMES) {
                    if (!(name in state)) continue;
                    const w = getWidget(this, name);
                    if (!w) continue;
                    w.value = state[name];
                    if (w.inputEl) w.inputEl.value = state[name];
                }
            }
            requestAnimationFrame(() => updateSlotVisibility());
        };

        // Initial layout once widgets are realized.
        requestAnimationFrame(updateSlotVisibility);
    },
});
