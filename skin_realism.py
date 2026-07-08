"""
Skin Realism (De-Plastic) - post-processing for AI-rendered people.

Qwen-Image-Edit (and diffusion models generally) often render skin that reads
'plastic': poreless, waxy-smooth, with unnaturally clean highlight gradients.
With no original photo to borrow texture from, this node SYNTHESIZES realism:

  - pores:    band-passed noise at a controllable spatial scale, strongest in
              midtones and automatically attenuated where the image already
              has texture (so it never double-textures).
  - grain:    fine luminance grain that unifies the treated region.
  - sheen:    soft-knee compression of highlights inside the skin region to
              kill the waxy specular look.
  - mottling: very low-frequency chroma variation (subsurface redness
              patches) that lifts the airbrushed, single-tone look.

Targeting: by default the node detects skin-colored regions (YCrCb range,
morphologically cleaned, feathered). Wire a MASK into 'mask' to override with
a precise person/face mask. The treated mask is also emitted as an output so
you can inspect exactly what was touched.

Implementation is pure torch + numpy (no cv2/scipy) so the pack needs no
extra dependencies. All randomness is seeded -> identical re-runs.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _tensor_to_rgb_uint8(image_tensor):
    return [
        (image_tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        for i in range(image_tensor.shape[0])
    ]


def _rgb_uint8_list_to_tensor(arrays):
    stacked = np.stack(arrays, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


def _gauss_blur(arr, sigma):
    """Gaussian blur of a 2D float32 numpy array via separable torch conv
    with reflect padding. Returns float32 numpy."""
    if sigma <= 0:
        return arr.astype(np.float32, copy=True)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    kt = torch.from_numpy(k)
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.pad(t, (radius, radius, 0, 0), mode="reflect")
    t = F.conv2d(t, kt.view(1, 1, 1, -1))
    t = F.pad(t, (0, 0, radius, radius), mode="reflect")
    t = F.conv2d(t, kt.view(1, 1, -1, 1))
    return t[0, 0].numpy()


def _dilate(mask01, k):
    """Binary/float dilation via max-pool. mask01: 2D float32 in [0,1]."""
    if k <= 1:
        return mask01
    pad = k // 2
    t = torch.from_numpy(mask01.astype(np.float32))[None, None]
    t = F.max_pool2d(F.pad(t, (pad, pad, pad, pad), mode="replicate"), k, stride=1)
    return t[0, 0].numpy()


def _erode(mask01, k):
    return 1.0 - _dilate(1.0 - mask01, k)


def _rgb_to_ycrcb(rgb_f32):
    """BT.601 full-range. rgb float32 0-255 -> (Y, Cr, Cb) float32."""
    r, g, b = rgb_f32[..., 0], rgb_f32[..., 1], rgb_f32[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return y, cr, cb


def _ycrcb_to_rgb(y, cr, cb):
    r = y + 1.403 * (cr - 128.0)
    g = y - 0.714 * (cr - 128.0) - 0.344 * (cb - 128.0)
    b = y + 1.773 * (cb - 128.0)
    return np.stack([r, g, b], axis=-1)


class SkinRealism:
    CATEGORY = "image/postprocessing"

    @classmethod
    def INPUT_TYPES(cls):
        s = lambda d, t="": ("FLOAT", {"default": d, "min": 0.0, "max": 1.0,
                                       "step": 0.05, "display": "slider", "tooltip": t})
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": s(0.5, "Master intensity. Scales every effect below. 0 = pass-through."),
                "texture_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.25,
                    "tooltip": "Pore/mottle size multiplier. ~1.0 suits a face occupying most of a ~1024px frame; increase for close-ups, decrease for small faces.",
                }),
                "pores": s(0.5, "Micro-texture (band-passed, midtone-weighted). Auto-attenuated where the image already has texture."),
                "grain": s(0.25, "Fine luminance grain over the whole treated region."),
                "sheen_reduction": s(0.3, "Soft compression of skin highlights - kills the waxy specular look."),
                "mottling": s(0.2, "Low-frequency color variation (subsurface redness patches)."),
                "detect_skin": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON: auto-detect skin-colored regions (approximate; color-based). OFF: treat the whole frame. Ignored when a mask is wired in.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True,
                                 "tooltip": "Texture pattern seed. Same seed = identical texture."}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "treated_mask")
    FUNCTION = "process"
    DESCRIPTION = (
        "De-plastic AI-rendered skin: synthesized pores, grain, waxy-highlight "
        "compression, and subtle color mottling, applied only to smooth "
        "skin-colored regions (or a wired-in mask). Emits the treated mask."
    )

    # ------------------------------------------------------------------ mask
    @staticmethod
    def _skin_mask(rgb_u8):
        """Approximate skin mask from YCrCb ranges, cleaned + feathered.
        Color-based, so it's intentionally forgiving; wire a real person/face
        MASK into the node for precision work."""
        y, cr, cb = _rgb_to_ycrcb(rgb_u8.astype(np.float32))
        m = ((cr >= 135.0) & (cr <= 180.0) &
             (cb >= 85.0) & (cb <= 135.0) &
             (y >= 40.0)).astype(np.float32)
        m = _erode(_dilate(m, 5), 5)      # close small holes
        m = _dilate(_erode(m, 5), 5)      # drop small specks
        return m

    # --------------------------------------------------------------- process
    def process(self, image, strength, texture_scale, pores, grain,
                sheen_reduction, mottling, detect_skin, seed, mask=None):
        if float(strength) <= 0.0:
            frames = _tensor_to_rgb_uint8(image)
            empty = torch.zeros((len(frames),) + frames[0].shape[:2], dtype=torch.float32)
            return (image, empty)

        frames = _tensor_to_rgb_uint8(image)
        ext_masks = None
        if mask is not None:
            mk = mask.cpu().numpy()
            if mk.ndim == 2:
                mk = mk[None]
            ext_masks = [mk[min(i, mk.shape[0] - 1)] for i in range(len(frames))]

        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        ts = float(texture_scale)
        out_frames, out_masks = [], []
        for idx, rgb in enumerate(frames):
            h, w = rgb.shape[:2]
            # --- treated-region mask -------------------------------------
            if ext_masks is not None:
                m = ext_masks[idx].astype(np.float32)
                if m.shape != (h, w):
                    m = F.interpolate(torch.from_numpy(m)[None, None], size=(h, w),
                                      mode="bilinear", align_corners=False)[0, 0].numpy()
                m = np.clip(m, 0.0, 1.0)
            elif detect_skin:
                m = self._skin_mask(rgb)
            else:
                m = np.ones((h, w), dtype=np.float32)
            m = _gauss_blur(m, 4.0)
            cov = float(m.mean())
            if cov < 0.001:
                print("[SkinRealism] no treatable region found (mask empty) - "
                      "frame passed through. Wire a MASK or disable detect_skin.")
                out_frames.append(rgb)
                out_masks.append(m)
                continue

            y, cr, cb = _rgb_to_ycrcb(rgb.astype(np.float32))

            # Smoothness weighting: only add pores where high-frequency energy
            # is low (the plastic areas). Textured areas keep their texture.
            hf = np.abs(y - _gauss_blur(y, 1.5))
            hf_local = _gauss_blur(hf, 4.0)
            w_smooth = np.clip((3.5 - hf_local) / 3.5, 0.0, 1.0)

            # Midtone weighting: pores read in midtones, not in deep shadow
            # or blown highlights.
            yn = np.clip(y / 255.0, 0.0, 1.0)
            w_mid = np.clip(4.0 * yn * (1.0 - yn), 0.0, 1.0)

            st = float(strength)

            # --- pores: band-passed noise --------------------------------
            if pores > 0.0:
                n = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
                band = _gauss_blur(n, 0.8 * ts) - _gauss_blur(n, 2.2 * ts)
                band /= max(1e-6, float(band.std()))
                y = y + band * (5.0 * float(pores) * st) * w_mid * w_smooth * m

            # --- grain: fine unifying noise -------------------------------
            if grain > 0.0:
                g = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
                g = _gauss_blur(g, 0.6)
                g /= max(1e-6, float(g.std()))
                y = y + g * (2.5 * float(grain) * st) * m

            # --- sheen: soft-knee highlight compression -------------------
            if sheen_reduction > 0.0:
                sel = m > 0.2
                if int(sel.sum()) > 256:
                    knee = float(np.percentile(y[sel], 82.0))
                    excess = np.maximum(0.0, y - knee)
                    y = y - excess * (0.65 * float(sheen_reduction) * st) * m

            # --- mottling: low-frequency chroma variation -----------------
            if mottling > 0.0:
                amp = 3.5 * float(mottling) * st
                for chan in (cr, cb):
                    f = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
                    f = _gauss_blur(f, 18.0 * ts)
                    f /= max(1e-6, float(f.std()))
                    chan += f * amp * m

            res = _ycrcb_to_rgb(y, cr, cb)
            out_frames.append(np.clip(np.rint(res), 0, 255).astype(np.uint8))
            out_masks.append(np.clip(m, 0.0, 1.0))
            print(f"[SkinRealism] frame {idx}: treated {cov*100:.1f}% of frame "
                  f"(strength {st:.2f}, scale {ts:.2f}).")

        images = _rgb_uint8_list_to_tensor(out_frames)
        masks = torch.from_numpy(np.stack(out_masks, axis=0).astype(np.float32))
        return (images, masks)


NODE_CLASS_MAPPINGS = {"RB_SkinRealism": SkinRealism}
NODE_DISPLAY_NAME_MAPPINGS = {"RB_SkinRealism": "Skin Realism (De-Plastic)"}
