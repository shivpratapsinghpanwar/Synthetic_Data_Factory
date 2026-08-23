"""Per-class prompts for text-conditioned generation (HAM10000 vocabulary).

One fixed prompt per class: each LoRA is class-specific, so the prompt's job is
to anchor the pretrained prior in the right visual neighbourhood, not to carry
fine-grained conditioning.
"""

from __future__ import annotations

# dx code -> (long name, prompt)
CLASS_PROMPTS: dict[str, tuple[str, str]] = {
    "akiec": (
        "actinic keratosis / intraepithelial carcinoma",
        "a dermatoscopy photograph of actinic keratosis on human skin, "
        "clinical dermoscopy image",
    ),
    "bcc": (
        "basal cell carcinoma",
        "a dermatoscopy photograph of basal cell carcinoma on human skin, "
        "clinical dermoscopy image",
    ),
    "bkl": (
        "benign keratosis-like lesion",
        "a dermatoscopy photograph of a benign keratosis lesion on human skin, "
        "clinical dermoscopy image",
    ),
    "df": (
        "dermatofibroma",
        "a dermatoscopy photograph of a dermatofibroma on human skin, "
        "clinical dermoscopy image",
    ),
    "mel": (
        "melanoma",
        "a dermatoscopy photograph of a melanoma on human skin, "
        "clinical dermoscopy image",
    ),
    "nv": (
        "melanocytic nevus",
        "a dermatoscopy photograph of a melanocytic nevus on human skin, "
        "clinical dermoscopy image",
    ),
    "vasc": (
        "vascular lesion",
        "a dermatoscopy photograph of a vascular skin lesion on human skin, "
        "clinical dermoscopy image",
    ),
}


class PromptError(KeyError):
    pass


def prompt_for(cls: str) -> str:
    try:
        return CLASS_PROMPTS[cls][1]
    except KeyError:
        raise PromptError(
            f"no prompt defined for class {cls!r} (SD backends need one; "
            f"known: {sorted(CLASS_PROMPTS)}). The ddpm backend needs no prompt."
        ) from None
