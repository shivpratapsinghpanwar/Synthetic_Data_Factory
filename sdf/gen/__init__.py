"""Generator backends. Heavy ML imports live inside functions, never at module
top level - the control machine has no torch and must still import this package."""

from .prompts import prompt_for  # noqa: F401


class BackendError(RuntimeError):
    pass


def get_backend(name: str):
    """Resolve a backend module by name. Each backend exposes train(), sample()
    and ADAPTER_DIR_NAME with identical signatures."""
    from . import ddpm, sd15_lora, sdxl_lora

    backends = {"sd15_lora": sd15_lora, "ddpm": ddpm, "sdxl_lora": sdxl_lora}
    try:
        return backends[name]
    except KeyError:
        raise BackendError(
            f"unknown generator backend {name!r}; available: {sorted(backends)}"
        ) from None
