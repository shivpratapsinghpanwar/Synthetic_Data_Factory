"""Generator backends. Heavy ML imports live inside functions, never at module
top level - the control machine has no torch and must still import this package."""

from .prompts import prompt_for  # noqa: F401
