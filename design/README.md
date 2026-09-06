# Design documents

`SDF_Design_Document.docx` — the full system design (MDX, CF-Edit, generator
committee, evaluation protocol, publication plan) with diagrams. Studies are
referenced by codename only (cond_a, cond_b, ...); see the confidentiality
policy in the repo root.

Rebuild after edits:

```
uv run --with python-docx --with matplotlib python design/build_design_doc.py
```

The script regenerates `figures/*.png` and the docx from scratch.
