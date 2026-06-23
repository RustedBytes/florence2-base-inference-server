# Task Fields

Requests use the same task fields as the Florence-2 Space:

- `task_type`: `Single task` or `Cascased task` (`Cascaded task` is also accepted)
- `task_prompt`: defaults to `Caption`
- `text_input`: optional text for prompts that need it

Single task prompts:

- `Caption`
- `Detailed Caption`
- `More Detailed Caption`
- `Object Detection`
- `Dense Region Caption`
- `Region Proposal`
- `Caption to Phrase Grounding`
- `Referring Expression Segmentation`
- `Region to Segmentation`
- `Open Vocabulary Detection`
- `Region to Category`
- `Region to Description`
- `OCR`
- `OCR with Region`

Cascased task prompts:

- `Caption + Grounding`
- `Detailed Caption + Grounding`
- `More Detailed Caption + Grounding`
