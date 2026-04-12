## repo environment
- Use the project virtual environment at `./.venv`.
- Run Python and pip via `./.venv/bin/python` and `./.venv/bin/pip` instead of the system interpreter.

## model environment
This project need backend serving LLM. By default, you can think there are two services running already:
- ollama running at: http://localhost:11434/v1, serving multiple models.
- vllm running at: http://localhost:8000/v1, serving model "deepseek-ai/DeepSeek-OCR"

Don't try to start or stop these backend model services.

## repo rules
- Do not create a git commit automatically after completing code or document changes. Only create a commit when the user explicitly asks for one.
- when writing design docs, use Mermaid diagrams when it can clarify things better.
  - When writing Mermaid diagrams, quote node text or rewrite labels if they contain syntax-like tokens such as `[[` or `]]`, so the diagram remains parseable.

## test
- in directory `./tests/data`, there are sample pdf files and sample pdf page images can be used for test.
  - `one_page.pdf` is a one page pdf file.
  - `book.pdf` is a whole book pdf.
  - `one_page.png` is a pdf page in png format.
- during your test, you can generate intermediate files, put them in `./test/data/output`
