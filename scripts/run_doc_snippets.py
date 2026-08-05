"""Execute the runnable documentation snippets.

Extracts every ```` ```{.python .run} ```` fence from the given Markdown
files or directories and executes each page's snippets in order in one shared
namespace, mirroring the browser session semantics of
docs-theme/js/examples.js. Runs on stdlib + the documented packages only, so
the same command works natively and inside a Pyodide virtual environment.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_FENCE = re.compile(r"^```\{\.python \.run\}\n(.*?)^```$", re.DOTALL | re.MULTILINE)


def _pages(arguments: list[str]) -> list[Path]:
    """Resolve the arguments to a sorted list of Markdown files."""
    pages: set[Path] = set()
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            pages.update(path.rglob("*.md"))
        else:
            pages.add(path)
    return sorted(pages)


def main(arguments: list[str]) -> int:
    """Run every runnable snippet and report one line per page."""
    total = 0
    for page in _pages(arguments or ["docs"]):
        snippets = _FENCE.findall(page.read_text(encoding="utf-8"))
        if not snippets:
            continue
        namespace: dict[str, object] = {"__name__": f"docs_snippets_{page.stem}"}
        for index, source in enumerate(snippets):
            code = compile(source, f"{page}#snippet{index + 1}", "exec")
            exec(code, namespace)  # noqa: S102 — running the docs is the point
        total += len(snippets)
        print(f"{page}: {len(snippets)} snippets ok")
    if not total:
        message = "no runnable snippets found"
        raise SystemExit(message)
    print(f"{total} snippets ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
