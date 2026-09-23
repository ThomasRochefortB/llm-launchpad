"""Source-level guards for TUI mistakes that fail silently at runtime.

Both defects these catch rendered without an error or a failing test: they
were found only by looking at screenshots (PR #92).
"""

from __future__ import annotations

import io
import re
import tokenize
import unittest
from pathlib import Path

import llm_launchpad.tui as tui_package

TUI_DIR = Path(tui_package.__file__).parent
STYLESHEET = TUI_DIR / "theme.tcss"

# Theme variables and fixed ANSI colours. In Textual 5 content markup a tag
# naming a theme variable without `$` parses without complaint and renders as
# plain text; a fixed colour renders, but ignores the chosen theme.
_THEME_NAMES = (
    "primary|secondary|accent|success|warning|error|foreground|background|"
    "surface|panel|boost|text|text-muted"
)
_FIXED_COLOURS = "green|yellow|red|blue|cyan|magenta|white|black|orange|purple"
_STYLE_WORDS = r"(?:bold |dim |italic |underline |reverse |strike |not )*"
_BAD_TAG = re.compile(
    r"\[/?" + _STYLE_WORDS + r"(?:" + _THEME_NAMES + "|" + _FIXED_COLOURS + r")"
    r"(?: on [^\]]+)?\]"
)


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Every string (and f-string) literal in ``path`` with its line number."""
    source = path.read_text(encoding="utf-8")
    literals: list[tuple[int, str]] = []
    # f-strings arrive as FSTRING_MIDDLE pieces on Python 3.12+.
    kinds = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in kinds:
            literals.append((token.start[0], token.string))
    return literals


def _python_sources() -> list[Path]:
    return sorted(TUI_DIR.rglob("*.py"))


class MarkupThemeNameTests(unittest.TestCase):
    def test_markup_names_theme_colours_through_variables(self) -> None:
        offenders = [
            f"{path.relative_to(TUI_DIR)}:{line}: {match.group(0)}"
            for path in _python_sources()
            for line, literal in _string_literals(path)
            for match in _BAD_TAG.finditer(literal)
        ]
        self.assertEqual(
            offenders,
            [],
            "Use `[$primary]` (etc.), not `[primary]` or a fixed colour: "
            "Textual 5 prints the former as plain text.",
        )

    def test_the_guard_recognises_what_it_forbids(self) -> None:
        for bad in ("[bold primary]", "[success]", "[/warning]", "[green]", "[dim red]"):
            with self.subTest(tag=bad):
                self.assertIsNotNone(_BAD_TAG.search(bad))
        for good in ("[bold $primary]", "[$success]", "[/$warning]", "[dim]", "[bold]", "[/]"):
            with self.subTest(tag=good):
                self.assertIsNone(_BAD_TAG.search(good))


# Classes Textual itself sets on widgets, which the app never names.
_TEXTUAL_CLASSES = frozenset({
    "-on",          # Switch
    "-invalid",     # Input
    "-primary",     # Button variant="primary"
    "-error",       # Button variant="error"
    "-warning",     # Button variant="warning"
    "-success",     # Button variant="success"
    "--highlight",  # ListItem
})
# Class-name families the app builds with f-strings, by prefix.
_DYNAMIC_PREFIXES = ("viewport-", "density-", "state-")


def _stylesheet_classes() -> set[str]:
    css = re.sub(r"/\*.*?\*/", "", STYLESHEET.read_text(encoding="utf-8"), flags=re.S)
    selectors = "".join(re.findall(r"([^{}]*)\{", css))
    return set(re.findall(r"\.(-{0,2}[A-Za-z][\w-]*)", selectors))


class StylesheetClassTests(unittest.TestCase):
    def test_every_class_the_stylesheet_styles_is_set_somewhere(self) -> None:
        """A rule for a class nothing sets is dead, and fails without a sound.

        `Button.-danger` was meant to style destructive buttons; Textual's
        `variant="error"` sets `-error`, so the rule never matched.
        """
        literals = [
            literal
            for path in _python_sources()
            for _line, literal in _string_literals(path)
        ]
        corpus = "\n".join(literals)
        unused = sorted(
            name
            for name in _stylesheet_classes()
            if "--" not in name.lstrip("-")  # component classes: option-list--option
            and name not in _TEXTUAL_CLASSES
            and not name.startswith(_DYNAMIC_PREFIXES)
            and not re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", corpus)
        )
        self.assertEqual(unused, [], "stylesheet classes no Python code sets")


if __name__ == "__main__":
    unittest.main()
