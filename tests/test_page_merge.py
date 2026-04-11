import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from translate_page_merge import decide_page_boundary_merge, merge_cross_page_paragraphs  # noqa: E402
from translate_service import TranslationService  # noqa: E402


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, model: str, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages})
        user_prompt = messages[-1]["content"]
        return _FakeResponse(f"ZH::{user_prompt}")


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = _FakeChat(self.completions)


class PageMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeClient()

    def test_boundary_merge_joins_when_left_looks_truncated(self) -> None:
        decision = decide_page_boundary_merge(
            client=None,
            model=None,
            left_block="This paragraph continues",
            right_block="on the next page.",
        )

        self.assertEqual(decision, "JOIN")

    def test_boundary_merge_joins_when_right_starts_lowercase(self) -> None:
        decision = decide_page_boundary_merge(
            client=None,
            model=None,
            left_block="The result follows",
            right_block="because the assumption still holds.",
        )

        self.assertEqual(decision, "JOIN")

    def test_boundary_merge_splits_on_heading_or_list(self) -> None:
        self.assertEqual(
            decide_page_boundary_merge(
                client=None,
                model=None,
                left_block="The previous paragraph ends here",
                right_block="# New Section",
            ),
            "SPLIT",
        )
        self.assertEqual(
            decide_page_boundary_merge(
                client=None,
                model=None,
                left_block="The previous paragraph ends here",
                right_block="1. First item",
            ),
            "SPLIT",
        )

    def test_boundary_merge_splits_when_left_last_line_is_heading(self) -> None:
        decision = decide_page_boundary_merge(
            client=None,
            model=None,
            left_block="Some intro\n# Section Title",
            right_block="Continued body text",
        )

        self.assertEqual(decision, "SPLIT")

    def test_boundary_merge_splits_when_left_ends_with_block_latex(self) -> None:
        self.assertEqual(
            decide_page_boundary_merge(
                client=None,
                model=None,
                left_block="$$\na^2+b^2=c^2\n$$",
                right_block="The next paragraph starts here.",
            ),
            "SPLIT",
        )
        self.assertEqual(
            decide_page_boundary_merge(
                client=None,
                model=None,
                left_block="\\[\na+b\n\\]",
                right_block="The next paragraph starts here.",
            ),
            "SPLIT",
        )

    def test_boundary_merge_splits_when_left_looks_like_title(self) -> None:
        decision = decide_page_boundary_merge(
            client=None,
            model=None,
            left_block="Chapter 3",
            right_block="The discussion begins here.",
        )

        self.assertEqual(decision, "SPLIT")

    def test_boundary_merge_splits_when_both_sides_look_complete(self) -> None:
        decision = decide_page_boundary_merge(
            client=None,
            model=None,
            left_block="This paragraph ends here.",
            right_block="Another paragraph starts here.",
        )

        self.assertEqual(decision, "SPLIT")

    def test_merge_cross_page_paragraphs_moves_next_first_block_backward(self) -> None:
        merged_pages, decisions = merge_cross_page_paragraphs(
            pages=[
                "Heading\n\nThis paragraph continues",
                "on the next page.\n\nNew paragraph starts here.",
                "Definitely another section.",
            ],
            client=self.client,
            model="fake-model",
            page_numbers=[10, 11, 12],
        )

        self.assertEqual(
            merged_pages,
            [
                "Heading\n\nThis paragraph continues\non the next page.",
                "New paragraph starts here.",
                "Definitely another section.",
            ],
        )
        self.assertEqual(
            [(item.page_number_left, item.page_number_right, item.decision) for item in decisions],
            [(10, 11, "JOIN"), (11, 12, "SPLIT")],
        )

    def test_translate_pages_translates_each_page_independently(self) -> None:
        service = TranslationService(
            client=self.client,
            model="fake-model",
            temperature=0.2,
        )

        result = service.translate_pages(
            ["Page one.", "Page two."],
            max_workers=1,
        )

        self.assertEqual(result, ["ZH::Page one.", "ZH::Page two."])


if __name__ == "__main__":
    unittest.main()
