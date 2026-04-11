import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from translate_service import (  # noqa: E402
    ANNOTATION_SYSTEM_PROMPT,
    DIRECT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT,
    DEFAULT_HTML_TRANSLATION_SYSTEM_PROMPT,
    DEFAULT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT,
    DEFAULT_PLAIN_TEXT_TRANSLATION_SYSTEM_PROMPT,
    TranslationService,
    init_translation_service,
)
from annotation import AnnotationService  # noqa: E402


class TranslationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            max_chunk_chars=80,
        )

    def test_repair_translation_latex_replaces_modified_fragments(self) -> None:
        source = "Use $E=mc^2$ and $$\\int_0^1 x^2 dx$$ in the note."
        translation = "在说明中使用 $E=mc^{2}$ 和 $$\\int_0^1 x^2\\,dx$$。"

        repaired, success = self.service._repair_translation_latex(source, translation)

        self.assertTrue(success)
        self.assertEqual(repaired, "在说明中使用 $E=mc^2$ 和 $$\\int_0^1 x^2 dx$$。")

    def test_repair_translation_latex_accepts_unchanged_latex(self) -> None:
        source = "Use $E=mc^2$ in the note."
        translation = "在说明中使用 $E=mc^2$。"

        repaired, success = self.service._repair_translation_latex(source, translation)

        self.assertTrue(success)
        self.assertEqual(repaired, translation)

    def test_repair_translation_latex_accepts_equivalent_delimiters(self) -> None:
        source = r"Use \(E=mc^2\) in the note."
        translation = "在说明中使用 $E=mc^2$。"

        repaired, success = self.service._repair_translation_latex(source, translation)

        self.assertTrue(success)
        self.assertEqual(repaired, r"在说明中使用 \(E=mc^2\)。")

    def test_repair_translation_latex_replaces_spacing_only_difference_with_source_formula(self) -> None:
        source = "Use $F=ma$ in the note."
        translation = "在说明中使用 $F = m a$。"

        repaired, success = self.service._repair_translation_latex(source, translation)

        self.assertTrue(success)
        self.assertEqual(repaired, translation)

    def test_repair_translation_latex_marks_unfixable_formula_and_returns_failure(self) -> None:
        source = "Use $E=mc^2$ in the note."
        translation = "在说明中使用 $E=mc^2$ 和 $a+b$。"

        repaired, success = self.service._repair_translation_latex(source, translation)

        self.assertFalse(success)
        self.assertEqual(repaired, "在说明中使用 $E=mc^2$ 和 $a+b\\textcircled{?}$。")

    def test_repair_translation_latex_accepts_same_formula_set(self) -> None:
        source = "Use $a$ and $b$ in the note."
        translation = "在说明中先使用 $b$，再使用 $a$。"

        repaired, success = self.service._repair_translation_latex(source, translation)

        self.assertTrue(success)
        self.assertEqual(repaired, translation)

    def test_repair_translation_latex_matches_remaining_formulas_by_longest_prefix(self) -> None:
        source = "Use $E=mc^2$ and $F=ma$ in the note."
        translation = "在说明中使用 $E=mc^{2}$ 和 $F = ma$。"

        repaired, success = self.service._repair_translation_latex(source, translation)

        self.assertTrue(success)
        self.assertEqual(repaired, "在说明中使用 $E=mc^2$ 和 $F = ma$。")

    def test_repair_translation_latex_skips_edit_distance_for_impossible_length_candidates(self) -> None:
        source = "Use $a$ and $abcdefghijklmnopqrst$ and $mnopqrstu$ in the note."
        translation = "在说明中使用 $mnopqrstuv$。"

        with patch("translate_latex.edit_distance", return_value=1) as mocked_distance:
            self.service._repair_translation_latex(source, translation)

        self.assertEqual(mocked_distance.call_count, 1)

    def test_looks_suspicious_allows_identical_latex(self) -> None:
        source = "Use $E=mc^2$ and $$\\int_0^1 x^2 dx$$ in the note."
        translation = "在说明中使用 $E=mc^2$ 和 $$\\int_0^1 x^2 dx$$。"

        self.assertFalse(self.service._looks_suspicious_translation(source, translation))

    def test_looks_suspicious_rejects_low_diversity_repetition(self) -> None:
        source = "This is a sufficiently long source text for the heuristic to inspect repeated output." * 2
        translation = "word " * 20

        self.assertTrue(self.service._looks_suspicious_translation(source, translation))

    def test_looks_suspicious_allows_untranslated_bibliography_entry(self) -> None:
        source = (
            "[145] WEIL, A.: L'integration dans les groupes topologiques et ses applications. "
            "Paris: Hermann 1940. Reprinted in Oeuvres Scientifiques, vol. 2."
        )
        translation = source

        self.assertFalse(self.service._looks_suspicious_translation(source, translation))

    def test_looks_suspicious_allows_looser_reference_style(self) -> None:
        source = (
            "12. John Smith and Peter Brown. Compact groups and approximation. Journal of Algebra "
            "45 (2008), pp. 101-120. doi:10.1000/example"
        )
        translation = source

        self.assertFalse(self.service._looks_suspicious_translation(source, translation))

    def test_looks_suspicious_still_flags_untranslated_regular_paragraph(self) -> None:
        source = (
            "This paragraph explains why the construction works in the compact case and how "
            "the argument extends to locally compact groups after a standard approximation step."
        )
        translation = source

        self.assertTrue(self.service._looks_suspicious_translation(source, translation))

    def test_strip_noise_for_similarity_removes_code_latex_and_html(self) -> None:
        text = (
            "Alpha <b>beta</b>.\n"
            "```python\nprint('hello')\n```\n"
            "Keep $E=mc^2$ and \\[x+y\\]."
        )

        cleaned = self.service._strip_noise_for_similarity(text)

        self.assertEqual(cleaned, "Alpha beta. Keep and .")

    def test_need_translate_ignores_latex_markdown_images_and_html_tags(self) -> None:
        text = "![diagram](figure.png)<div>$E=mc^2$</div> \\[x+y\\]"

        self.assertFalse(self.service._need_translate(text))

    def test_need_translate_still_translates_meaningful_text_after_stripping_noise(self) -> None:
        text = "![diagram](figure.png)<p>结论</p> $E=mc^2$"

        self.assertTrue(self.service._need_translate(text))

    def test_markdown_prompt_explicitly_forbids_modifying_latex(self) -> None:
        self.assertIn("[FORMULA_1]", DEFAULT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT)
        self.assertIn("Do NOT modify any LaTeX formulas", DEFAULT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT)

    def test_html_prompt_explicitly_requires_preserving_html(self) -> None:
        self.assertIn("Keep all HTML tags", DEFAULT_HTML_TRANSLATION_SYSTEM_PROMPT)
        self.assertIn("Output only the translated HTML", DEFAULT_HTML_TRANSLATION_SYSTEM_PROMPT)

    def test_build_messages_uses_markdown_prompt_by_default(self) -> None:
        messages = self.service._build_messages("Hello")

        self.assertEqual(messages[0]["content"], DEFAULT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT)

    def test_build_messages_wraps_short_markdown_input_with_translation_instruction(self) -> None:
        messages = self.service._build_messages("or")

        self.assertIn("Translate the following Markdown into Simplified Chinese.", messages[-1]["content"])
        self.assertIn("Return only the translated Markdown.", messages[-1]["content"])
        self.assertIn("<BEGIN_SOURCE>\nor\n<END_SOURCE>", messages[-1]["content"])

    def test_build_messages_keeps_long_markdown_input_raw(self) -> None:
        text = "This paragraph is long enough to avoid the short-input wrapper."

        messages = self.service._build_messages(text)

        self.assertEqual(messages[-1]["content"], text)

    def test_build_messages_uses_direct_markdown_prompt_in_direct_mode(self) -> None:
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            latex_formula_handling="direct",
        )

        messages = service._build_messages("Hello")

        self.assertEqual(messages[0]["content"], DIRECT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT)

    def test_build_messages_uses_html_prompt_in_html_mode(self) -> None:
        messages = self.service._build_messages("<p>Hello</p>", mode="html")

        self.assertEqual(messages[0]["content"], DEFAULT_HTML_TRANSLATION_SYSTEM_PROMPT)

    def test_build_messages_uses_plain_text_prompt_in_plain_text_mode(self) -> None:
        messages = self.service._build_messages("Hello", mode="plain_text")

        self.assertEqual(messages[0]["content"], DEFAULT_PLAIN_TEXT_TRANSLATION_SYSTEM_PROMPT)
        self.assertIn("Translate the following text into Simplified Chinese.", messages[-1]["content"])

    def test_translate_with_retry_preserves_short_input_wrapper_when_replacing_user_message(self) -> None:
        source = "or"
        messages = self.service._build_messages(source)

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="或",
        ) as mocked_create:
            result = self.service._translate_with_retry(source, messages)

        sent_messages = mocked_create.call_args.kwargs["messages"]
        self.assertIn("<BEGIN_SOURCE>\nor\n<END_SOURCE>", sent_messages[-1]["content"])
        self.assertEqual(result, "或")

    def test_annotation_prompt_emphasizes_intuition_and_brevity(self) -> None:
        self.assertIn("Focus on mathematical intuition", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("Do NOT restate the formal definition in detail", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("Keep it short and sharp", ANNOTATION_SYSTEM_PROMPT)

    def test_annotation_service_recognizes_numbered_and_bold_headings(self) -> None:
        service = AnnotationService(client=None, model="fake", enabled=True)

        self.assertTrue(service._should_annotate(None, "**Theorem 1.2.** Every compact group ..."))
        self.assertTrue(service._should_annotate(None, "Lemma 3.4 Let G be a group."))
        self.assertTrue(service._should_annotate(None, "3.2 proposition Let G be compact."))
        self.assertTrue(service._should_annotate(None, "**3.2 proposition** Let G be compact."))
        self.assertTrue(service._should_annotate(None, "> Corollary 2.1: The map is injective."))
        self.assertTrue(service._should_annotate(None, "Definition 5. A metric space is ..."))
        self.assertFalse(service._should_annotate(None, "Remark 1.2 This is only a comment."))

    def test_annotation_service_recognizes_continuation_blocks(self) -> None:
        service = AnnotationService(client=None, model="fake", enabled=True)

        self.assertTrue(service._should_annotate("Theorem 1.2", "  (i) first claim"))
        self.assertTrue(service._should_annotate("Theorem 1.2", "- second claim"))
        self.assertTrue(service._should_annotate("Theorem 1.2", "\\[\nX = Y\n\\]\n"))
        self.assertTrue(service._should_annotate("Theorem 1.2", "$$\nX = Y\n$$\n"))
        self.assertFalse(service._should_annotate("Theorem 1.2", "Proof. This is not part of the statement."))

    def test_protect_latex_replaces_formulas_with_placeholders(self) -> None:
        source = "The formula is $E=mc^2$ and $$x+y$$."

        protected, formula_map = self.service._protect_latex(source)

        self.assertEqual(protected, "The formula is [FORMULA_1] and [FORMULA_2].")
        self.assertEqual(
            formula_map,
            [("[FORMULA_1]", "$E=mc^2$"), ("[FORMULA_2]", "$$x+y$$")],
        )

    def test_restore_latex_restores_placeholders_back_to_source_formulas(self) -> None:
        translation = "公式是 [FORMULA_1]，并且 [FORMULA_2] 保持不变。"
        formula_map = [("[FORMULA_1]", "$E=mc^2$"), ("[FORMULA_2]", "$$x+y$$")]

        restored, success = self.service._restore_latex(translation, formula_map)

        self.assertTrue(success)
        self.assertEqual(restored, "公式是 $E=mc^2$，并且 $$x+y$$ 保持不变。")

    def test_translate_with_retry_uses_source_formula_representation_when_repairing(self) -> None:
        source = r"Use \(E=mc^2\) in the note."

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="在说明中使用 [FORMULA_1]。",
        ) as mocked_create:
            result = self.service._translate_with_retry(
                source,
                [{"role": "user", "content": source}],
            )

        self.assertEqual(result, r"在说明中使用 \(E=mc^2\)。")
        self.assertEqual(mocked_create.call_count, 1)

    def test_translate_with_retry_repairs_latex_without_retry(self) -> None:
        source = "Use $E=mc^2$ in the note."

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="在说明中使用 [FORMULA_1]。",
        ) as mocked_create:
            result = self.service._translate_with_retry(
                source,
                [{"role": "user", "content": source}],
            )

        self.assertEqual(result, "在说明中使用 $E=mc^2$。")
        self.assertEqual(mocked_create.call_count, 1)

    def test_translate_with_retry_direct_mode_sends_formula_text_without_placeholders(self) -> None:
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            latex_formula_handling="direct",
        )
        source = "Use $E=mc^2$ in the note."

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="在说明中使用 $E=mc^2$。",
        ) as mocked_create:
            result = service._translate_with_retry(
                source,
                service._build_messages(source),
            )

        sent_messages = mocked_create.call_args.kwargs["messages"]
        self.assertEqual(sent_messages[-1]["content"], source)
        self.assertNotIn("[FORMULA_1]", sent_messages[-1]["content"])
        self.assertEqual(result, "在说明中使用 $E=mc^2$。")

    def test_translate_with_retry_requests_reasoning_none_for_openai_compatible(self) -> None:
        source = "First paragraph."

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="第一段。",
        ) as mocked_create:
            self.service._translate_with_retry(
                source,
                [{"role": "user", "content": source}],
            )

        self.assertEqual(mocked_create.call_args.kwargs["reasoning_effort"], "none")

    def test_translate_with_retry_retries_when_latex_cannot_be_repaired(self) -> None:
        source = "Use $E=mc^2$ in the note."
        responses = iter(
            [
                "在说明中使用 [FORMULA_1] 和 $a+b$。",
                "在说明中使用 [FORMULA_1]。",
            ]
        )

        with patch(
            "translate_service.create_chat_completion_with_retry",
            side_effect=lambda **_: next(responses),
        ) as mocked_create:
            result = self.service._translate_with_retry(
                source,
                [{"role": "user", "content": source}],
            )

        self.assertEqual(result, "在说明中使用 $E=mc^2$。")
        self.assertEqual(mocked_create.call_count, 2)

    def test_translate_with_retry_returns_last_failed_latex_result_after_max_retries(self) -> None:
        source = "Use $E=mc^2$ in the note."

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="在说明中使用 [FORMULA_1] 和 $a+b$。",
        ) as mocked_create:
            result = self.service._translate_with_retry(
                source,
                [{"role": "user", "content": source}],
            )

        self.assertEqual(result, "在说明中使用 $E=mc^2$ 和 $a+b$。")
        self.assertEqual(mocked_create.call_count, 3)

    def test_iter_translation_blocks_keeps_display_math_block_intact(self) -> None:
        text = (
            "Paragraph one is intentionally long to trigger chunking.\n\n"
            "\\[\nX = \\{ x \\mid f(x) = 0 \\}.\n\\]\n\n"
            "Paragraph two is also long enough to require another chunk."
        )

        blocks = list(self.service._iter_translation_blocks(text))
        protected_blocks = [block.text for block in blocks if block.protected]

        self.assertTrue(protected_blocks)
        self.assertIn("\\[\nX = \\{ x \\mid f(x) = 0 \\}.\n\\]\n", protected_blocks)

    def test_translate_text_block_uses_multiple_requests_for_long_text(self) -> None:
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            max_chunk_chars=40,
        )
        text = (
            "First paragraph has enough words to exceed the chunk limit.\n\n"
            "Second paragraph also has enough words to exceed the same chunk limit."
        )

        with patch.object(
            service,
            "_translate_with_retry",
            side_effect=lambda source_text, messages: f"ZH::{source_text}",
        ) as mocked_translate:
            result = service.translate_text_block(text)

        expected = (
            "ZH::First paragraph has enough words to exceed the chunk limit.\n\n"
            "ZH::Second paragraph also has enough words to exceed the same chunk limit."
        )
        self.assertEqual(result, expected)
        self.assertGreater(mocked_translate.call_count, 1)

    def test_translate_with_retry_restores_trailing_paragraph_breaks(self) -> None:
        source = "First paragraph.\n\n"

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="第一段。",
        ):
            result = self.service._translate_with_retry(
                source,
                [{"role": "user", "content": source}],
            )

        self.assertEqual(result, "第一段。\n\n")

    def test_translate_with_retry_strips_outer_whitespace_before_request(self) -> None:
        source = "\n\nFirst paragraph.\n\n"

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="第一段。",
        ) as mocked_create:
            result = self.service._translate_with_retry(
                source,
                [{"role": "system", "content": "system"}, {"role": "user", "content": source}],
            )

        sent_messages = mocked_create.call_args.kwargs["messages"]
        self.assertEqual(sent_messages[-1]["content"], "First paragraph.")
        self.assertEqual(result, "\n\n第一段。\n\n")

    def test_translate_text_block_preserves_code_and_display_math_without_translation(self) -> None:
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            max_chunk_chars=40,
        )
        text = (
            "First paragraph needs translation.\n\n"
            "```python\nprint('hello')\n```\n\n"
            "\\[\nX = \\{ x \\mid f(x) = 0 \\}.\n\\]\n\n"
            "Second paragraph also needs translation."
        )

        with patch.object(
            service,
            "_translate_with_retry",
            side_effect=lambda source_text, messages: f"ZH::{source_text}",
        ) as mocked_translate:
            result = service.translate_text_block(text)

        self.assertIn("```python\nprint('hello')\n```\n", result)
        self.assertIn("\\[\nX = \\{ x \\mid f(x) = 0 \\}.\n\\]\n", result)
        translated_inputs = [call.args[0] for call in mocked_translate.call_args_list]
        self.assertTrue(translated_inputs)
        self.assertTrue(all("```python" not in chunk for chunk in translated_inputs))
        self.assertTrue(all("\\[\nX = \\{ x \\mid f(x) = 0 \\}.\n\\]\n" not in chunk for chunk in translated_inputs))

    def test_iter_translation_blocks_does_not_force_split_without_newlines(self) -> None:
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            max_chunk_chars=10,
        )

        text = "averylongplainsegmentwithoutanynewline"

        blocks = list(service._iter_translation_blocks(text))

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, text)
        self.assertFalse(blocks[0].protected)
        self.assertFalse(blocks[0].is_annotation)

    def test_iter_translation_blocks_inserts_combined_annotation_block_after_continuations(self) -> None:
        annotation_service = AnnotationService(client=None, model="fake", enabled=True)
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            max_chunk_chars=200,
            _annotation_service=annotation_service,
        )
        text = (
            "Theorem 1.2. Every compact group is complete.\n\n"
            "- The completion does not add new points.\n\n"
            "\\[\nX = \\{ x \\mid f(x) = 0 \\}.\n\\]\n\n"
            "Proof. Omitted.\n"
        )

        blocks = list(service._iter_translation_blocks(text))

        self.assertEqual(len([block for block in blocks if block.is_annotation]), 1)
        annotation_block = next(block for block in blocks if block.is_annotation)
        self.assertEqual(
            annotation_block.text,
            (
                "Theorem 1.2. Every compact group is complete.\n\n"
                "- The completion does not add new points.\n\n"
                "\\[\nX = \\{ x \\mid f(x) = 0 \\}.\n\\]\n"
            ),
        )
        self.assertEqual(blocks[-1].text, "Proof. Omitted.\n")

    def test_translate_text_block_annotates_combined_multiblock_statement_once(self) -> None:
        annotation_service = AnnotationService(client=None, model="fake", enabled=True)
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            max_chunk_chars=200,
            _annotation_service=annotation_service,
        )
        text = (
            "Definition 2.1. A metric space is a set with a distance.\n\n"
            "  (i) The distance is non-negative.\n\n"
            "\\[\n d(x,y) = 0 \\iff x = y\n\\]\n"
        )

        with patch.object(
            service,
            "_translate_with_retry",
            side_effect=lambda source_text, messages: f"ZH::{source_text}",
        ) as mocked_translate, patch.object(
            annotation_service,
            "annotate",
            return_value="\n\n> **直观理解：** 注释\n",
        ) as mocked_annotate:
            result = service.translate_text_block(text)

        self.assertIn("ZH::Definition 2.1. A metric space is a set with a distance.\n\n", result)
        self.assertIn("ZH::  (i) The distance is non-negative.\n\n", result)
        self.assertIn("\\[\n d(x,y) = 0 \\iff x = y\n\\]\n", result)
        self.assertIn("> **直观理解：** 注释", result)
        self.assertEqual(mocked_translate.call_count, 2)
        mocked_annotate.assert_called_once_with(
            (
                "Definition 2.1. A metric space is a set with a distance.\n\n"
                "  (i) The distance is non-negative.\n\n"
                "\\[\n d(x,y) = 0 \\iff x = y\n\\]\n"
            )
        )

    def test_translate_pages_passes_mode_through_to_text_block(self) -> None:
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            max_chunk_chars=40,
        )

        with patch.object(
            service,
            "translate_text_block",
            side_effect=lambda text, mode="markdown": f"{mode}::{text}",
        ) as mocked_translate:
            result = service.translate_pages(["<p>hello</p>"], mode="html")

        self.assertEqual(result, ["html::<p>hello</p>"])
        self.assertEqual(mocked_translate.call_args.kwargs["mode"], "html")

    def test_init_translation_service_uses_selected_provider(self) -> None:
        args = SimpleNamespace(
            translation_provider="openai_compatible",
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="gemma4:26b",
            translation_reasoning_effort="none",
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_latex_formula_handling="placeholder",
            enable_annotation=False,
            annotation_provider=None,
            annotation_base_url=None,
            annotation_api_key=None,
            annotation_model=None,
            annotation_reasoning_effort=None,
        )

        with patch("translate_service.configure_translation_client", return_value="codex-client") as mocked_configure:
            service = init_translation_service(args)

        mocked_configure.assert_called_once_with(
            provider="openai_compatible",
            base_url="http://translate/v1",
            api_key="translate-key",
            model="gemma4:26b",
            reasoning_effort="none",
        )
        self.assertEqual(service.client, "codex-client")
        self.assertEqual(service.reasoning_effort, "none")

    def test_init_translation_service_builds_annotation_service_with_own_backend(self) -> None:
        args = SimpleNamespace(
            translation_provider="openai_compatible",
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="gemma4:26b",
            translation_reasoning_effort="none",
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_latex_formula_handling="placeholder",
            enable_annotation=True,
            annotation_provider="openai_compatible",
            annotation_base_url="http://annotation/v1",
            annotation_api_key="annotation-key",
            annotation_model="gpt-4o-mini",
            annotation_reasoning_effort="medium",
        )

        with patch(
            "translate_service.configure_translation_client",
            side_effect=["annotation-client", "translation-client"],
        ) as mocked_configure:
            service = init_translation_service(args)

        self.assertEqual(mocked_configure.call_count, 2)
        self.assertEqual(service.client, "translation-client")
        self.assertIsNotNone(service._annotation_service)
        self.assertEqual(service._annotation_service.client, "annotation-client")
        self.assertEqual(service._annotation_service.model, "gpt-4o-mini")
        self.assertEqual(service._annotation_service.reasoning_effort, "medium")

    def test_init_translation_service_annotation_defaults_to_translation_backend(self) -> None:
        args = SimpleNamespace(
            translation_provider="openai_compatible",
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="gemma4:26b",
            translation_reasoning_effort="low",
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_latex_formula_handling="placeholder",
            enable_annotation=True,
            annotation_provider=None,
            annotation_base_url=None,
            annotation_api_key=None,
            annotation_model=None,
            annotation_reasoning_effort=None,
        )

        with patch(
            "translate_service.configure_translation_client",
            side_effect=["annotation-client", "translation-client"],
        ) as mocked_configure:
            service = init_translation_service(args)

        self.assertEqual(mocked_configure.call_count, 2)
        self.assertEqual(
            mocked_configure.call_args_list[0].kwargs,
            {
                "provider": "openai_compatible",
                "base_url": "http://translate/v1",
                "api_key": "translate-key",
                "model": "gemma4:26b",
                "reasoning_effort": "low",
            },
        )
        self.assertIsNotNone(service._annotation_service)
        self.assertEqual(service._annotation_service.model, "gemma4:26b")


if __name__ == "__main__":
    unittest.main()
