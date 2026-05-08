import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from io import StringIO

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
    GenericChatTranslator,
    TranslationService,
    TranslateGemmaTranslationService,
    TranslateGemmaTranslator,
    init_annotation_service,
    init_translator,
    init_translation_service,
    validate_translation_args,
)
from annotation import AnnotationService  # noqa: E402
from annotation import PAGE_ANNOTATION_SYSTEM_PROMPT  # noqa: E402


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

    def test_get_suspicious_translation_reason_reports_repetition(self) -> None:
        source = "This is a sufficiently long source text for the heuristic to inspect repeated output." * 2
        translation = "word " * 20

        reason = self.service._get_suspicious_translation_reason(source, translation)

        self.assertEqual(reason, "low diversity or repeated content")

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

    def test_translategemma_build_messages_use_tagged_single_user_message(self) -> None:
        service = TranslateGemmaTranslator(
            client=None,
            model="translategemma-12b-it",
            temperature=0.2,
            source_lang="en",
            target_lang="zh",
        )

        messages = service._build_messages("Hello", mode="plain_text")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("<<<source>>>en<<<target>>>zh<<<text>>>", messages[0]["content"])
        self.assertIn("Translate the following text into natural, fluent Simplified Chinese.", messages[0]["content"])

    def test_translategemma_build_messages_wrap_html_with_preservation_instruction(self) -> None:
        service = TranslateGemmaTranslator(
            client=None,
            model="translategemma-12b-it",
            temperature=0.2,
        )

        messages = service._build_messages("<p>Hello</p>", mode="html")

        self.assertIn("Preserve all HTML tags, attributes, URLs, and LaTeX formulas exactly.", messages[0]["content"])
        self.assertTrue(messages[0]["content"].endswith("<p>Hello</p>"))

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
        self.assertIn("what the statement means in plain language", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("why this concept/result is natural or useful", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("Do NOT restate the formal definition or theorem in detail", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("Keep it short and sharp", ANNOTATION_SYSTEM_PROMPT)

    def test_page_annotation_prompt_emphasizes_intuition_motivation_and_missing_details(self) -> None:
        self.assertIn("Treat the page as a coherent whole", PAGE_ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("Strongly emphasize intuition and motivation", PAGE_ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("fill in the missing explanatory bridge", PAGE_ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("Expand on ideas that are under-explained", PAGE_ANNOTATION_SYSTEM_PROMPT)

    def test_page_annotation_normalization_preserves_paragraph_breaks(self) -> None:
        service = AnnotationService(client=None, model="fake", enabled=True, mode="page")

        normalized = service._normalize_output("第一段说明。\n\n第二段补充动机。")

        self.assertEqual(normalized, "第一段说明。\n\n第二段补充动机。\n\n")

    def test_page_annotation_render_quotes_every_paragraph(self) -> None:
        service = AnnotationService(client=None, model="fake", enabled=True, mode="page")

        with patch.object(
            service,
            "_call_annotate",
            return_value="第一段说明。\n\n第二段补充动机。\n\n",
        ):
            rendered = service.annotate("source")

        self.assertEqual(
            rendered,
            "\n\n> **本页导读：** 第一段说明。\n>\n> 第二段补充动机。\n",
        )

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
        self.assertTrue(service._should_annotate("Theorem 1.2", "Moreover, the same conclusion holds for quotients."))
        self.assertTrue(service._should_annotate("Definition 2.1", "Where $d$ denotes the metric function."))
        self.assertTrue(service._should_annotate("Theorem 1.2", "> $$\nX = Y\n$$\n"))
        self.assertTrue(service._should_annotate("Theorem 1.2", "**(i)** first claim"))
        self.assertTrue(service._should_annotate("Theorem 1.2", "\\[\nX = Y\n\\]\n"))
        self.assertTrue(service._should_annotate("Theorem 1.2", "$$\nX = Y\n$$\n"))
        self.assertFalse(service._should_annotate("Theorem 1.2", "Proof. This is not part of the statement."))
        self.assertFalse(service._should_annotate("Theorem 1.2", "Remark 1.3. This is only a side note."))

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

    def test_translate_with_retry_skips_latex_placeholder_protection_for_html_mode(self) -> None:
        source = "<p>Use $E=mc^2$ in the note.</p>"

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="<p>在说明中使用 $E=mc^2$。</p>",
        ) as mocked_create, patch.object(
            self.service,
            "_protect_latex",
            side_effect=AssertionError("html mode should not protect latex"),
        ), patch.object(
            self.service,
            "_repair_translation_latex",
            side_effect=AssertionError("html mode should not repair latex"),
        ):
            result = self.service._translate_with_retry(
                source,
                self.service._build_messages(source, mode="html"),
                mode="html",
            )

        sent_messages = mocked_create.call_args.kwargs["messages"]
        self.assertEqual(sent_messages[-1]["content"], source)
        self.assertEqual(result, "<p>在说明中使用 $E=mc^2$。</p>")

    def test_translate_with_retry_skips_latex_placeholder_protection_for_plain_text_mode(self) -> None:
        source = "Use $E=mc^2$ in the note."

        with patch(
            "translate_service.create_chat_completion_with_retry",
            return_value="在说明中使用 $E=mc^2$。",
        ) as mocked_create, patch.object(
            self.service,
            "_protect_latex",
            side_effect=AssertionError("plain_text mode should not protect latex"),
        ), patch.object(
            self.service,
            "_repair_translation_latex",
            side_effect=AssertionError("plain_text mode should not repair latex"),
        ):
            result = self.service._translate_with_retry(
                source,
                self.service._build_messages(source, mode="plain_text"),
                mode="plain_text",
            )

        sent_messages = mocked_create.call_args.kwargs["messages"]
        self.assertEqual(sent_messages[-1]["content"], source)
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

    def test_translate_with_retry_logs_suspicious_reason(self) -> None:
        source = "This is a sufficiently long source text for the heuristic to inspect repeated output." * 2
        log_stream = StringIO()
        service = TranslationService(client=None, model="fake-model", temperature=0.2)

        with patch(
            "translate_service.create_chat_completion_with_retry",
            side_effect=["word " * 20, "有效译文"],
        ) as mocked_create, patch("sys.stderr", log_stream):
            result = service._translate_with_retry(
                source,
                service._build_messages(source),
            )

        self.assertEqual(result, "有效译文")
        self.assertIn("reason: low diversity or repeated content", log_stream.getvalue())
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
            side_effect=lambda source_text, messages, mode="markdown": f"ZH::{source_text}",
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
            side_effect=lambda source_text, messages, mode="markdown": f"ZH::{source_text}",
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

    def test_iter_translation_blocks_keeps_plain_contextual_continuations_in_annotation_block(self) -> None:
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
            "Where the distance function may take the value +infinity.\n\n"
            "Moreover, this convention is useful in geometric applications.\n\n"
            "Remark 2.2. This is only a comment.\n"
        )

        blocks = list(service._iter_translation_blocks(text))

        self.assertEqual(len([block for block in blocks if block.is_annotation]), 1)
        annotation_block = next(block for block in blocks if block.is_annotation)
        self.assertEqual(
            annotation_block.text,
            (
                "Definition 2.1. A metric space is a set with a distance.\n\n"
                "Where the distance function may take the value +infinity.\n\n"
                "Moreover, this convention is useful in geometric applications.\n\n"
            ),
        )
        self.assertEqual(blocks[-1].text, "Remark 2.2. This is only a comment.\n")

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
            side_effect=lambda source_text, messages, mode="markdown": f"ZH::{source_text}",
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

    def test_translate_text_block_appends_page_annotation_once_per_page(self) -> None:
        annotation_service = AnnotationService(client=None, model="fake", enabled=True, mode="page")
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            max_chunk_chars=200,
            _annotation_service=annotation_service,
        )
        text = (
            "Definition 2.1. A metric space is a set with a distance.\n\n"
            "This page explains why the triangle inequality matters.\n"
        )

        with patch.object(
            service,
            "_translate_with_retry",
            side_effect=lambda source_text, messages, mode="markdown": f"ZH::{source_text}",
        ) as mocked_translate, patch.object(
            annotation_service,
            "annotate",
            return_value="\n\n> **本页导读：** 这一页主要解释距离概念的动机。\n",
        ) as mocked_annotate:
            result = service.translate_text_block(text)

        self.assertEqual(
            result,
            (
                "ZH::Definition 2.1. A metric space is a set with a distance.\n\n"
                "ZH::This page explains why the triangle inequality matters.\n"
                "\n\n> **本页导读：** 这一页主要解释距离概念的动机。\n"
            ),
        )
        self.assertEqual(mocked_translate.call_count, 2)
        mocked_annotate.assert_called_once_with(text)

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

    def test_page_scope_translates_whole_page_in_single_request(self) -> None:
        service = TranslationService(
            client=None,
            model="fake-model",
            temperature=0.2,
            max_chunk_chars=10,
            scope="page",
        )
        text = "First paragraph.\n\nSecond paragraph."

        with patch.object(
            service,
            "_translate_with_retry",
            return_value="整页译文",
        ) as mocked_translate, patch.object(
            service,
            "_iter_translation_blocks",
            side_effect=AssertionError("page scope should not iterate block translation"),
        ):
            result = service.translate_text_block(text)

        self.assertEqual(result, "整页译文")
        mocked_translate.assert_called_once()
        self.assertEqual(mocked_translate.call_args.args[0], text)

    def test_init_translator_uses_selected_provider(self) -> None:
        args = SimpleNamespace(
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="gemma4:26b",
            translation_reasoning_effort="none",
            translation_timeout_seconds=75.0,
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_scope="block",
            translation_latex_formula_handling="placeholder",
            annotation_mode="none",
            annotation_base_url=None,
            annotation_api_key=None,
            annotation_model=None,
            annotation_reasoning_effort=None,
            annotation_timeout_seconds=60.0,
        )

        with patch("translate_service.configure_openai", return_value="codex-client") as mocked_configure:
            service = init_translator(args)

        mocked_configure.assert_called_once_with(
            base_url="http://translate/v1",
            api_key="translate-key",
            timeout_seconds=75.0,
        )
        self.assertEqual(service.client, "codex-client")
        self.assertEqual(service.reasoning_effort, "none")
        self.assertEqual(service.scope, "block")
        self.assertIsInstance(service, GenericChatTranslator)
        self.assertFalse(hasattr(service, "annotator"))

    def test_init_translator_builds_translategemma_translator(self) -> None:
        args = SimpleNamespace(
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="translategemma-12b-it",
            translation_profile="translategemma",
            translation_source_lang="en",
            translation_target_lang="zh",
            translation_reasoning_effort="none",
            translation_timeout_seconds=75.0,
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_scope="block",
            translation_latex_formula_handling="placeholder",
            document_type=None,
            do_latex_repair=None,
            annotation_mode="none",
            annotation_base_url=None,
            annotation_api_key=None,
            annotation_model=None,
            annotation_reasoning_effort=None,
            annotation_timeout_seconds=60.0,
        )

        with patch("translate_service.configure_openai", return_value="codex-client"):
            service = init_translator(args)

        self.assertIsInstance(service, TranslateGemmaTranslator)
        self.assertEqual(service.source_lang, "en")
        self.assertEqual(service.target_lang, "zh")
        self.assertEqual(service.document_type, "general")
        self.assertFalse(service._do_latex_repair)

    def test_init_annotation_service_builds_service_with_own_backend(self) -> None:
        args = SimpleNamespace(
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="gemma4:26b",
            translation_reasoning_effort="none",
            translation_timeout_seconds=75.0,
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_scope="block",
            translation_latex_formula_handling="placeholder",
            annotation_mode="item",
            annotation_base_url="http://annotation/v1",
            annotation_api_key="annotation-key",
            annotation_model="gpt-4o-mini",
            annotation_reasoning_effort="medium",
            annotation_timeout_seconds=25.0,
        )

        with patch(
            "translate_service.configure_openai",
            return_value="annotation-client",
        ) as mocked_configure:
            annotation_service = init_annotation_service(args)

        mocked_configure.assert_called_once_with(
            base_url="http://annotation/v1",
            api_key="annotation-key",
            timeout_seconds=25.0,
        )
        self.assertIsNotNone(annotation_service)
        self.assertEqual(annotation_service.client, "annotation-client")
        self.assertEqual(annotation_service.model, "gpt-4o-mini")
        self.assertEqual(annotation_service.reasoning_effort, "medium")
        self.assertEqual(annotation_service.mode, "item")

    def test_init_translation_service_annotation_defaults_to_translation_backend(self) -> None:
        args = SimpleNamespace(
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="gemma4:26b",
            translation_reasoning_effort="low",
            translation_timeout_seconds=80.0,
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_scope="page",
            translation_latex_formula_handling="placeholder",
            annotation_mode="page",
            annotation_base_url=None,
            annotation_api_key=None,
            annotation_model=None,
            annotation_reasoning_effort=None,
            annotation_timeout_seconds=35.0,
        )

        with patch(
            "translate_service.configure_openai",
            side_effect=["translation-client", "annotation-client"],
        ) as mocked_configure:
            service = init_translation_service(args)

        self.assertEqual(mocked_configure.call_count, 2)
        self.assertEqual(
            mocked_configure.call_args_list[0].kwargs,
            {
                "base_url": "http://translate/v1",
                "api_key": "translate-key",
                "timeout_seconds": 80.0,
            },
        )
        self.assertEqual(
            mocked_configure.call_args_list[1].kwargs,
            {
                "base_url": "http://translate/v1",
                "api_key": "translate-key",
                "timeout_seconds": 35.0,
            },
        )
        self.assertIsNotNone(service.annotator)
        self.assertEqual(service.annotator.model, "gemma4:26b")
        self.assertEqual(service.annotator.mode, "page")
        self.assertEqual(service.scope, "page")

    def test_init_translation_service_attaches_annotation_service(self) -> None:
        args = SimpleNamespace(
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="gemma4:26b",
            translation_reasoning_effort="none",
            translation_timeout_seconds=80.0,
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_scope="block",
            translation_latex_formula_handling="placeholder",
            annotation_mode="item",
            annotation_base_url="http://annotation/v1",
            annotation_api_key="annotation-key",
            annotation_model="gpt-4o-mini",
            annotation_reasoning_effort="medium",
            annotation_timeout_seconds=25.0,
        )

        with patch(
            "translate_service.configure_openai",
            side_effect=["translation-client", "annotation-client"],
        ):
            service = init_translation_service(args)

        self.assertEqual(service.client, "translation-client")
        self.assertIsNotNone(service.annotator)
        self.assertEqual(service.annotator.client, "annotation-client")
        self.assertEqual(service.annotator.model, "gpt-4o-mini")

    def test_init_translation_service_preserves_translategemma_profile(self) -> None:
        args = SimpleNamespace(
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="translategemma-12b-it",
            translation_profile="translategemma",
            translation_source_lang="en",
            translation_target_lang="zh",
            translation_reasoning_effort="none",
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_scope="block",
            translation_latex_formula_handling="placeholder",
            document_type=None,
            do_latex_repair=None,
            annotation_mode=None,
            annotation_base_url=None,
            annotation_api_key=None,
            annotation_model=None,
            annotation_reasoning_effort=None,
        )

        with patch("translate_service.configure_openai", return_value="translation-client"):
            service = init_translation_service(args)

        self.assertIsInstance(service, TranslateGemmaTranslationService)
        self.assertEqual(service.source_lang, "en")
        self.assertEqual(service.target_lang, "zh")
        self.assertEqual(service.document_type, "general")
        self.assertFalse(service._do_latex_repair)
        self.assertIsNone(service.annotator)

    def test_validate_translation_args_defaults_translategemma_to_general_and_no_annotation(self) -> None:
        args = SimpleNamespace(
            translation_profile="translategemma",
            document_type=None,
            annotation_mode=None,
            do_latex_repair=None,
            translation_scope="block",
        )

        validate_translation_args(args)

        self.assertEqual(args.document_type, "general")
        self.assertEqual(args.annotation_mode, "none")
        self.assertFalse(args.do_latex_repair)

    def test_init_translation_service_rejects_item_annotation_with_page_scope(self) -> None:
        args = SimpleNamespace(
            translation_base_url="http://translate/v1",
            translation_api_key="translate-key",
            translation_model="gemma4:26b",
            translation_reasoning_effort="low",
            translation_temperature=0.2,
            translation_max_chunk_chars=1200,
            translation_scope="page",
            translation_latex_formula_handling="placeholder",
            annotation_mode="item",
            annotation_base_url=None,
            annotation_api_key=None,
            annotation_model=None,
            annotation_reasoning_effort=None,
        )

        with self.assertRaisesRegex(ValueError, "requires --translation-scope=block"):
            init_translation_service(args)


if __name__ == "__main__":
    unittest.main()
