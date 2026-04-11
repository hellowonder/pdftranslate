import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from translate_stage import run_translate_pages_step, run_translate_stage  # noqa: E402


class FakeProgressBar:
    def __init__(self, *args, **kwargs) -> None:
        self.updated = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def update(self, value: int) -> None:
        self.updated += value


class TranslationStageProgressTest(unittest.TestCase):
    def test_uses_tqdm_directly_when_lock_is_available(self) -> None:
        args = SimpleNamespace(no_translation=True, translation_workers=1, skip_first_page_translation=False)
        output_paths = {"ocr_dir": "/unused", "translate_dir": "/unused"}
        progress_bar = FakeProgressBar()

        with (
            patch("translate_stage.read_page_markdown_files", return_value=["p1", "p2"]),
            patch("translate_stage.translate_page_markdown", side_effect=lambda *a, **k: a[3]),
            patch("translate_stage.tqdm", return_value=progress_bar),
        ):
            run_translate_pages_step(args, output_paths, [1, 2])

        self.assertEqual(progress_bar.updated, 2)

    def test_prints_progress_when_tqdm_lock_is_unavailable(self) -> None:
        args = SimpleNamespace(no_translation=True, translation_workers=1, skip_first_page_translation=False)
        output_paths = {"ocr_dir": "/unused", "translate_dir": "/unused"}
        fake_tqdm = SimpleNamespace()

        with (
            patch("translate_stage.read_page_markdown_files", return_value=["p1", "p2"]),
            patch("translate_stage.translate_page_markdown", side_effect=lambda *a, **k: a[3]),
            patch("translate_stage.tqdm", fake_tqdm),
            patch("translate_stage.print") as print_mock,
        ):
            run_translate_pages_step(args, output_paths, [1, 2])

        progress_messages = [call.args[0] for call in print_mock.call_args_list if call.args]
        self.assertIn("Translating pages (1 workers): 0/2", progress_messages)
        self.assertIn("Translating pages (1 workers): 1/2", progress_messages)
        self.assertIn("Translating pages (1 workers): 2/2", progress_messages)

    def test_run_translate_stage_wakes_and_sleeps_vllm(self) -> None:
        args = SimpleNamespace(
            no_translation=False,
            translation_base_url="http://localhost:8001/v1",
            translation_workers=1,
            translation_model="gemma4:26b",
            vllm_sleep=True,
        )
        output_paths = {
            "translated_markdown": "translated.md",
            "ocr_images_dir": "ocr_images",
            "translate_images_dir": "translate_images",
            "translate_dir": "translate",
        }

        with patch("translate_stage.translate_stage_outputs_exist", return_value=False), patch(
            "translate_stage.notify_vllm_control_endpoint"
        ) as mocked_notify, patch("translate_stage.sync_stage_images"), patch(
            "translate_stage.run_translate_pages_step"
        ), patch(
            "translate_stage.read_page_markdown_files", return_value=["第一页"]
        ), patch(
            "translate_stage.write_paged_markdown_document"
        ):
            run_translate_stage(args, output_paths, [1])

        self.assertEqual(
            mocked_notify.call_args_list,
            [
                call("http://localhost:8001/v1", "wake_up"),
                call("http://localhost:8001/v1", "sleep?level=1"),
            ],
        )

    def test_run_translate_stage_skips_vllm_lifecycle_when_no_translation(self) -> None:
        args = SimpleNamespace(
            no_translation=True,
            translation_base_url="http://localhost:8001/v1",
            translation_workers=1,
            translation_model="gemma4:26b",
            vllm_sleep=True,
        )
        output_paths = {
            "translated_markdown": "translated.md",
            "ocr_images_dir": "ocr_images",
            "translate_images_dir": "translate_images",
            "translate_dir": "translate",
        }

        with patch("translate_stage.translate_stage_outputs_exist", return_value=False), patch(
            "translate_stage.notify_vllm_control_endpoint"
        ) as mocked_notify, patch("translate_stage.sync_stage_images"), patch(
            "translate_stage.run_translate_pages_step"
        ), patch(
            "translate_stage.read_page_markdown_files", return_value=["page 1"]
        ), patch(
            "translate_stage.write_paged_markdown_document"
        ):
            run_translate_stage(args, output_paths, [1])

        mocked_notify.assert_not_called()

    def test_run_translate_stage_skips_vllm_lifecycle_when_vllm_sleep_disabled(self) -> None:
        args = SimpleNamespace(
            no_translation=False,
            translation_base_url="http://localhost:8001/v1",
            translation_workers=1,
            translation_model="gemma4:26b",
            vllm_sleep=False,
        )
        output_paths = {
            "translated_markdown": "translated.md",
            "ocr_images_dir": "ocr_images",
            "translate_images_dir": "translate_images",
            "translate_dir": "translate",
        }

        with patch("translate_stage.translate_stage_outputs_exist", return_value=False), patch(
            "translate_stage.notify_vllm_control_endpoint"
        ) as mocked_notify, patch("translate_stage.sync_stage_images"), patch(
            "translate_stage.run_translate_pages_step"
        ), patch(
            "translate_stage.read_page_markdown_files", return_value=["第一页"]
        ), patch(
            "translate_stage.write_paged_markdown_document"
        ):
            run_translate_stage(args, output_paths, [1])

        mocked_notify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
