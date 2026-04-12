import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import call, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from ocr_stage import build_vllm_control_url, run_ocr_stage  # noqa: E402


class OCRStageVllmLifecycleTest(unittest.TestCase):
    def test_build_vllm_control_url_uses_base_origin(self) -> None:
        self.assertEqual(
            build_vllm_control_url("http://localhost:8000/v1", "wake_up"),
            "http://localhost:8000/wake_up",
        )
        self.assertEqual(
            build_vllm_control_url("https://example.com:9000/custom/v1", "sleep?level=1"),
            "https://example.com:9000/sleep?level=1",
        )

    def test_run_ocr_stage_wakes_and_sleeps_when_vllm_sleep_enabled(self) -> None:
        args = argparse.Namespace(
            vllm_sleep=True,
            ocr_base_url="http://localhost:8000/v1",
            ocr_model="model",
            ocr_dpi=96,
            ocr_workers=1,
        )
        output_paths = {
            "input_pdf": "input.pdf",
            "ocr_raw_dir": "ocr_raw",
            "ocr_input_images_dir": "ocr_input_images",
            "ocr_images_dir": "ocr_images",
            "ocr_dir": "ocr",
            "ocr_page_merge": "merge.json",
        }

        with patch("ocr_stage.all_raw_ocr_outputs_exist", return_value=False), patch(
            "ocr_stage.pdf_to_images_high_quality", return_value=["image"]
        ), patch("ocr_stage.notify_vllm_control_endpoint") as mocked_notify, patch(
            "ocr_stage.init_ocr_client", return_value="ocr_client"
        ), patch("ocr_stage.run_ocr_pages", return_value=["page markdown"]), patch(
            "ocr_stage.apply_pdf_bold_marks", return_value=["page markdown"]
        ), patch(
            "ocr_stage.page_merge_outputs_exist", return_value=True
        ):
            run_ocr_stage(args, output_paths, [1])

        self.assertEqual(
            mocked_notify.call_args_list,
            [
                call("http://localhost:8000/v1", "wake_up"),
                call("http://localhost:8000/v1", "sleep?level=1"),
            ],
        )

    def test_run_ocr_stage_skips_vllm_lifecycle_when_vllm_sleep_disabled(self) -> None:
        args = argparse.Namespace(
            vllm_sleep=False,
            ocr_base_url="http://localhost:8000/v1",
            ocr_model="model",
            ocr_dpi=96,
            ocr_workers=1,
        )
        output_paths = {
            "input_pdf": "input.pdf",
            "ocr_raw_dir": "ocr_raw",
            "ocr_input_images_dir": "ocr_input_images",
            "ocr_images_dir": "ocr_images",
            "ocr_dir": "ocr",
            "ocr_page_merge": "merge.json",
        }

        with patch("ocr_stage.all_raw_ocr_outputs_exist", return_value=False), patch(
            "ocr_stage.pdf_to_images_high_quality", return_value=["image"]
        ), patch("ocr_stage.notify_vllm_control_endpoint") as mocked_notify, patch(
            "ocr_stage.init_ocr_client", return_value="ocr_client"
        ), patch("ocr_stage.run_ocr_pages", return_value=["page markdown"]), patch(
            "ocr_stage.apply_pdf_bold_marks", return_value=["page markdown"]
        ), patch(
            "ocr_stage.page_merge_outputs_exist", return_value=True
        ):
            run_ocr_stage(args, output_paths, [1])

        mocked_notify.assert_not_called()

    def test_run_ocr_stage_skips_image_loading_when_raw_and_merged_outputs_exist(self) -> None:
        args = argparse.Namespace(
            vllm_sleep=False,
            ocr_base_url="http://localhost:8000/v1",
            ocr_model="model",
            ocr_dpi=96,
            ocr_workers=1,
        )
        output_paths = {
            "input_pdf": "input.pdf",
            "ocr_raw_dir": "ocr_raw",
            "ocr_input_images_dir": "ocr_input_images",
            "ocr_images_dir": "ocr_images",
            "ocr_dir": "ocr",
            "ocr_page_merge": "merge.json",
        }

        with patch("ocr_stage.all_raw_ocr_outputs_exist", return_value=True), patch(
            "ocr_stage.page_merge_outputs_exist", return_value=True
        ), patch("ocr_stage.pdf_to_images_high_quality") as mocked_pdf_loader:
            run_ocr_stage(args, output_paths, [1])

        mocked_pdf_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
