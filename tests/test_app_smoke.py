"""Smoke tests for Streamlit demonstration application and pages using AppTest.

Tests run the main application and all subpages, verifying clean execution
without unhandled exceptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from streamlit.testing.v1 import AppTest

# Skip entire test module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available, skipping app smoke tests.",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = REPO_ROOT / "app"
PAGES_DIR = APP_ROOT / "pages"

PAGE_FILES = [str(APP_ROOT / "streamlit_app.py")] + (
    sorted(str(p) for p in PAGES_DIR.glob("*.py")) if PAGES_DIR.is_dir() else []
)


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=lambda p: Path(p).name)
def test_app_pages_smoke(page_path: str) -> None:
    """Run app/streamlit_app.py and every page under app/pages with default_timeout=180 and assert no exceptions."""
    at = AppTest.from_file(page_path, default_timeout=180)
    at.run()
    assert not at.exception, f"Exceptions occurred while running {Path(page_path).name}: {at.exception}"


def test_app_select_dataset_sample() -> None:
    """Select a dataset sample image and run the page again to execute resize and compress code paths."""
    app_file = str(APP_ROOT / "streamlit_app.py")
    at = AppTest.from_file(app_file, default_timeout=180)
    at.run()
    assert not at.exception, f"Exceptions during initial run: {at.exception}"

    # Find the 'Choose Sample' selectbox in the sidebar
    sample_selectboxes = [
        sb for sb in at.sidebar.selectbox if "Choose Sample" in getattr(sb.proto, "label", "")
    ]
    assert len(sample_selectboxes) > 0, "Could not find 'Choose Sample' selectbox"
    sample_sb = sample_selectboxes[0]

    assert len(sample_sb.options) > 0, "No sample image options found in selectbox"
    # Select a dataset sample and re-run to trigger image loading, resize_long_side, and compression
    target_option = sample_sb.options[1] if len(sample_sb.options) > 1 else sample_sb.options[0]
    sample_sb.select(target_option).run()

    assert not at.exception, f"Exceptions occurred after selecting sample: {at.exception}"
