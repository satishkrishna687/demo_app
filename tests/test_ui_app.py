"""UI smoke test: execute ui/app.py headlessly with Streamlit's AppTest.

This catches script-execution errors (e.g. imports) that a bare server-start health
check would miss, because Streamlit only runs the script when a client connects.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "ui" / "app.py"


def test_app_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_APP), default_timeout=60).run()
    assert not at.exception, f"ui/app.py raised: {list(at.exception)}"
    # The pipeline selector should render three options.
    assert at.selectbox, "pipeline selectbox did not render"
    assert len(at.selectbox[0].options) == 3