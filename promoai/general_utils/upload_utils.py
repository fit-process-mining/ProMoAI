"""Temporary storage for files consumed by process-mining importers."""

import tempfile
from pathlib import Path

from promoai.general_utils.artifact_store import get_staging_dir


def write_uploaded_file_to_staging(uploaded_file, suffix: str | None = None) -> str:
    file_suffix = suffix or "".join(Path(uploaded_file.name).suffixes).lower() or ".bin"
    if file_suffix == ".gz":
        file_suffix = ".xes.gz"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=get_staging_dir("promoai_uploads"),
        suffix=file_suffix,
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name
