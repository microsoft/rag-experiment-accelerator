from unittest.mock import Mock

import pytest
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.doc_loader.pdfLoader import load_pdf_files


@pytest.mark.parametrize(
    "pypdf_enabled, expected_content",
    [
        (True, "this is a sample pdf with text."),
        (
            False,
            "this is a sample of text captured as an image and stored as pdf.this is a sample pdf with text.",
        ),
    ],
)
def test_load_pdf_files(pypdf_enabled: bool, expected_content: str):
    file_path = "./data/pdf/text-as-image.pdf"

    environment_mock = Mock()
    actual_docs = load_pdf_files(
        environment=environment_mock,
        index_config=IndexConfig.from_dict({"pypdf_enabled": pypdf_enabled}),
        file_paths=[file_path],
    )

    assert len(actual_docs) == 1
    actual_content = list(actual_docs[0].values())[0]["content"]
    assert actual_content == expected_content
