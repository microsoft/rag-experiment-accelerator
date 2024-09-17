from unittest.mock import Mock

from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.doc_loader.docxLoader import load_docx_files
from rag_experiment_accelerator.config.paths import get_all_file_paths


def test_load_docx_files():
    folder_path = "./data/docx"

    original_doc = load_docx_files(
        environment=Mock(),
        index_config=IndexConfig.from_dict(
            {"chunking": {"chunk_size": 1000, "overlap_size": 400}}
        ),
        file_paths=get_all_file_paths(folder_path),
    )

    assert len(original_doc) == 3

    assert "We recently commissioned" in list(original_doc[0].values())[0]["content"]
    assert "We recently commissioned" in list(original_doc[1].values())[0]["content"]
    assert (
        "We recently commissioned" not in list(original_doc[2].values())[0]["content"]
    )
