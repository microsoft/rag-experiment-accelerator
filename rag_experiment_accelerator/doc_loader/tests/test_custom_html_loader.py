from unittest.mock import Mock

from rag_experiment_accelerator.doc_loader.htmlLoader import load_html_files
from rag_experiment_accelerator.config.paths import get_all_file_paths


def test_load_html_files():
    chunks = load_html_files(
        environment=Mock(),
        file_paths=get_all_file_paths("./data/html"),
        chunk_size=1000,
        overlap_size=200,
    )

    assert len(chunks) == 20

    assert (
        "Deep Neural Nets: 33 years ago and 33 years from now"
        in list(chunks[0].values())[0]["content"]
    )
    assert (
        "Deep Neural Nets: 33 years ago and 33 years from now"
        not in list(chunks[5].values())[0]["content"]
    )
    assert "Musings of a Computer Scientist." in list(chunks[19].values())[0]["content"]
