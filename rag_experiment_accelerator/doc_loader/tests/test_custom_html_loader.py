from rag_experiment_accelerator.doc_loader.htmlLoader import load_html_files
from rag_experiment_accelerator.config.paths import get_all_files


def test_load_html_files():
    folder_path = "./data/html"
    chunk_size = 1000
    overlap_size = 200

    chunks = load_html_files(
        file_paths=get_all_files(folder_path),
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )

    assert len(chunks) == 20

    assert (
        "Deep Neural Nets: 33 years ago and 33 years from now" in chunks[0].page_content
    )
    assert (
        "Deep Neural Nets: 33 years ago and 33 years from now"
        not in chunks[5].page_content
    )
    assert "Musings of a Computer Scientist." in chunks[19].page_content
