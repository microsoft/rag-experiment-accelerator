from rag_experiment_accelerator.doc_loader.htmlLoader import load_html_files


def test_load_html_files():
    chunking_strategy = "basic"
    AzureDocumentIntelligenceCredentials = None
    folder_path = "./data/"
    chunk_size = 1000
    overlap_size = 200
    glob_patterns: list[str] = ["html", "htm", "xhtml", "html5"]

    chunks = load_html_files(
        chunking_strategy=chunking_strategy,
        AzureDocumentIntelligenceCredentials=AzureDocumentIntelligenceCredentials,
        folder_path=folder_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        glob_patterns=glob_patterns,
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
