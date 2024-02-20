from rag_experiment_accelerator.doc_loader.docxLoader import load_docx_files


def test_load_docx_files():
    folder_path = "./data/"
    chunk_size = 1000
    overlap_size = 400

    original_doc = load_docx_files(
        chunking_strategy="basic",
        AzureDocumentIntelligenceCredentials=None,
        folder_path=folder_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )

    assert len(original_doc) == 3

    assert "We recently commissioned" in list(original_doc[0].values())[0]
    assert "We recently commissioned" in list(original_doc[1].values())[0]
    assert "We recently commissioned" not in list(original_doc[2].values())[0]
