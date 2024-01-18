from rag_experiment_accelerator.io.local.loaders.local_loader import LocalLoader


def test__get_file_ext():
    class TestLocalLoader(LocalLoader):
        def load(self, src: str, data, **kwargs):
            pass

        def can_handle(self, src: str):
            pass

    loader_impl = TestLocalLoader()

    filename = "test.txt"
    assert loader_impl._get_file_ext(filename) == ".txt"
