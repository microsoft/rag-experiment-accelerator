import os
import shutil
import uuid
import pytest

from rag_experiment_accelerator.writers.local.local_writer import LocalWriter


@pytest.fixture()
def temp_dirname():
    dir = "/tmp/" + uuid.uuid4().__str__()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


@pytest.fixture()
def writer_impl():
    # create a class that inherits from LocalWriter and implements the abstract method write
    class TestLocalWriter(LocalWriter):
        def write_file(self, path: str, data, **kwargs):
            pass

    yield TestLocalWriter()


def test__try_make_dir(temp_dirname: str, writer_impl: LocalWriter):
    writer_impl._try_make_dir(temp_dirname)
    assert os.path.exists(temp_dirname)


def test_copy(temp_dirname: str, writer_impl: LocalWriter):
    # create a file to copy
    src = temp_dirname + "/src.txt"
    os.makedirs(temp_dirname)
    with open(src, "w") as f:
        f.write("test")
    # copy the file
    dest = temp_dirname + "/dest.txt"
    writer_impl.copy(src, dest)
    # check that the file was copied
    assert os.path.exists(dest)


def test_delete(temp_dirname: str, writer_impl: LocalWriter):
    # create a file to delete
    src = temp_dirname + "/src.txt"
    os.makedirs(temp_dirname)
    with open(src, "w") as f:
        f.write("test")
    # delete the file
    writer_impl.delete(src)
    # check that the file was deleted
    assert not os.path.exists(src)


def test_list_filenames(temp_dirname: str, writer_impl: LocalWriter):
    # create a file to delete
    src = temp_dirname + "/src.txt"
    os.makedirs(temp_dirname)
    with open(src, "w") as f:
        f.write("test")
    # delete the file
    filenames = writer_impl.list_filenames(temp_dirname)
    # check that the file was deleted
    assert len(filenames) == 1
    assert filenames[0] == "src.txt"
