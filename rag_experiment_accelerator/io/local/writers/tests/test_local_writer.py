import os
import shutil
import tempfile
import uuid

import pytest

from rag_experiment_accelerator.io.exceptions import CopyException, WriteException
from rag_experiment_accelerator.io.local.writers.local_writer import LocalWriter


@pytest.fixture()
def temp_dirname():
    # get temp dir name but don't create it
    dir = "/tmp/" + uuid.uuid4().__str__()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


@pytest.fixture()
def temp_dir():
    # create a temp dir
    dir = tempfile.mkdtemp()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


@pytest.fixture()
def writer_impl():
    # create a class that inherits from LocalWriter and implements the abstract method _write_file
    # optionally raises to test exception handling
    # marks the method as called to test if it was called
    class TestLocalWriter(LocalWriter):
        def _write_file(self, path: str, data, **kwargs):
            should_raise = kwargs.pop("should_raise", False)
            if should_raise:
                raise Exception()
            self._write_file_called = True

    yield TestLocalWriter()


def test__make_dir(temp_dirname: str, writer_impl: LocalWriter):
    # make dir
    writer_impl._make_dir(temp_dirname)

    # ensure dir was created
    assert os.path.exists(temp_dirname)


def test__make_dir_raises(writer_impl: LocalWriter):
    # try to make dir in sudo location
    with pytest.raises(Exception):
        writer_impl._make_dir("/test123")

    assert not os.path.exists("/test123")


def test_write_calls__write_file(temp_dir: str, writer_impl: LocalWriter):
    # set path
    path = f"{temp_dir}/test.txt"

    # call write
    writer_impl.write(path, "test")

    # ensure _write_file was called
    assert writer_impl._write_file_called is True


def test_write_creates_parent_dir(temp_dirname: str, writer_impl: LocalWriter):
    path = f"{temp_dirname}/test.txt"

    writer_impl.write(path, "test")

    assert os.path.exists(temp_dirname)


def test_write_raises_write_exception(writer_impl: LocalWriter):
    path = "/tmp/test.txt"
    with pytest.raises(WriteException):
        writer_impl.write(path, "test", should_raise=True)


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


def test_copy_raises_copy_exception(temp_dirname: str, writer_impl: LocalWriter):
    # create a file to copy
    src = temp_dirname + "/src.txt"
    os.makedirs(temp_dirname)
    with open(src, "w") as f:
        f.write("test")

    # copy the file to location that needs sudo permissions
    dest = "/dest.txt"
    with pytest.raises(CopyException):
        writer_impl.copy(src, dest)

    # should not have been copied
    assert not os.path.exists(dest)


def test_copy_raises_file_not_found(temp_dirname: str, writer_impl: LocalWriter):
    # create src dir but don't create the file
    src = temp_dirname + "/src.txt"
    os.makedirs(temp_dirname)
    dest = temp_dirname + "/dest.txt"

    with pytest.raises(FileNotFoundError):
        writer_impl.copy(src, dest)

    # should not have been copied
    assert not os.path.exists(dest)


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
    # create a files in a temp dir
    file1 = temp_dirname + "/src1.txt"
    os.makedirs(temp_dirname)
    with open(file1, "w") as f:
        f.write("test")
    file2 = temp_dirname + "/src2.txt"
    with open(file2, "w") as f:
        f.write("test")

    # list files
    filenames = writer_impl.list_filenames(temp_dirname)

    # check all filenames are returned
    assert len(filenames) == 2
    assert "src1.txt" in filenames
    assert "src2.txt" in filenames
