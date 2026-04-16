import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_files
from tqdm import tqdm
from run_logger import RunLogger


IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE_NAME = "aclImdb_v1.tar.gz"
LOGGER = RunLogger("imdb_svm_progress.jsonl")


def resolve_data_root(data_root="data_cache"):
    """
    Resolve the cache root for IMDB data.

    Priority:
    1. Explicit environment override OFFICIAL_QOS_IMDB_DATA_ROOT
    2. Caller-provided data_root
    """
    override = os.environ.get("OFFICIAL_QOS_IMDB_DATA_ROOT")
    if override:
        return str(Path(override).expanduser())
    return data_root


def _safe_extract_tar(tar, path):
    abs_path = os.path.abspath(path)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(path, member.name))
        if not member_path.startswith(abs_path + os.sep) and member_path != abs_path:
            raise RuntimeError("Unsafe path detected in tar archive.")
    tar.extractall(path)


def download_imdb_data(data_root="data_cache"):
    """
    Downloads and extracts the IMDB dataset into data_root.
    Returns the path to the extracted aclImdb directory.
    """
    data_root = resolve_data_root(data_root)
    os.makedirs(data_root, exist_ok=True)
    archive_path = os.path.join(data_root, IMDB_ARCHIVE_NAME)
    imdb_path = os.path.join(data_root, "aclImdb")
    LOGGER.log(
        "imdb_cache_resolved",
        data_root=str(data_root),
        archive_path=str(archive_path),
        imdb_path=str(imdb_path),
    )

    if not os.path.exists(archive_path):
        LOGGER.log("imdb_download_start", archive_path=str(archive_path), url=IMDB_URL)
        tqdm.write(f"Downloading IMDB dataset to {archive_path}...")
        urllib.request.urlretrieve(IMDB_URL, archive_path)
        LOGGER.log("imdb_download_done", archive_path=str(archive_path))
    else:
        LOGGER.log("imdb_archive_present", archive_path=str(archive_path))
        tqdm.write(f"IMDB archive already exists at {archive_path}, skipping download.")

    if not os.path.exists(imdb_path):
        LOGGER.log("imdb_extract_start", archive_path=str(archive_path), data_root=str(data_root))
        tqdm.write(f"Extracting IMDB dataset to {data_root}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract_tar(tar, data_root)
        LOGGER.log("imdb_extract_done", imdb_path=str(imdb_path))
    else:
        LOGGER.log("imdb_extract_present", imdb_path=str(imdb_path))

    return imdb_path


def load_imdb_data(download_if_missing=True, data_root="data_cache"):
    """
    Loads the full IMDB dataset from aclImdb or data_root/aclImdb.
    Returns (data, target), where data is a list of strings and target is a list/array of labels.
    """
    # Check both potential locations
    data_root = resolve_data_root(data_root)
    potential_paths = [
        os.path.join(data_root, "aclImdb"),
        "aclImdb",
    ]
    LOGGER.log(
        "imdb_load_paths",
        download_if_missing=bool(download_if_missing),
        data_root=str(data_root),
        potential_paths=[str(p) for p in potential_paths],
    )
    imdb_path = None
    for p in potential_paths:
        if os.path.exists(p):
            imdb_path = p
            LOGGER.log("imdb_path_found", imdb_path=str(imdb_path))
            break

    if imdb_path is None and download_if_missing:
        try:
            download_imdb_data(data_root=data_root)
        except Exception as exc:
            raise FileNotFoundError(
                f"IMDB dataset not found in {potential_paths} and download failed."
            ) from exc
        for p in potential_paths:
            if os.path.exists(p):
                imdb_path = p
                LOGGER.log("imdb_path_found_post_download", imdb_path=str(imdb_path))
                break

    if imdb_path is None:
        raise FileNotFoundError(
            f"IMDB dataset not found in {potential_paths}. Please download it from "
            f"{IMDB_URL} and extract it."
        )

    LOGGER.log("imdb_train_load_start", imdb_path=str(imdb_path))
    tqdm.write("Loading IMDB Train Data...")
    train_data = load_files(
        os.path.join(imdb_path, "train"), categories=["pos", "neg"], encoding="utf-8"
    )
    LOGGER.log(
        "imdb_train_load_done",
        num_train_docs=int(len(train_data.data)),
        num_train_labels=int(len(train_data.target)),
    )

    LOGGER.log("imdb_test_load_start", imdb_path=str(imdb_path))
    tqdm.write("Loading IMDB Test Data...")
    test_data = load_files(
        os.path.join(imdb_path, "test"), categories=["pos", "neg"], encoding="utf-8"
    )
    LOGGER.log(
        "imdb_test_load_done",
        num_test_docs=int(len(test_data.data)),
        num_test_labels=int(len(test_data.target)),
    )

    # Combine Train and Test
    all_data = train_data.data + test_data.data
    all_target = np.concatenate([train_data.target, test_data.target])
    LOGGER.log(
        "imdb_combine_done",
        total_docs=int(len(all_data)),
        total_labels=int(len(all_target)),
    )

    return all_data, all_target
