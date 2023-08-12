import hashlib
from typing import Union

import dask.dataframe as dd
import pandas as pd

DataFrame = Union[pd.DataFrame, dd.DataFrame]


def get_ununsed_column_name(df: DataFrame, prefix: str) -> str:
    columns = set(df.columns)
    out = prefix
    while prefix in columns:
        out = f"{prefix}_{generate_hash(out)}"
    return prefix


def generate_hash(data):
    # Create a new SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Convert the data to bytes if it's not already
    if not isinstance(data, bytes):
        data = str(data).encode("utf-8")

    # Update the hash object with the data
    sha256_hash.update(data)

    # Get the hexadecimal representation of the hash
    hash_value = sha256_hash.hexdigest()

    return hash_value
