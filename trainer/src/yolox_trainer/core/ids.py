from __future__ import annotations

import re
import secrets
from datetime import datetime

from .errors import InvalidArgument

ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"
_ID_RE = re.compile(ID_PATTERN)
# 64 - len("YYYYMMDD-HHMMSS-") - len("-xxxxxx")
_MAX_DATASET_PART = 41


def validate_id(value: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise InvalidArgument(f"invalid id: {value!r}")
    return value


def new_model_id(dataset_name: str, now: datetime, rand_hex: str | None = None) -> str:
    suffix = rand_hex if rand_hex is not None else secrets.token_hex(3)
    dataset = dataset_name.replace("_", "-")[:_MAX_DATASET_PART].rstrip("-")
    return f"{now:%Y%m%d-%H%M%S}-{dataset}-{suffix}"
