"""Pydantic schemas for the ``.can`` artifact format (v0.26.0 Part E)."""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

CAN_FORMAT_VERSION = 1

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,127}$")
_HF_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-./]{0,127}$")


class DataRef(BaseModel):
    """How to fetch the training data after unpacking a can.

    ``kind`` is one of:
      - ``url``: HTTPS URL pointing at a JSONL file
      - ``hf``: HuggingFace dataset id (``org/dataset``)
      - ``local``: relative path — user must supply it locally
    """

    kind: Literal["url", "hf", "local"] = Field(
        description="Data source type",
    )
    url: Optional[str] = Field(default=None, description="HTTPS URL")
    hf_dataset: Optional[str] = Field(default=None, description="HF dataset id")
    local_path: Optional[str] = Field(default=None, description="Relative local path")

    @field_validator("url")
    @classmethod
    def _https_only(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not value.startswith("https://"):
            raise ValueError(
                f"data_ref.url must be https:// (got: {value}) - "
                "plain http is forbidden for remote fetches"
            )
        return value

    @field_validator("hf_dataset")
    @classmethod
    def _valid_hf_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not _HF_NAME_RE.match(value):
            raise ValueError(
                f"hf_dataset '{value}' is invalid - "
                "use 'org/name' with alphanumeric + _-./"
            )
        return value


class Manifest(BaseModel):
    """Top-level metadata for a ``.can`` file."""

    can_format_version: int = Field(description="Format version integer")
    name: str = Field(description="Recipe name")
    author: str = Field(description="Author handle", max_length=128)
    created_at: str = Field(description="ISO-8601 timestamp or YYYY-MM-DD")
    base_hash: str = Field(description="SHA-256 of the config (from registry)")
    description: Optional[str] = Field(default=None, max_length=4096)
    tags: list[str] = Field(default_factory=list)

    @field_validator("can_format_version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value != CAN_FORMAT_VERSION:
            raise ValueError(
                f"unknown can_format_version {value}; this build of Soup "
                f"only supports version {CAN_FORMAT_VERSION}. Upgrade Soup "
                "or re-pack the can with an older format."
            )
        return value

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError(
                f"can name '{value}' is invalid - "
                "alphanumeric + _-. only, must start with alphanumeric"
            )
        return value

    @field_validator("author")
    @classmethod
    def _clean_author(cls, value: str) -> str:
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("author must not contain null bytes or newlines")
        return value

    @field_validator("created_at")
    @classmethod
    def _parseable_created_at(cls, value: str) -> str:
        # Accept YYYY-MM-DD or any ISO-8601 datetime parseable by fromisoformat
        from datetime import datetime as _dt

        try:
            _dt.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"created_at '{value}' is not valid ISO-8601"
            ) from exc
        return value
