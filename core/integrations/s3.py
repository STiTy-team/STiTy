import gzip
import json
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import NamedTuple

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from core.utils.json import dumps

CONFLICT_CODES = {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}
MISSING_CODES = {"NoSuchKey", "404"}
CLOCK_KEY = ".clock"


class Conflict(Exception):
    pass


class Stored(NamedTuple):
    data: object
    etag: str


def error_code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", ""))


class Bucket:
    def __init__(self, name: str, prefix: str, client: BaseClient | None = None):
        self.name = name
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        self.client = client or boto3.client("s3")

    def get_bytes(self, key: str) -> Stored | None:
        try:
            response = self._request("get_object", key)
        except ClientError as e:
            if error_code(e) in MISSING_CODES:
                return None
            raise
        return Stored(response["Body"].read(), response["ETag"])

    def get_json(self, key: str) -> Stored | None:
        stored = self.get_bytes(key)
        return None if stored is None else Stored(json.loads(stored.data), stored.etag)

    def put_bytes(
        self,
        key: str,
        body: bytes,
        *,
        if_match: str | None = None,
        if_absent: bool = False,
        content_type: str = "application/octet-stream",
    ) -> str:
        conditions = {}
        if if_match is not None:
            conditions["IfMatch"] = if_match
        if if_absent:
            conditions["IfNoneMatch"] = "*"
        try:
            response = self._request(
                "put_object", key, Body=body, ContentType=content_type, **conditions
            )
        except ClientError as e:
            if if_match is not None and error_code(e) in MISSING_CODES:
                raise Conflict(f"{key}: deleted by someone else") from e
            raise
        return response["ETag"]

    def put_json(
        self, key: str, data: dict | list, *, if_match: str | None = None, if_absent: bool = False
    ) -> str:
        return self.put_bytes(
            key,
            dumps(data, indent=2).encode("utf-8"),
            if_match=if_match,
            if_absent=if_absent,
            content_type="application/json",
        )

    def put_file(
        self, key: str, path: Path, *, gzipped: bool = False, if_absent: bool = False
    ) -> str:
        body = path.read_bytes()
        return self.put_bytes(key, gzip.compress(body) if gzipped else body, if_absent=if_absent)

    def delete(self, key: str, *, if_match: str | None = None) -> None:
        if if_match is None:
            self._request("delete_object", key)
            return
        try:
            self._request("delete_object", key, IfMatch=if_match)
        except ClientError as e:
            if error_code(e) in MISSING_CODES:
                raise Conflict(f"{key}: already deleted by someone else") from e
            raise

    def list(self, prefix: str = "") -> list[str]:
        return list(self.modified(prefix))

    def modified(self, prefix: str = "") -> dict[str, datetime]:
        return {key: item["LastModified"] for key, item in self._listing(prefix)}

    def etags(self, prefix: str = "") -> dict[str, str]:
        return {key: item["ETag"] for key, item in self._listing(prefix)}

    def _listing(self, prefix: str):
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.name, Prefix=self.prefix + prefix
        )
        for page in pages:
            for item in page.get("Contents", []):
                yield item["Key"].removeprefix(self.prefix), item

    def now(self) -> datetime:
        """S3's clock, read from the Date header of a HEAD on a key that never exists.

        A HEAD costs what a GET does, a thirteenth of a LIST. Its 404 still carries the header,
        and so does the 403 S3 answers when the caller may not list that key's prefix.
        """
        try:
            response = self.client.head_object(Bucket=self.name, Key=self.prefix + CLOCK_KEY)
        except ClientError as e:
            if error_code(e) not in MISSING_CODES | {"403"}:
                raise
            response = e.response
        server_date = response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("date")
        if server_date is None:
            return datetime.now(UTC)
        return parsedate_to_datetime(server_date).astimezone(UTC)

    def _request(self, operation: str, key: str, **params) -> dict:
        try:
            return getattr(self.client, operation)(
                Bucket=self.name, Key=self.prefix + key, **params
            )
        except ClientError as e:
            if error_code(e) in CONFLICT_CODES:
                raise Conflict(f"{key}: changed by someone else") from e
            raise
