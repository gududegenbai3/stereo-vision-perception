"""Read only the evaluation members from the official Booster ZIP64 archive."""

import hashlib
from pathlib import Path
import struct
import time
import urllib.request
import zipfile
import zlib

def http_range(url: str, size: int, begin: int, end: int) -> bytes:
    """Require exact partial-response identity, not a silent full ZIP download."""
    request: urllib.request.Request = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
            raise ValueError("Official server did not honor the requested archive range")
        payload: bytes = response.read()
    if len(payload) != end-begin+1:
        raise ValueError("Incomplete archive range")
    return payload


def parse_central_directory(payload: bytes) -> list[dict[str, object]]:
    """Parse ZIP/ZIP64 metadata, never execute or extract archive contents."""
    records: list[dict[str, object]] = []
    position: int = 0
    while position < len(payload):
        values: tuple = struct.unpack_from("<4s6H3I5H2I", payload, position)
        if values[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP central-directory signature")
        name_size, extra_size, comment_size = values[10:13]
        name: str = payload[position+46:position+46+name_size].decode("utf-8" if values[3] & 0x800 else "cp437")
        extra: bytes = payload[position+46+name_size:position+46+name_size+extra_size]
        uncompressed, compressed, offset = values[9], values[8], values[16]
        if 0xffffffff in (uncompressed, compressed, offset):
            index: int = 0
            found: bool = False
            while index < len(extra):
                tag, length = struct.unpack_from("<HH", extra, index)
                content: bytes = extra[index+4:index+4+length]
                index += 4+length
                if tag == 1:
                    cursor: int = 0
                    if uncompressed == 0xffffffff:
                        uncompressed = struct.unpack_from("<Q", content, cursor)[0]
                        cursor += 8
                    if compressed == 0xffffffff:
                        compressed = struct.unpack_from("<Q", content, cursor)[0]
                        cursor += 8
                    if offset == 0xffffffff:
                        offset = struct.unpack_from("<Q", content, cursor)[0]
                    found = True
                    break
            if not found:
                raise ValueError("ZIP64 size/offset marker lacks its extra record")
        records.append({"name": name, "crc32": values[7], "compressed_size": compressed,
            "size": uncompressed, "offset": offset, "compression": values[4]})
        position += 46+name_size+extra_size+comment_size
    return records


def booster_index(url: str, size: int) -> list[dict[str, object]]:
    tail: bytes = http_range(url, size, max(0, size-65536), size-1)
    end_offset: int = tail.rfind(b"PK\x05\x06")
    if end_offset < 0:
        raise ValueError("ZIP end record missing")
    end: tuple = struct.unpack_from("<4s4H2LH", tail, end_offset)
    length, offset = end[5], end[6]
    if offset == 0xffffffff or length == 0xffffffff:
        zip64_offset: int = tail.rfind(b"PK\x06\x06")
        if zip64_offset < 0:
            raise ValueError("ZIP64 end record missing from bounded tail")
        zip64: tuple = struct.unpack_from("<4sQ2H2L4Q", tail, zip64_offset)
        length, offset = zip64[-2], zip64[-1]
    if length > 16*1024*1024 or offset+length > size:
        raise ValueError("Unexpected archive-index bounds")
    return parse_central_directory(http_range(url, size, offset, offset+length-1))


def download_member(url: str, archive_size: int, member: dict[str, object], root: Path) -> dict[str, object]:
    """Fetch one stored/deflated ZIP member, verify exact bytes/CRC, prevent traversal."""
    name: str = member["name"]
    target: Path = root/name
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe archive member: {name}")
    if target.exists():
        payload: bytes = target.read_bytes()
        if len(payload) != member["size"] or zlib.crc32(payload) != member["crc32"]:
            raise ValueError(f"Existing file differs from archive: {name}")
    else:
        begin: int = member["offset"]
        end: int = min(archive_size-1, begin+30+len(name.encode())+65535+member["compressed_size"]-1)
        request: urllib.request.Request = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}", "User-Agent": "stereo-course-research/1.0"})
        downloaded: bytes | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{archive_size}":
                        raise ValueError("Server did not honor exact HTTP range")
                    downloaded = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2*(attempt+1))
        if downloaded is None or len(downloaded) != end-begin+1:
            raise ValueError("Truncated HTTP ZIP range")
        header: tuple = struct.unpack_from("<4s5H3I2H", downloaded)
        if header[0] != b"PK\x03\x04":
            raise ValueError("Invalid local ZIP header")
        start: int = 30+header[-2]+header[-1]
        encoded: bytes = downloaded[start:start+member["compressed_size"]]
        if member["compression"] == zipfile.ZIP_DEFLATED:
            payload = zlib.decompress(encoded, -15)
        elif member["compression"] == zipfile.ZIP_STORED:
            payload = encoded
        else:
            raise ValueError("Unsupported ZIP compression")
        if len(payload) != member["size"] or zlib.crc32(payload) != member["crc32"]:
            raise ValueError(f"ZIP size/CRC mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path = target.with_suffix(target.suffix+".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
    return {"name": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "archive_offset": member["offset"], "archive_crc32": member["crc32"]}
