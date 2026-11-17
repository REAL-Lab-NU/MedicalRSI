"""HAPI FHIR JPA search-index hashes (for the offline FHIR server).

HAPI stores token parameters with hash columns computed by BaseResourceIndexedSearchParam.hash(): Guava
Hashing.murmur3_128(0) over the UTF-8 bytes of each component (URL-escaped) followed by the delimiter "|", read with
HashCode.asLong() (= the first 64-bit word of MurmurHash3_x64_128, little endian, signed). An OR list such as
code=GLU,K becomes HASH_VALUE IN (...) and the database returns the rows grouped by ascending hash, then by resource
id - which is the order the mock must reproduce.
"""
from __future__ import annotations

import urllib.parse

M64 = (1 << 64) - 1


def _rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & M64


def _fmix(k):
    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & M64
    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & M64
    k ^= k >> 33
    return k


def murmur3_x64_128_h1(data: bytes, seed: int = 0) -> int:
    c1, c2 = 0x87C37B91114253D5, 0x4CF5AD432745937F
    h1 = h2 = seed & M64
    n = len(data)
    nblocks = n // 16
    for i in range(nblocks):
        k1 = int.from_bytes(data[16 * i: 16 * i + 8], "little")
        k2 = int.from_bytes(data[16 * i + 8: 16 * i + 16], "little")
        k1 = (k1 * c1) & M64
        k1 = _rotl(k1, 31)
        k1 = (k1 * c2) & M64
        h1 ^= k1
        h1 = _rotl(h1, 27)
        h1 = (h1 + h2) & M64
        h1 = (h1 * 5 + 0x52DCE729) & M64
        k2 = (k2 * c2) & M64
        k2 = _rotl(k2, 33)
        k2 = (k2 * c1) & M64
        h2 ^= k2
        h2 = _rotl(h2, 31)
        h2 = (h2 + h1) & M64
        h2 = (h2 * 5 + 0x38495AB5) & M64
    tail = data[16 * nblocks:]
    k1 = k2 = 0
    if len(tail) > 8:
        k2 = int.from_bytes(tail[8:], "little")
        k2 = (k2 * c2) & M64
        k2 = _rotl(k2, 33)
        k2 = (k2 * c1) & M64
        h2 ^= k2
    if len(tail) > 0:
        k1 = int.from_bytes(tail[:8], "little")
        k1 = (k1 * c1) & M64
        k1 = _rotl(k1, 31)
        k1 = (k1 * c2) & M64
        h1 ^= k1
    h1 ^= n
    h2 ^= n
    h1 = (h1 + h2) & M64
    h2 = (h2 + h1) & M64
    h1 = _fmix(h1)
    h2 = _fmix(h2)
    h1 = (h1 + h2) & M64
    return h1 - (1 << 64) if h1 >= (1 << 63) else h1


def escape(v: str) -> str:
    """UrlUtil.escapeUrlParam: java.net.URLEncoder with UTF-8 ('+' for spaces)."""
    return urllib.parse.quote_plus(v, safe="*").replace("~", "%7E")


def hapi_hash(*values) -> int:
    data = b""
    for v in values:
        data += b"\x00" if v is None else escape(v).encode("utf-8")
        data += b"|"
    return murmur3_x64_128_h1(data)
