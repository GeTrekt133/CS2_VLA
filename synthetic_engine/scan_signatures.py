"""Scan CS2 server.dll for CS2TraceRay's gamedata signatures.

Tells us exactly which signatures match the user's pinned build.
Patterns: hex bytes separated by spaces; "?" or "??" means wildcard.

Usage:
    python scan_signatures.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SERVER_DLL = Path('D:/CS2_pinned/game/csgo/bin/win64/server.dll')

PATTERNS = {
    'TraceFunc (OLD CS2TraceRay)':  '4C 8B DC 49 89 5B ? 49 89 6B ? 49 89 73 ? 57 41 56 41 57 48 81 EC',
    'TraceFunc (NEW CS2Fixes)':     '48 89 5C 24 ? 48 89 6C 24 ? 48 89 74 24 ? 48 89 7C 24 ? 41 54 41 56 41 57 48 81 EC ? ? ? ? 45 33 E4',
    'TraceShape':         '48 89 5C 24 ? 48 89 4C 24 ? 55 57',
    'GameTraceManager':   '48 8B 0D ? ? ? ? 0C',
    'CTraceFilterVtable': '4C 8D 2D ? ? ? ? 24',
}


def hex_pattern_to_regex(pat: str) -> bytes:
    """Convert '4C 8B DC ? 49' → re.compile(b'\\x4c\\x8b\\xdc.\\x49', re.S)."""
    out = b''
    for tok in pat.split():
        if tok in ('?', '??'):
            out += b'.'
        else:
            out += b'\\x' + tok.encode().lower()
    return re.compile(out, re.S)


def main():
    if not SERVER_DLL.exists():
        print(f'server.dll not found at {SERVER_DLL}'); sys.exit(1)

    data = SERVER_DLL.read_bytes()
    print(f'server.dll size: {len(data) / 1e6:.1f} MB\n')

    for name, pat in PATTERNS.items():
        regex = hex_pattern_to_regex(pat)
        hits = list(regex.finditer(data))
        n = len(hits)
        if n == 0:
            tag = 'NOT FOUND'
        elif n == 1:
            tag = f'unique  @0x{hits[0].start():08x}'
        else:
            offsets = ', '.join(f'0x{m.start():08x}' for m in hits[:5])
            tag = f'{n} matches: {offsets}{"..." if n > 5 else ""}'
        print(f'  {name:22s} {tag}')

    # If TraceFunc not found, try increasingly lenient versions to see what *could* work
    print('\n--- Lenient TraceFunc probes (to suggest a relaxed pattern) ---')
    relaxed_variants = [
        '4C 8B DC 49 89 5B ? 49 89 6B ? 49 89 73 ?',
        '4C 8B DC 49 89 5B ? 49 89 6B ?',
        '4C 8B DC 49 89 5B',
        '4C 8B DC',
    ]
    for v in relaxed_variants:
        regex = hex_pattern_to_regex(v)
        n = len(list(regex.finditer(data)))
        print(f'  [{n:>5d} matches] {v}')


if __name__ == '__main__':
    main()
