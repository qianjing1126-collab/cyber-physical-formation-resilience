"""Restore the losslessly compressed, checksummed experiment records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble_archive(archive: Path, expected_sha256: str) -> None:
    """Rejoin the repository's small binary parts before normal verification."""
    if archive.is_file():
        return
    index_path = ROOT / 'data/ARCHIVE_PARTS.json'
    if not index_path.is_file():
        raise SystemExit('Missing archive and archive-parts index; download the complete repository.')
    index = json.loads(index_path.read_text())
    if index['archive_sha256'] != expected_sha256:
        raise SystemExit('Archive-parts index does not match the experiment manifest.')
    parts_root = (ROOT / 'data/archive_parts').resolve()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=archive.name + '.', suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            for entry in index['parts']:
                part = (ROOT / 'data' / entry['path']).resolve()
                if not part.is_relative_to(parts_root) or not part.is_file():
                    raise SystemExit(f'Missing or unsafe archive part: {entry["path"]}')
                content = part.read_bytes()
                if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
                    raise SystemExit(f'Archive part checksum mismatch: {entry["path"]}')
                output.write(content)
        if temporary.stat().st_size != index['archive_bytes'] or digest(temporary) != expected_sha256:
            raise SystemExit('Reassembled archive checksum mismatch.')
        temporary.replace(archive)
        print(f'Reassembled and verified {archive.name} from {len(index["parts"])} parts.')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='check an existing restoration')
    parser.add_argument('--force', action='store_true', help='replace existing files that differ')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/ARCHIVE_MANIFEST.json').read_text())
    archive = ROOT / 'data' / manifest['archive']
    assemble_archive(archive, manifest['archive_sha256'])
    if digest(archive) != manifest['archive_sha256']:
        raise SystemExit('Archive SHA-256 mismatch; restore the original download.')
    expected = {entry['path']: entry for entry in manifest['files']}
    targets = {}
    for name, entry in expected.items():
        target = (ROOT / name).resolve()
        if not target.is_relative_to(ROOT / 'runs'):
            raise SystemExit(f'Unsafe archive path: {name}')
        targets[name] = target
        if target.exists() and digest(target) != entry['sha256']:
            if args.verify_only or not args.force:
                raise SystemExit(f'Existing file differs: {name}; preserve your results before using --force.')
        if args.verify_only and not target.is_file():
            raise SystemExit(f'Missing restored file: {name}')
    if not args.verify_only:
        with zipfile.ZipFile(archive) as bundle:
            if set(bundle.namelist()) != set(expected):
                raise SystemExit('Archive entries do not match the file manifest.')
            for name, entry in expected.items():
                target = targets[name]
                if target.exists() and digest(target) == entry['sha256']:
                    continue
                content = bundle.read(name)
                if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
                    raise SystemExit(f'Archived file checksum mismatch: {name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    for name, entry in expected.items():
        if digest(targets[name]) != entry['sha256']:
            raise SystemExit(f'Restored checksum mismatch: {name}')
    print(f'Verified {len(expected)} archived files under runs/.')


if __name__ == '__main__':
    main()
