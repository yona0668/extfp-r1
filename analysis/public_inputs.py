"""Local input configuration for the EXTFP public analysis copies."""
import hashlib
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# EDIT these defaults, or set the environment variables named below.
# The default local_work/ is git-ignored; do not redistribute local data or outputs.
DEFAULT_DATA_ROOT = PACKAGE_ROOT / 'local_work'
DEFAULT_INPUT_MAP = PACKAGE_ROOT / 'inputs.local.json'


def data_root():
    return Path(os.environ.get('EXTFP_DATA_ROOT', str(DEFAULT_DATA_ROOT))).resolve()


def run_identifier():
    # EDIT/set this identifier for each independently recorded execution.
    return os.environ.get('EXTFP_RUN_ID', 'local-analysis')


def read_snapshot(relative_path):
    """Read a configured reference input and verify its recorded byte hash."""
    config_path = Path(os.environ.get('EXTFP_INPUT_MAP', str(DEFAULT_INPUT_MAP))).resolve()
    if not config_path.is_file():
        raise FileNotFoundError('Copy inputs.example.json to inputs.local.json and fill in local paths and SHA-256 values.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if relative_path not in config['snapshots']:
        raise KeyError(f'Add an input mapping for {relative_path}')
    row = config['snapshots'][relative_path]
    expected = row['sha256']
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected.lower()):
        raise ValueError(f'Set the independently recorded SHA-256 for {relative_path}')
    path = Path(row['path'])
    if not path.is_absolute():
        path = data_root() / path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected.lower():
        raise ValueError(f'Input SHA-256 mismatch: {relative_path}')
    return raw.decode('utf-8').replace('\r\n', '\n')
