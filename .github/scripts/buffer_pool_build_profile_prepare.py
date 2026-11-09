"""Build bounded observers and validate offsets against the measured wheel."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

from buffer_pool_build_resources import elf_identity

BASE = '4fec4ffb39aacc50b5926f1edb30c641c5ed3985'
HERE = Path(__file__).resolve().parent

def run(args):
    return subprocess.check_output([str(a) for a in args], text=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    args = parser.parse_args()
    output = args.artifact.resolve()
    changes = run(['git', 'diff', '--name-only', BASE, '--']).splitlines()
    assert all(name.startswith('.github/') for name in changes), 'product code changed'
    cache = dict(line.split('=', 1) for line in Path('build/CMakeCache.txt').read_text().splitlines()
                 if '=' in line and not line.startswith(('#', '//')))
    compiler = next(value for key, value in cache.items() if key.startswith('CMAKE_CXX_COMPILER:'))
    saved = json.loads((Path(os.environ['ZVEC_PROFILE_SYMBOLS']) / 'location.json').read_text())
    target = Path(saved['target'])
    backup = target.with_name(target.name + '.stripped-backup')
    # The wheel has already been built. Install a diagnostic copy with exactly
    # the same final RPATH, retaining only symbols that release strip removes.
    target.rename(backup)
    try:
        shutil.copyfile(saved['saved'], target)
        subprocess.run(['cmake', '--install', 'build', '--component', 'python'],
                       env=dict(os.environ, DESTDIR=str(output / 'unstripped')), check=True)
    finally:
        target.unlink(missing_ok=True)
        backup.rename(target)
    candidates = list((output / 'unstripped').rglob('_zvec*.so'))
    assert len(candidates) == 1
    unstripped = candidates[0]
    with zipfile.ZipFile(next(output.glob('*.whl'))) as wheel:
        members = [n for n in wheel.namelist() if '/_zvec' in n and n.endswith('.so')]
        assert len(members) == 1
        measured = output / 'measured-extension.so'
        measured.write_bytes(wheel.read(members[0]))
    identity = elf_identity(measured)
    assert identity == elf_identity(unstripped), 'unstripped and wheel mappings differ'
    symbols = run(['nm', '-anC', unstripped]).splitlines()
    instances = [line for line in symbols if 'zvec::ailego::MemoryLimitPool::get_instance()::instance' in line]
    assert len(instances) == 1, 'missing/ambiguous memory pool singleton symbol'
    address = int(instances[0].split()[0], 16)
    assert any(s['flags'] & 2 and s['vaddr'] <= address < s['vaddr'] + s['memsz'] for s in identity['segments'])
    helper = output / 'layout-helper'
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wno-invalid-offsetof',
                    '-Isrc/include', '-Isrc', '-Ibuild/src', str(HERE / 'buffer_pool_build_layout.cc'),
                    '-pthread', '-o', str(helper)], check=True)
    manifest = json.loads(run([helper]))
    assert len(manifest['fields']) <= 16
    assert all(offset % 8 == 0 and offset + 8 <= manifest['pool_sizeof'] for offset in manifest['fields'].values())
    manifest.update(base_commit=BASE, diagnostic_commit=run(['git', 'rev-parse', 'HEAD']).strip(),
                    singleton_vaddr=address, elf=identity,
                    extension_sha256=hashlib.sha256(measured.read_bytes()).hexdigest(),
                    wheel_sha256=hashlib.sha256(next(output.glob('*.whl')).read_bytes()).hexdigest(),
                    workload_sha256=hashlib.sha256((HERE / 'buffer_pool_build_memory.py').read_bytes()).hexdigest())
    probe = output / 'build-wait-probe.so'
    subprocess.run([compiler, '-std=c++17', '-O2', '-fPIC', '-shared', '-fno-exceptions',
                    '-fno-rtti', '-nodefaultlibs', str(HERE / 'buffer_pool_build_wait_probe.cc'),
                    '-lc', '-ldl', '-pthread', '-lgcc', '-lgcc_s', '-o', str(probe)], check=True)
    dependencies = run(['readelf', '-d', probe])
    assert 'libstdc++' not in dependencies
    manifest['probe_sha256'] = hashlib.sha256(probe.read_bytes()).hexdigest()
    (output / 'probe-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
