#!/usr/bin/env python3
"""Retain diagnostic symbols before the normal, unchanged release strip."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

directory = Path(os.environ['ZVEC_PROFILE_SYMBOLS'])
directory.mkdir(parents=True, exist_ok=True)
for argument in sys.argv[1:]:
    target = Path(argument)
    if target.name.startswith('_zvec') and target.is_file():
        symbols = subprocess.run(['nm', '-anC', str(target)], text=True, capture_output=True)
        if 'MemoryLimitPool::get_instance()::instance' in symbols.stdout:
            copy = directory / (target.name + '.unstripped')
            shutil.copyfile(target, copy)
            (directory / 'location.json').write_text(json.dumps(dict(target=str(target.resolve()), saved=str(copy.resolve()))))
os.execv('/usr/bin/strip', ['/usr/bin/strip', *sys.argv[1:]])
