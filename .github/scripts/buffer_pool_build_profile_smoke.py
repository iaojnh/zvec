"""Check the exact wheel/layout in a fresh process before expensive runs."""
import json
from pathlib import Path
import sys
import time

from buffer_pool_build_profile import configure_probe
from buffer_pool_build_resources import read_budget
import zvec

manifest = json.loads(Path(sys.argv[1]).read_text())
folder = Path(sys.argv[2]).resolve()
folder.mkdir(exist_ok=False)
(folder / 'build.jsonl').touch()
zvec.init(memory_limit_mb=512, query_threads=1, optimize_threads=4)
probe = configure_probe(manifest, folder, 512)
time.sleep(0.2)
state = json.loads((folder / 'budget-state.json').read_text())
row = read_budget(folder / 'budget.shared', state['field_names'])
assert 0 < row['pool_size'] <= 512 << 20
assert row['used'] <= row['pool_size'] and row['committed'] <= row['pool_size']
probe.zvec_build_probe_stop()
print('Actual wheel and native budget layout verified:', json.dumps(row))
