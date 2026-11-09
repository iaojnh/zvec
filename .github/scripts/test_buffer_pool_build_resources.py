"""Safety checks for exact ELF matching, counter reads and stage subtraction."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from buffer_pool_build_resources import MAGIC, elf_identity, read_budget, psi, log_stage
from buffer_pool_build_profile_report import delta

class ResourceTests(unittest.TestCase):
    def test_strip_metadata_can_change_but_loadable_code_cannot(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'module.so'
            data = bytearray(256)
            ident = b'\x7fELF\x02\x01' + b'\0' * 10
            struct.pack_into('<16sHHIQQQIHHHHHH', data, 0, ident, 3, 62, 1, 0, 64, 200, 0, 64, 56, 1, 64, 1, 0)
            struct.pack_into('<IIQQQQQQ', data, 64, 1, 5, 0, 0, 0, 192, 192, 4096)
            p.write_bytes(data)
            before = elf_identity(p)
            struct.pack_into('<Q', data, 40, 216)
            struct.pack_into('<HH', data, 60, 0, 0)
            p.write_bytes(data)
            self.assertEqual(before, elf_identity(p))
            data[160] = 1
            p.write_bytes(data)
            self.assertNotEqual(before, elf_identity(p))

    def test_wait_and_native_fields_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'shared'
            data = bytearray(4096)
            values = [MAGIC, 2, 1000, 7, 6, 900, 2, 1, 400, 512 << 20, 12] + [0] * 14
            struct.pack_into('<25Q', data, 0, *values)
            p.write_bytes(data)
            result = read_budget(p, ['pool_size', 'capacity_waits'])
            self.assertEqual(result['cv_wait_calls'], 7)
            self.assertEqual(result['cv_wait_active'], 1)
            self.assertEqual(result['capacity_waits'], 12)

    def test_torn_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'shared'
            p.write_bytes(struct.pack('<25Q', MAGIC, 3, *([0] * 23)) + bytes(3896))
            with self.assertRaises(RuntimeError):
                read_budget(p, [])

    def test_psi_preserves_full_and_some(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'io.pressure'
            p.write_text('some avg10=0.00 avg60=0.00 avg300=0.00 total=123\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=40\n')
            self.assertEqual(psi(p), {'some_total_us': 123, 'full_total_us': 40})

    def test_missing_metric_is_not_reported_as_zero(self):
        self.assertIsNone(delta({}, {}, 'budget', 'capacity_waits'))
        with self.assertRaises(AssertionError):
            delta({'v': 2}, {'v': 1}, 'v')

    def test_graph_passes_and_dump_are_distinct_stages(self):
        self.assertEqual(log_stage('[ INFO ] DiskANN graph build pass 1/2'), 'graph_pass_1')
        self.assertEqual(log_stage('[ INFO ] DiskANN graph build pass 2/2'), 'graph_pass_2')
        self.assertEqual(log_stage('Begin DiskAnnBuilder::dump'), 'index_dump')
        self.assertIsNone(log_stage('Built cnt 4, finished percent 0.001%'))

if __name__ == '__main__':
    unittest.main()
