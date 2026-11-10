"""Diagnostic only: give mapped contiguous rows the existing CPU-cache hint."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

source = Path("/work/capacity-src-20261110")
original = Path("/work/build")
build = Path("/work/capacity-build-v4-20261110")
out = Path("/work/mmap-equal-hint-20261110")
out.mkdir(exist_ok=True)
original_text = (source / "src/core/algorithm/ivf/ivf_entity.cc").read_text()
before = "if (list_data.data() && !packed_quantizer_) {"
after = "if (!packed_quantizer_ && scatter_data.spans.size() == 1 &&\n            (list_data.data() || inverted_->base_data())) {"
assert original_text.count(before) == 2
edited = original_text.replace(before, after)
path = out / "ivf_entity.cc"
path.write_text(edited)
flags = {}
for line in (original / "CMakeFiles/product_objects.dir/flags.make").read_text().splitlines():
    if " = " in line:
        k, v = line.split(" = ", 1)
        flags[k] = shlex.split(v.replace("/work/src", str(source)))
compile_command = ["/usr/bin/c++", *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"], *flags["CXX_FLAGS"],
                   "-std=c++17", "-I" + str(source / "src/core/algorithm/ivf"), "-c", str(path),
                   "-o", str(out / "ivf_entity.cc.o")]
subprocess.run(compile_command, check=True)
link = json.loads((build / "build-command.json").read_text())
old_object = str(build / "ivf_entity.cc.o")
assert link.count(old_object) == 1
link = [str(out / "ivf_entity.cc.o") if v == old_object else v for v in link]
link[link.index("-o") + 1] = str(out / "native_compare")
subprocess.run(link, cwd=original, check=True)
(out / "build-command.json").write_text(json.dumps(dict(compile=compile_command, link=link,
    original_source_sha256=hashlib.sha256(original_text.encode()).hexdigest(),
    edited_source_sha256=hashlib.sha256(edited.encode()).hexdigest(),
    scope="Only the two CPU-cache hint guards change; same current product objects and adapter. No disk prefetch or cache policy change."), indent=2) + "\n")
print(out / "native_compare")
