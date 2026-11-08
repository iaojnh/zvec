# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Exercise Arrow's generated nested download rules without compiling Arrow."""

from __future__ import annotations

import hashlib
import http.server
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIRROR = "https://zvec-bj.oss-cn-beijing.aliyuncs.com/thirdparty/"
PRIMARY = "https://github.com/JuliaStrings/utf8proc/archive/v2.10.0.tar.gz"


class ArrowDownloadFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmake = os.environ.get("CMAKE_BIN") or shutil.which("cmake")
        if not cls.cmake:
            raise RuntimeError("CMake is required for the Arrow download regression")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as output:
            data = b"verified local fixture\n"
            entry = tarfile.TarInfo("utf8proc-fixture/payload.txt")
            entry.size = len(data)
            output.addfile(entry, io.BytesIO(data))
        cls.payload = archive.getvalue()
        cls.checksum = hashlib.sha256(cls.payload).hexdigest()
        cls.requests = []
        payload = cls.payload
        requests = cls.requests

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                if self.path.startswith("/unavailable/"):
                    self.send_error(504, "Intentional primary download failure")
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()

    def run_case(self, *, msvc: bool, policy: str):
        env = os.environ.copy()
        for name in ("ARROW_UTF8PROC_URL", "ARROW_THRIFT_URL", "ARROW_BOOST_URL"):
            env.pop(name, None)
        env.update(NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
        if policy == "explicit":
            env["ARROW_UTF8PROC_URL"] = self.url + "/custom.tar.gz"
        request_start = len(self.requests)
        with tempfile.TemporaryDirectory(prefix="arrow nested build ") as temp:
            source = Path(temp) / "parent source"
            build = Path(temp) / "parent build"
            arrow = source / "apache-arrow-21.0.0" / "cpp"
            arrow.mkdir(parents=True)
            wrapper = (ROOT / "thirdparty/arrow/CMakeLists.txt").read_text()
            # Keep the production command/list escaping; only substitute the
            # endpoints so the primary deterministically returns HTTP 504.
            wrapper = wrapper.replace(
                PRIMARY, self.url + "/unavailable/utf8proc.tar.gz"
            )
            wrapper = wrapper.replace(MIRROR, self.url + "/mirror/")
            (source / "arrow-wrapper.cmake").write_text(wrapper)
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(arrow_parent NONE)\n"
                "function(apply_patch_once)\nendfunction()\n"
                "set(CMAKE_BUILD_TYPE Release)\n"
                f"set(MSVC {'TRUE' if msvc else 'FALSE'})\n"
                "set(CMAKE_SYSTEM_PROCESSOR ARM64)\n"
                "set(ZVEC_USE_STATIC_CRT ON)\n"
                f"set(USE_OSS_MIRROR {'ON' if policy == 'mirror' else 'OFF'})\n"
                'set(EXTERNAL_BINARY_DIR "${CMAKE_BINARY_DIR}/external")\n'
                'set(EXTERNAL_INC_DIR "${EXTERNAL_BINARY_DIR}/usr/local/include")\n'
                'set(EXTERNAL_LIB_DIR "${EXTERNAL_BINARY_DIR}/usr/local/lib")\n'
                'file(MAKE_DIRECTORY "${EXTERNAL_INC_DIR}" "${EXTERNAL_LIB_DIR}")\n'
                'include("${CMAKE_CURRENT_SOURCE_DIR}/arrow-wrapper.cmake")\n'
            )
            (arrow / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(arrow_child NONE)\n"
                "if(DEFINED ENV{ARROW_UTF8PROC_URL})\n"
                ' set(urls "$ENV{ARROW_UTF8PROC_URL}")\n'
                "else()\n"
                f' set(urls "{self.url}/unavailable/utf8proc.tar.gz")\n'
                "endif()\n"
                "list(LENGTH urls count)\n"
                'file(WRITE "${CMAKE_BINARY_DIR}/url-count.txt" "${count}")\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/urls.txt" "${urls}")\n'
                "include(ExternalProject)\n"
                "ExternalProject_Add(utf8proc_ep URL ${urls}\n"
                f" URL_HASH SHA256={self.checksum}\n"
                ' CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND "")\n'
            )
            generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
            commands = [
                [self.cmake, "-S", str(source), "-B", str(build), "-G", generator],
                [self.cmake, "--build", str(build), "--target", "ARROW.BUILD"],
            ]
            for command in commands:
                result = subprocess.run(
                    command,
                    check=False,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
            child = build / "arrow/src/ARROW.BUILD-build"
            urls = (child / "urls.txt").read_text()
            count = int((child / "url-count.txt").read_text())
            requested = self.requests[request_start:]
            if policy == "default":
                assert count == 2
                assert (
                    urls
                    == self.url
                    + "/unavailable/utf8proc.tar.gz;"
                    + self.url
                    + "/mirror/utf8proc-2.10.0.tar.gz"
                )
                assert "/unavailable/utf8proc.tar.gz" in requested
                assert "/mirror/utf8proc-2.10.0.tar.gz" in requested
            elif policy == "explicit":
                assert count == 1
                assert urls == self.url + "/custom.tar.gz"
                assert requested == ["/custom.tar.gz"]
            else:
                assert count == 1
                assert requested == ["/mirror/utf8proc-2.10.0.tar.gz"]
            extracted = child / "utf8proc_ep-prefix/src/utf8proc_ep/payload.txt"
            assert extracted.read_text() == "verified local fixture\n"

    def test_default_fallback_reaches_nested_project(self):
        for msvc in (False, True):
            with self.subTest(msvc_command=msvc):
                self.run_case(msvc=msvc, policy="default")

    def test_caller_url_is_preserved(self):
        for msvc in (False, True):
            with self.subTest(msvc_command=msvc):
                self.run_case(msvc=msvc, policy="explicit")

    def test_mirror_mode_keeps_single_source(self):
        for msvc in (False, True):
            with self.subTest(msvc_command=msvc):
                self.run_case(msvc=msvc, policy="mirror")


if __name__ == "__main__":
    unittest.main()
