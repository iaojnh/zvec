# FP16 platform validation

Run the same two targets on desktops, simulators and physical devices:

- `hnsw_platform_test`: 72 format/metric/layout/storage combinations and one
  dispatch test. The dispatch test checks raw and quantized FP16, single and
  batch distances, SIMD boundaries, subnormal inputs and large finite inputs.
- `fp16_platform_test`: Flat and IVF × native/quantized FP16 × L2/IP/cosine,
  plus native FP16 DiskANN × L2/IP/cosine where DiskANN is supported. All cases
  check independent FP64 scores, recall and two reopens. DiskANN L2 also checks
  batched graph and exhaustive searches.

Expected totals are 73 + 15 tests, or 73 + 12 when DiskANN is unsupported
(Intel macOS, Windows ARM64 and RISC-V). Unsupported DiskANN is recorded as a
platform capability, rather than reported as a skipped test.

## Desktop and RISC-V

From an already configured build:

```sh
cmake --build build --target hnsw_platform_test fp16_platform_test --parallel
ctest --test-dir build -R '^(hnsw|fp16)_platform_test$' --output-on-failure
python .github/scripts/hnsw_test_runner.py verify-build build
```

The checker rejects missing named cases, failures, skips and missing CPU
coverage. It prints and writes `build/test_tmp/fp16-coverage.json`.
Use `--expect-arch SSE2`, `NEON` or `Scalar` for the platform baseline.

On suitably equipped machines, require specific executed kernel paths:

```sh
python .github/scripts/hnsw_test_runner.py verify-build build --require-arch AVX2 --require-arch AVX512
python .github/scripts/hnsw_test_runner.py verify-build build --require-arch NEON_BASELINE --require-arch NEON_FP16
```

`NEON_FP16` records selection of the compiled native kernel, not just a CPU
feature flag. Missing required paths fail verification. CPU-dependent paths
still need suitable hardware. The RISC-V workflow is separate from the main
pipeline: manually run `07-linux-riscv-build.yml` on the tested branch.

## Android ARM64 physical device

Set `ANDROID_NDK_HOME`, then build self-contained SDK tests:

```sh
cmake -S . -B build_fp16_android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=29 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_ZVEC_SHARED=OFF -DBUILD_ZVEC_CORE_SHARED=OFF \
  -DBUILD_ZVEC_AILEGO_SHARED=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_C_BINDINGS=OFF \
  -DBUILD_TOOLS=OFF -DENABLE_NATIVE=OFF -DAUTO_DETECT_ARCH=OFF -DENABLE_WERROR=ON
cmake --build build_fp16_android --target hnsw_platform_test fp16_platform_test --parallel
python .github/scripts/hnsw_test_runner.py android --device SERIAL --build-dir build_fp16_android
```

The runner requires a physical ARM64 device, runs both suites and retains
separate XML reports and logs. Use an unlocked device with authorized USB
debugging. Append `--require-arch NEON_FP16` when validating that hardware path.

## iOS physical device

On macOS, install an Apple Development identity and provisioning profile that
covers the device and `$BUNDLE_ID`. Set `$TEAM_ID` and `$PROFILE_NAME` too.
Both apps may use the same bundle ID because the runner installs and executes
them sequentially.

```sh
cmake -S . -B build_fp16_ios -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS -DIOS=ON -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_SYSROOT=iphoneos -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DBUILD_ZVEC_SHARED=OFF -DBUILD_ZVEC_CORE_SHARED=OFF -DBUILD_ZVEC_AILEGO_SHARED=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_C_BINDINGS=OFF -DBUILD_TOOLS=OFF -DENABLE_WERROR=ON \
  -DZVEC_HNSW_TEST_BUNDLE_ID="$BUNDLE_ID" -DZVEC_FP16_TEST_BUNDLE_ID="$BUNDLE_ID" \
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_STYLE=Manual \
  -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="$TEAM_ID" \
  -DCMAKE_XCODE_ATTRIBUTE_PROVISIONING_PROFILE_SPECIFIER="$PROFILE_NAME" \
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY="Apple Development"
cmake --build build_fp16_ios --target hnsw_platform_test fp16_platform_test --config Release --parallel
python .github/scripts/hnsw_test_runner.py ios --device UDID --build-dir build_fp16_ios
```

The device must be unlocked and trusted, with Developer Mode enabled. The
runner installs both signed apps using `devicectl`, verifies every named test
and retains console logs under `build/hnsw-device-results/` by default.
Simulator success and cross-compilation do not establish physical-device
correctness; record those runs separately when devices become available.
