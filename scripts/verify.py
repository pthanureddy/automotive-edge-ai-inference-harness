from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from edge_ai.cli import build_artifacts
from edge_ai.exporting import write_manifest
from edge_ai.verification import verify_metrics


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    build_artifacts(root)
    checks = verify_metrics(root)
    run([sys.executable, "-m", "ruff", "check", "."], root)
    run([sys.executable, "-m", "pytest"], root)

    if not args.skip_build:
        build_dir = root / "build"
        build_dir.mkdir(exist_ok=True)
        if sys.platform == "win32":
            import ziglang

            zig = Path(ziglang.__file__).parent / "zig.exe"
            common = ["-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
            run(
                [
                    str(zig),
                    "cc",
                    "-std=c11",
                    *common,
                    "-Ifirmware/include",
                    "-Ifirmware/generated",
                    "-c",
                    "firmware/src/model.c",
                    "-o",
                    "build/model.o",
                ],
                root,
            )
            for target, source in (
                ("edge_ai_tests.exe", "firmware/tests/test_inference.cpp"),
                ("edge_ai_benchmark.exe", "firmware/app/benchmark.cpp"),
            ):
                run(
                    [
                        str(zig),
                        "c++",
                        "-std=c++20",
                        *common,
                        "-Ifirmware/include",
                        source,
                        "build/model.o",
                        "-o",
                        f"build/{target}",
                    ],
                    root,
                )
            run(
                [
                    str(build_dir / "edge_ai_tests.exe"),
                    str(root / "artifacts" / "reference_vectors.csv"),
                ],
                root,
            )
        else:
            run(["cmake", "-S", ".", "-B", "build", "-G", "Ninja"], root)
            run(["cmake", "--build", "build"], root)
            run(["ctest", "--test-dir", "build", "--output-on-failure"], root)
        benchmark_name = "edge_ai_benchmark.exe" if sys.platform == "win32" else "edge_ai_benchmark"
        benchmark = root / "build" / benchmark_name
        run(
            [
                str(benchmark),
                str(root / "artifacts" / "reference_vectors.csv"),
                str(root / "artifacts" / "host_benchmark.json"),
            ],
            root,
        )
        benchmark_metrics = json.loads(
            (root / "artifacts" / "host_benchmark.json").read_text(encoding="utf-8")
        )
        if benchmark_metrics["p99_us"] > benchmark_metrics["host_budget_us"]:
            raise RuntimeError("host benchmark p99 exceeded the 1 ms engineering budget")
        write_manifest(
            [
                root / "artifacts" / "metrics.json",
                root / "artifacts" / "model.onnx",
                root / "artifacts" / "reference_vectors.csv",
                root / "artifacts" / "host_benchmark.json",
                root / "firmware" / "generated" / "model_data.h",
            ],
            root / "artifacts" / "manifest.json",
            root,
        )

    print(f"verification complete: {len(checks)} metric gates passed")


if __name__ == "__main__":
    main()
