from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile


def build():
    import pybind11
    directory = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="active-passage-smac-build-") as temporary:
        command = ["cmake", "-S", str(directory / "native"), "-B", temporary,
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_PREFIX_PATH=/opt/ros/humble",
            "-DPython3_EXECUTABLE=/usr/bin/python3", f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={directory}"]
        for arguments in (command, ["cmake", "--build", temporary, "--parallel", "2"]):
            subprocess.run(["bash", "-c", 'source /opt/ros/humble/setup.bash && exec "$@"',
                            "active-passage-build", *arguments], check=True)


if __name__ == "__main__":
    build()
