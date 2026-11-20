"""Check the review notebooks and execute their fast defaults into HTML.

Run from the checkout with ``uv run --all-extras python scripts/check_notebooks.py``.
Numerical assertions are independently covered by the marked notebook tests.
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import socket
import subprocess
import sys
from importlib.metadata import version
from importlib.resources import as_file, files
from pathlib import Path


def _check_local_ipc() -> None:
    """Fail promptly where an execution sandbox blocks asyncio wakeup sockets."""
    try:
        sender, receiver = socket.socketpair()
        with sender, receiver:
            receiver.settimeout(1.0)
            sender.sendall(b"notebook-runtime")
            if receiver.recv(32) != b"notebook-runtime":
                msg = "The notebook runtime could not exchange a local socket message."
                raise SystemExit(msg)
    except OSError as error:
        msg = (
            "The notebook runtime requires local socket communication for asyncio. "
            "This environment blocks it; enable local IPC/network permission for "
            "the check command and retry. No internet access is needed by the "
            "default notebook calculations."
        )
        raise SystemExit(msg) from error


def _environment(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    defaults = {
        "XDG_CONFIG_HOME": str(root / ".cache" / "notebook-config"),
        "XDG_CACHE_HOME": str(root / ".cache"),
        "MPLCONFIGDIR": str(root / ".cache" / "matplotlib"),
        "MPLBACKEND": "Agg",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    for name, value in defaults.items():
        env.setdefault(name, value)
    env["MARIMO_SKIP_UPDATE_CHECK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    (Path(env["XDG_CACHE_HOME"]) / "fontconfig").mkdir(parents=True, exist_ok=True)
    return env


def _run(
    arguments: list[str],
    *,
    root: Path,
    env: dict[str, str],
    timeout: float,
) -> None:
    try:
        subprocess.run(arguments, cwd=root, env=env, check=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        msg = (
            f"Notebook command exceeded its {timeout:g}s budget: {' '.join(arguments)}"
        )
        raise SystemExit(msg) from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


def _write_index(output_dir: Path, notebooks: list[Path]) -> None:
    titles = {
        "01_graph_regression": "Graph regression and borrowing strength",
        "02_mesothelioma": "Mesothelioma classification",
        "03_house_prices": "Spatial house-price regression",
        "04_solver_diagnostics": "Solver diagnostics and reusable work",
        "05_streaming": "Streaming and explicit composition",
        "06_workflow": "Tuning, sklearn, and fitted artifacts",
    }
    items = "\n".join(
        f'<li><a href="{html.escape(path.stem)}.html">'
        f"{html.escape(titles.get(path.stem, path.stem.replace('_', ' ')))}</a></li>"
        for path in notebooks
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stratified Models · notebook review</title>
<style>
body {{ margin: 4rem auto; max-width: 54rem; padding: 0 1.5rem;
font: 1.1rem/1.6 system-ui, sans-serif; color: #173047; background: #fbfcfd; }}
h1 {{ line-height: 1.15; font-size: 2.6rem; }}
li {{ margin: 1rem 0; }} a {{ color: #146f83; text-underline-offset: .2em; }}
.eyebrow {{ text-transform: uppercase; font-size: .8rem; letter-spacing: .12em; }}
</style></head><body>
<p class="eyebrow">Stratified Models</p>
<h1>Six experiments to explore and review</h1>
<p>Rendered quick runs with fixed seeds, measured results and solver diagnostics.
Open the corresponding Python notebooks in marimo to change controls and run
the larger experiments.</p>
<ol>{items}</ol>
<p>Keep this folder together when copying the previews; their display assets
are bundled locally.</p>
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _bundle_assets(output_dir: Path, notebooks: list[Path]) -> None:
    """Use the installed frontend so rendered previews work without a CDN."""
    resources = files("marimo").joinpath("_static")
    with as_file(resources) as static_directory:
        shutil.copytree(
            static_directory,
            output_dir / "_marimo",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("index.html"),
        )
    prefix = (
        f"https://cdn.jsdelivr.net/npm/@marimo-team/frontend@{version('marimo')}/dist/"
    )
    for notebook in notebooks:
        path = output_dir / f"{notebook.stem}.html"
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace(prefix, "_marimo/"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("notebook_exports"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    root = Path(__file__).resolve().parents[1]
    notebooks = args.notebooks or sorted((root / "examples").glob("[0-9][0-9]_*.py"))
    if not notebooks:
        parser.error("No review notebooks were found")
    notebooks = [path.resolve() for path in notebooks]
    for path in notebooks:
        if not path.is_file():
            parser.error(f"Notebook does not exist: {path}")

    _check_local_ipc()
    env = _environment(root)
    marimo = [sys.executable, "-m", "marimo"]
    print(f"Checking {len(notebooks)} notebook(s)", flush=True)
    _run(
        [*marimo, "check", "--strict", *(str(path) for path in notebooks)],
        root=root,
        env=env,
        timeout=args.timeout,
    )
    if args.check_only:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in notebooks:
        output = output_dir / f"{path.stem}.html"
        print(f"Executing and exporting {path.name}", flush=True)
        _run(
            [
                *marimo,
                "export",
                "html",
                str(path),
                "-o",
                str(output),
                "--no-include-code",
                "--force",
            ],
            root=root,
            env=env,
            timeout=args.timeout,
        )
    _bundle_assets(output_dir, notebooks)
    _write_index(output_dir, notebooks)
    print(f"Rendered notebooks: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
