"""Extract marimo v1 session snapshots from already executed HTML exports.

Example:
  .venv/bin/python scripts/export_notebook_sessions.py \
    --source-dir examples --output-dir examples/__marimo__/session notebook_exports/[0-9][0-9]_*.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

from marimo._session.state.serialize import deserialize_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", nargs="+", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        help="Explicit source for one differently named HTML export",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.source is not None and len(args.html) != 1:
        parser.error("--source requires exactly one HTML file")
    if args.source is None and args.source_dir is None:
        parser.error("Supply --source or --source-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for html_path in args.html:
        document = html_path.read_text(encoding="utf-8")
        mount = document.find("window.__MARIMO_MOUNT_CONFIG__")
        marker = '"session":'
        start = document.find(marker, mount)
        if mount < 0 or start < 0:
            raise ValueError(f"No embedded session in {html_path}")
        session, _ = json.JSONDecoder().raw_decode(
            document[start + len(marker) :].lstrip()
        )
        if session.get("version") != "1":
            raise ValueError("Unsupported session schema version")
        source = (
            args.source
            if args.source is not None
            else args.source_dir / (html_path.stem + ".py")
        )
        app = runpy.run_path(
            str(source.resolve()), run_name="session_snapshot_validation"
        )["app"]
        cells = list(app._cell_manager.valid_cells())
        hashes = [
            hashlib.md5(cell._cell.code.encode(), usedforsecurity=False).hexdigest()
            for _, cell in cells
        ]
        if hashes != [cell["code_hash"] for cell in session["cells"]]:
            raise ValueError(f"Exported outputs are stale relative to {source}")
        for cell in session["cells"]:
            if set(cell) != {"id", "code_hash", "outputs", "console"}:
                raise ValueError(f"Unexpected cell schema in {source}")
            if any(
                output.get("type") != "data" or not isinstance(output.get("data"), dict)
                for output in cell["outputs"]
            ):
                raise ValueError(f"Unsupported or failed cell output in {source}")
        view = deserialize_session(
            session,
            {
                code_hash: cell_id
                for code_hash, (cell_id, _) in zip(hashes, cells, strict=True)
            },
        )
        if len(view.cell_notifications) != len(cells):
            raise ValueError(f"Not all outputs restored for {source}")
        output = args.output_dir / (source.name + ".json")
        payload = (
            json.dumps(session, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode()
        output.write_bytes(payload)
        reports.append(
            {
                "path": str(output.resolve()),
                "bytes": len(payload),
                "cells": len(cells),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "session_sha256": hashlib.sha256(payload).hexdigest(),
                "schema_version": session["version"],
                "marimo_version": session["metadata"]["marimo_version"],
                "source_hashes_match": True,
                "error_outputs": 0,
            }
        )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
