from __future__ import annotations

import json

from tools.train_kgalagadi_sites import manifest_sites, main


def test_manifest_sites_and_dry_run(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"site_id": site})
            for site in ("KGA:A02", "KGA:A01", "KGA:A02")
        )
        + "\n",
        encoding="utf-8",
    )
    assert manifest_sites(str(manifest)) == ["KGA:A01", "KGA:A02"]
    assert main([
        "--manifest", str(manifest),
        "--run-root", str(tmp_path / "runs"),
        "--max-sites", "1",
        "--dry-run",
    ]) == 0


def test_completed_site_does_not_consume_max_sites(tmp_path, capsys):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '\n'.join(json.dumps({"site_id": site}) for site in ("KGA:A01", "KGA:A02")) + '\n',
        encoding="utf-8",
    )
    best = tmp_path / "runs" / "A01" / "checkpoints" / "best.ckpt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"checkpoint")
    assert main([
        "--manifest", str(manifest),
        "--run-root", str(tmp_path / "runs"),
        "--max-sites", "1",
        "--dry-run",
    ]) == 0
    output = capsys.readouterr().out
    assert "[KGA:A01] skip" in output
    assert "[KGA:A02]" in output
