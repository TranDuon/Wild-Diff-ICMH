"""CPU-only behavior checks for the permanent Colab helper launcher."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def notebook_builder():
    sys.path.insert(0, str(ROOT / "tools"))
    return importlib.import_module("build_colab_phase3_notebook")


def p3_definition():
    """Extract only p3(), never execute preparation or touch Drive/GPU."""
    source = next(cell["source"] for cell in notebook_builder().build()["cells"]
                  if cell["cell_type"] == "code" and "def p3(" in cell["source"])
    return next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "p3")


def launcher(exit_code=0):
    events = []
    launches = []

    class Process:
        def __init__(self, command, **kwargs):
            launches.append((command, kwargs))
            self.stdout = iter(("stdout first\n", "stderr second\n"))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def wait(self):
            events.append(("wait", exit_code))
            return exit_code

    def output(value, **kwargs):
        events.append(("print", value, kwargs))

    def legacy_run(command, **kwargs):
        launches.append((command, kwargs))
        if kwargs.get("check") and exit_code:
            raise subprocess.CalledProcessError(exit_code, command)
        return SimpleNamespace(returncode=exit_code)

    api = SimpleNamespace(
        Popen=Process, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
        CalledProcessError=subprocess.CalledProcessError,
        # Keep the pre-fix launcher runnable: RED must fail on behavior,
        # not on a missing stub or an unrelated fixture exception.
        run=legacy_run,
    )
    repo = Path("/content/Wild-Diff-ICMH")
    namespace = {"subprocess": api, "sys": SimpleNamespace(executable="python"),
                 "DRIVE_ROOT": Path("/content/drive/MyDrive/wild_diff_icmh"),
                 "CU_RATE": 1.54, "REPO": repo, "print": output}
    exec(compile(ast.Module(body=[p3_definition()], type_ignores=[]), "notebook-p3", "exec"), namespace)
    return namespace["p3"], events, launches, repo


def test_p3_streams_and_flushes_merged_output_before_waiting():
    p3, events, launches, repo = launcher()
    p3("train", "--target-steps", 1000)
    assert events == [
        ("print", "stdout first\n", {"end": "", "flush": True}),
        ("print", "stderr second\n", {"end": "", "flush": True}),
        ("wait", 0),
    ]
    command, kwargs = launches[0]
    assert command == ["python", "-u", "tools/phase3_h1.py", "train", "--drive-root",
                       str(Path("/content/drive/MyDrive/wild_diff_icmh")), "--cu-rate", "1.54",
                       "--target-steps", "1000"]
    assert kwargs == {"cwd": repo, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                      "text": True, "bufsize": 1}


def test_p3_streams_failure_details_and_preserves_nonzero_exit():
    p3, events, launches, _ = launcher(exit_code=7)
    with pytest.raises(subprocess.CalledProcessError) as exc:
        p3("evaluate", "--scope", "quick")
    assert exc.value.returncode == 7 and exc.value.cmd == launches[0][0]
    assert [event[1] for event in events if event[0] == "print"] == [
        "stdout first\n", "stderr second\n"]
    assert events[-1] == ("wait", 7)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_p3_real_cpu_child_streams_stdout_stderr_and_exit_code(tmp_path, capsys, exit_code):
    script = tmp_path / "tools" / "phase3_h1.py"
    script.parent.mkdir()
    script.write_text(
        "import sys\nprint('stdout from real child', flush=True)\n"
        "print('stderr from real child', file=sys.stderr, flush=True)\n"
        f"raise SystemExit({exit_code})\n", encoding="utf-8")
    namespace = {"subprocess": subprocess, "sys": sys, "DRIVE_ROOT": tmp_path,
                 "CU_RATE": 1.54, "REPO": tmp_path}
    exec(compile(ast.Module(body=[p3_definition()], type_ignores=[]), "notebook-p3", "exec"), namespace)
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError) as exc:
            namespace["p3"]("prepare")
        assert exc.value.returncode == exit_code
    else:
        namespace["p3"]("prepare")
    captured = capsys.readouterr()
    # Sandboxed Python launchers may print diagnostics before the child starts.
    assert captured.out.endswith("stdout from real child\nstderr from real child\n")
    assert captured.err == ""


def test_current_notebook_matches_builder_compiles_and_keeps_safe_defaults():
    builder = notebook_builder()
    notebook = json.loads((ROOT / "Wild_Diff_ICMH_Phase3_H1.ipynb").read_text(encoding="utf-8"))
    assert notebook == builder.build()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"cell-{index}", "exec")
    source = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "SESSION_HOURS = 1.5" in source
    for name in ("RUN_EXTENSION", "RUN_FULL_DEV", "DISCONNECT_NOW"):
        assert f"{name} = False" in source
    assert "git', 'pull', '--ff-only'" in source
    assert "local_changes" in source and "after == remote" in source
