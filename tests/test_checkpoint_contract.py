"""Focused tests for published-checkpoint compatibility contracts."""

from __future__ import annotations

import unittest
import ast
import tempfile
from pathlib import Path

from utils.checkpoint_contract import (
    PROJECT_CHECKPOINT_CONTRACT_VERSION,
    migrate_legacy_entropy_bottleneck_checkpoint,
    resolve_run_model_config,
    validate_entropy_bottleneck_checkpoint,
    validate_project_resume_checkpoint,
)


PREFIX = "preprocess_model.entropy_bottleneck"


def current_entropy_keys() -> list[str]:
    return (
        [f"{PREFIX}.matrices.{index}" for index in range(5)]
        + [f"{PREFIX}.biases.{index}" for index in range(5)]
        + [f"{PREFIX}.factors.{index}" for index in range(4)]
    )


def legacy_entropy_state() -> dict[str, object]:
    state: dict[str, object] = {"unrelated.weight": object()}
    for kind, count in (("matrix", 5), ("bias", 5), ("factor", 4)):
        for index in range(count):
            state[f"{PREFIX}._{kind}{index}"] = object()
    return state


class ModelStub:
    def state_dict(self):
        return {key: object() for key in current_entropy_keys()}


class EntropyCheckpointMigrationTests(unittest.TestCase):
    def test_migrates_complete_legacy_layout_and_preserves_checkpoint_metadata(self):
        legacy_state = legacy_entropy_state()
        unrelated = legacy_state["unrelated.weight"]
        checkpoint = {"state_dict": legacy_state, "global_step": 19}

        migrated, migrations = migrate_legacy_entropy_bottleneck_checkpoint(checkpoint)

        self.assertEqual(len(migrations), 14)
        self.assertEqual(migrated["global_step"], 19)
        self.assertIs(migrated["state_dict"]["unrelated.weight"], unrelated)
        self.assertTrue(set(current_entropy_keys()).issubset(migrated["state_dict"]))
        self.assertFalse(
            any("._matrix" in key or "._bias" in key or "._factor" in key
                for key in migrated["state_dict"])
        )
        validate_entropy_bottleneck_checkpoint(ModelStub(), migrated)

        # The migration is intentionally non-mutating because callers may
        # still need the raw checkpoint for diagnostics.
        self.assertIn(f"{PREFIX}._matrix0", checkpoint["state_dict"])

    def test_rejects_legacy_and_current_key_collision(self):
        state = legacy_entropy_state()
        state[f"{PREFIX}.matrices.0"] = object()

        with self.assertRaisesRegex(ValueError, "key collision"):
            migrate_legacy_entropy_bottleneck_checkpoint({"state_dict": state})

    def test_rejects_incomplete_parameter_list_after_migration(self):
        state = legacy_entropy_state()
        del state[f"{PREFIX}._factor3"]
        migrated, _ = migrate_legacy_entropy_bottleneck_checkpoint(
            {"state_dict": state}
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Incomplete entropy-bottleneck.*factors\.3",
        ):
            validate_entropy_bottleneck_checkpoint(ModelStub(), migrated)

    def test_training_migrates_and_validates_before_shape_check_and_load(self):
        source = open("train.py", encoding="utf-8").read()
        migration = source.index("migrate_legacy_entropy_bottleneck_checkpoint(", source.index("def main"))
        validation = source.index("validate_entropy_bottleneck_checkpoint(model, checkpoint)")
        shape_check = source.index("state_dict_shape_mismatches(model, checkpoint)")
        loading = source.index("load_state_dict(model, checkpoint, strict=False)")

        self.assertLess(migration, validation)
        self.assertLess(validation, shape_check)
        self.assertLess(shape_check, loading)

    def test_inference_migrates_and_validates_before_shape_check_and_load(self):
        source = open("inference_partition.py", encoding="utf-8").read()
        function = source[source.index("def _load_checkpoint"):source.index("def process")]
        migration = function.index("migrate_legacy_entropy_bottleneck_checkpoint")
        validation = function.index("validate_entropy_bottleneck_checkpoint")
        shape_check = function.index("state_dict_shape_mismatches")
        loading = function.index("load_state_dict")

        self.assertLess(migration, validation)
        self.assertLess(validation, shape_check)
        self.assertLess(shape_check, loading)

    def test_project_resume_requires_current_contract_and_optimizer_state(self):
        valid = {
            "wild_diff_checkpoint_contract_version": PROJECT_CHECKPOINT_CONTRACT_VERSION,
            "global_step": 20,
            "optimizer_states": [{"state": {}}],
        }
        validate_project_resume_checkpoint(valid)

        with self.assertRaisesRegex(RuntimeError, "predates.*migration fix"):
            validate_project_resume_checkpoint({
                "global_step": 20,
                "optimizer_states": [{"state": {}}],
            })
        with self.assertRaisesRegex(RuntimeError, "optimizer_states"):
            validate_project_resume_checkpoint({
                "wild_diff_checkpoint_contract_version": PROJECT_CHECKPOINT_CONTRACT_VERSION,
                "global_step": 20,
            })

    def test_short_training_run_saves_final_full_state_checkpoint(self):
        source = Path("train.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_save_final_checkpoint"
        )
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "train.py", "exec"), namespace)

        class TrainerStub:
            def __init__(self):
                self.weights_only = None

            def save_checkpoint(self, path, weights_only):
                self.weights_only = weights_only
                Path(path).write_bytes(b"full-state")

        with tempfile.TemporaryDirectory() as temporary:
            trainer = TrainerStub()
            output = namespace["_save_final_checkpoint"](trainer, temporary)
            self.assertEqual(Path(output).read_bytes(), b"full-state")
            self.assertFalse(trainer.weights_only)

        fit = source.index("trainer.fit(")
        final_save = source.index("_save_final_checkpoint(trainer, save_dir)")
        self.assertLess(fit, final_save)

    def test_inference_prefers_model_config_saved_with_project_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "runs" / "h1_v2" / "A01"
            checkpoint = run_dir / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            saved_config = run_dir / "config_model.yaml"
            saved_config.write_text("control_model_ratio: 1.0\n", encoding="utf-8")

            self.assertEqual(
                resolve_run_model_config("configs/model/diffeic.yaml", str(checkpoint)),
                str(saved_config),
            )

            saved_config.unlink()
            self.assertEqual(
                resolve_run_model_config("configs/model/diffeic.yaml", str(checkpoint)),
                "configs/model/diffeic.yaml",
            )


if __name__ == "__main__":
    unittest.main()
