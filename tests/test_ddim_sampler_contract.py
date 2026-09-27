"""Regression coverage for the custom DDIM sampler/model contract."""
import ast
from pathlib import Path


def test_ddim_sampler_uses_model_diffusion_schedule_length():
    source = (
        Path(__file__).resolve().parents[1] / "model" / "ddim_sampler.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    sampler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DDIMSampler"
    )
    constructor = next(
        node
        for node in sampler.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    schedule_assignment = next(
        node
        for node in constructor.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "ddpm_num_timesteps"
            for target in node.targets
        )
    )

    assert ast.unparse(schedule_assignment.value) == "model.num_timesteps"

