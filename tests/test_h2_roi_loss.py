"""Exercise the actual ROI reducers AND optimization branch without SD weights."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def production_model(alpha=2, enabled=True):
    tree = ast.parse((ROOT / "model/diffeic.py").read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DiffEIC")
    names = {"_weighted_spatial_mean", "_region_spatial_mean", "p_losses", "training_step"}
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    # Compile original function bodies, not a copy of their mathematical formula.
    module = ast.Module(body=[ast.ClassDef(name="ProductionSlice", bases=[], keywords=[],
                                        body=methods, decorator_list=[])], type_ignores=[])
    namespace = {"torch": torch, "F": F, "default": lambda value, fallback: value if value is not None else fallback()}
    exec(compile(ast.fix_missing_locations(module), "model/diffeic.py:ROI-production-slice", "exec"), namespace)
    model = namespace["ProductionSlice"]()
    model.roi_weight, model.roi_background_weight = alpha, 1
    model.roi_loss_enabled, model.roi_semantic_enabled = enabled, False
    model.training, model.parameterization = True, "eps"
    model.logvar, model.device, model.learn_logvar = torch.zeros(1), "cpu", False
    model.l_simple_weight, model.l_bpp_weight, model.l_guide_weight = 1, 2, 3
    model.l_semantic_weight, model.use_scheduler, model.ucg_training = 0, False, {}
    model.q_sample = lambda x_start, t, noise: x_start
    model.apply_model = lambda x, t, cond: torch.zeros_like(x)
    model.get_loss = lambda output, target, mean=False: (output - target).square()
    model.preprocess_model = SimpleNamespace(aux_loss=lambda: torch.tensor(.5))
    model.log_dict = lambda values, **kwargs: setattr(model, "logged", values)
    return model


def test_alpha1_is_uniform_for_multichannel_and_batched_loss():
    model = production_model(alpha=1)
    loss_map = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :4, :4] = 1
    assert torch.allclose(model._weighted_spatial_mean(loss_map, mask), loss_map.mean())
    assert torch.allclose(model._weighted_spatial_mean(loss_map, None), loss_map.mean())


def test_alpha2_changes_actual_total_optimization_gradient():
    mask = torch.tensor([[[[1., 0.], [0., 0.]]]])
    gradients, totals = [], []
    for alpha, enabled in ((1, False), (2, True)):
        model = production_model(alpha, enabled)
        latent = torch.tensor([[[[2., 1.], [1., 1.]]]], requires_grad=True)
        original = torch.zeros_like(latent)
        cond = {"c_latent": [latent], "roi_mask": mask,
                "bpp": torch.tensor(0.), "q_bpp": torch.tensor(0.),
                "sem_bpp": torch.tensor(0.), "tag_bpp": torch.tensor(0.)}
        model.shared_step = lambda batch: model.p_losses(original, cond, torch.tensor([0]), noise=original)
        total = model.training_step({}, 0)
        assert torch.allclose(total, model.logged["T/loss"] + model.logged["T/aux_loss"])
        assert torch.allclose(total, model.logged["T/optim_loss"])
        total.backward()
        gradients.append(latent.grad.clone())
        totals.append(total.detach())
        if enabled:
            assert set(("T/l_guide_roi", "T/l_guide_bg", "T/roi_fraction")) <= model.logged.keys()
            assert model.logged["T/roi_fraction"] == .25
    uniform_ratio = gradients[0][0, 0, 0, 0] / gradients[0][0, 0, 0, 1]
    roi_ratio = gradients[1][0, 0, 0, 0] / gradients[1][0, 0, 0, 1]
    assert torch.allclose(roi_ratio, 2 * uniform_ratio)
    assert not torch.allclose(totals[0], totals[1])


@pytest.mark.parametrize("kind", ["empty", "full", "tiny", "downsample", "out_of_range"])
def test_roi_edge_cases_have_finite_loss_regions_and_gradient(kind):
    model = production_model()
    loss_map = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).requires_grad_()
    mask = torch.zeros(1, 1, 16, 16)
    if kind == "full":
        mask.fill_(1)
    elif kind == "tiny":
        mask[0, 0, 7, 7] = 1  # May disappear under nearest-neighbor downsampling.
    elif kind == "downsample":
        mask[:, :, :8, :8] = 1
    elif kind == "out_of_range":
        mask[:, :, :8, :8] = 2
        mask[:, :, 8:, :] = -1
    loss = model._weighted_spatial_mean(loss_map, mask)
    assert torch.isfinite(loss)
    assert torch.isfinite(model._region_spatial_mean(loss_map, mask, True))
    assert torch.isfinite(model._region_spatial_mean(loss_map, mask, False))
    loss.backward()
    assert torch.isfinite(loss_map.grad).all()
    if kind in {"empty", "full", "tiny"}:
        assert torch.allclose(loss, loss_map.mean())
