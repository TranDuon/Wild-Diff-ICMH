---
status: resolved
trigger: "Step 9 decode fails while constructing DDIMSampler because DiffEIC has no used_timesteps attribute."
created: 2026-09-27
updated: 2026-09-27
---

## Symptoms

- Expected: Step 9 decodes the two-image smoke-test partition with DDIM.
- Actual: decode exits immediately when `DDIMSampler(model)` is constructed.
- Error: `AttributeError: 'DiffEIC' object has no attribute 'used_timesteps'. Did you mean: 'num_timesteps'?`

## Current Focus

hypothesis: The custom DDIM sampler accidentally reads a SpacedSampler-local name instead of the latent-diffusion model schedule contract.
test: Instantiate the custom DDIM sampler with a minimal model exposing only `num_timesteps`.
expecting: The sampler records the model's complete diffusion schedule length without requiring `used_timesteps`.
next_action: Pull the fix in Colab and rerun Step 9 only.

## Evidence

- timestamp: 2026-09-27T15:53:00+07:00
  finding: `DiffEIC` inherits `DDPM.register_schedule`, which sets `self.num_timesteps`.
  implication: `num_timesteps` is the model's authoritative diffusion schedule length.
- timestamp: 2026-09-27T15:54:00+07:00
  finding: The repository's canonical `ldm.models.diffusion.ddim.DDIMSampler` and PLMS sampler both read `model.num_timesteps`.
  implication: The custom sampler's `model.used_timesteps` reference is an isolated typo.
- timestamp: 2026-09-27T15:55:00+07:00
  finding: `used_timesteps` exists only as a local variable inside `model/spaced_sampler.py`.
  implication: It is not and should not be a `DiffEIC` attribute.

## Eliminated

- hypothesis: The project checkpoint is missing a serialized schedule attribute.
  evidence: Schedule length is configured and registered during model construction, not loaded as a checkpoint field.
- hypothesis: The DDIM smoke-test step count should replace the model diffusion schedule length.
  evidence: `--steps 5` controls the reduced DDIM schedule generated from the full `num_timesteps` schedule.

## Resolution

root_cause: `model/ddim_sampler.py` referenced nonexistent `model.used_timesteps` instead of the standard `model.num_timesteps` attribute.
fix: Read `model.num_timesteps` and add a regression test for the sampler/model contract.
verification: `python -m pytest -q` passes with 85 passed and 2 skipped; the focused DDIM/Phase-1 tests pass 5/5.
files_changed:
  - model/ddim_sampler.py
  - tests/test_ddim_sampler_contract.py

