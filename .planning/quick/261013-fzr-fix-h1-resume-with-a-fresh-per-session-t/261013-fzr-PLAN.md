---
quick: 261013-fzr
phase: quick
plan: 01
type: execute
wave: 1
depends_on: []
autonomous: true
files_modified:
  - train.py
  - utils/session_timer.py
  - tools/phase3_h1.py
  - tools/build_colab_phase3_notebook.py
  - Wild_Diff_ICMH_Phase3_H1.ipynb
  - tests/test_session_timer.py
  - tests/test_phase3_h1.py
  - COLAB_TRAINING.md
must_haves:
  truths:
    - An opt-in Phase 3 resume starts a fresh training time allowance while restoring weights, optimizer, scheduler, loops and global step.
    - Resuming the existing step-862 run requires neither editing its checkpoint nor changing its frozen experiment contract.
    - Legacy training without the session flag retains the existing cumulative Lightning Timer behavior.
    - Colab displays subprocess progress, preparation and checkpoint staging instead of only dependency setup output.
    - GPU experiments, Drive files, test selection, protocol and expensive-option defaults are not changed by this quick fix.
---

<objective>
Fix the H1 resume time allowance and expose the current helper's progress in the generated Colab notebook. The live 500-to-1000 extension stopped at step 862 because Lightning restored the previous Timer elapsed state into the same 90-minute limit. Reset only the time allowance, never the learning state. This is an operational correction, not evidence of better model quality or Phase 3 completion.
</objective>

<context>
@AGENTS.md
@.planning/STATE.md
@.planning/quick/261008-m6w-build-resource-efficient-phase3-h1-noteb/261008-m6w-SUMMARY.md
@train.py
@tools/phase3_h1.py
@tools/build_colab_phase3_notebook.py
@tests/test_phase3_h1.py
@utils/checkpoint_contract.py

Locked scope: preserve configuration-chain hashes, frozen data/protocol, checkpoints and existing B0 caches; no GPU jobs or Drive mutations during implementation. Extensions/full development-set evaluation stay explicitly off. API coverage and schema detectors are not applicable: no external API integration or database/schema migration is introduced. Assumption delta: SESSION_HOURS means a fresh allowance for each trainer.fit training session, not total process/setup wall time and not a cumulative run allowance.
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Resume one H1 run with a fresh timer and unchanged learning state</name>
  <files>train.py, utils/session_timer.py, tools/phase3_h1.py, tests/test_session_timer.py, tests/test_phase3_h1.py</files>
  <behavior>
    - An exhausted saved Timer allowance does not stop an opt-in resumed fit immediately.
    - Timer-only reset leaves model, optimizer, scheduler, loop progress and global step intact; repeating resume grants another bounded session.
    - A normal train.py invocation without the flag preserves cumulative Timer behavior.
    - The Phase 3 helper passes the explicit session flag while keeping the existing full-state --resume checkpoint and max_steps target.
  </behavior>
  <action>First add failing regression coverage for a saved Timer whose elapsed training time already reaches its duration. Add a narrowly scoped public Lightning Timer subclass in utils/session_timer.py and an opt-in --session-time-limit flag in train.py. Override the documented callback state restoration behavior to reset only its elapsed allowance; use public callback/state-key behavior verified against the installed Lightning version or official source. When enabled, supply exactly one timer callback using the configured max_time and prevent the trainer from adding a second cumulative Timer. Keep all other callback checkpoint restoration and trainer.fit(ckpt_path=resume_path) unchanged. Wire the flag into tools/phase3_h1.py's train command without changing any configs or hashes, checkpoint payloads, target limits, gating or automatic-restart behavior. Print the new session allowance and resumed starting step clearly. Add a small real CPU Lightning fit/save/resume test when the runtime is available; otherwise retain runnable deterministic behavioral tests and report the integration-test limitation explicitly.</action>
  <verify><automated>python -m pytest tests/test_session_timer.py tests/test_phase3_h1.py -q -p no:cacheprovider</automated></verify>
  <done>The opt-in path resumes an exhausted-timer checkpoint and continues from the saved learning step; non-timer learning state is preserved, no second Timer exists, default behavior remains compatible, and existing frozen configuration files are untouched.</done>
</task>

<task type="auto" tdd="true">
  <name>Show helper progress and failures through the permanent notebook workflow</name>
  <files>tools/phase3_h1.py, tools/build_colab_phase3_notebook.py, Wild_Diff_ICMH_Phase3_H1.ipynb, tests/test_phase3_h1.py</files>
  <behavior>
    - Notebook p3() explicitly streams merged stdout/stderr line by line with flushing and raises on a nonzero exit.
    - Preparation, checkpoint reading/copying, snapshot checks and evaluation stages announce their work before long operations.
    - Generated cells compile and match their builder; fixed pull and all expensive options remain unchanged.
  </behavior>
  <action>Update the builder's existing p3() wrapper to launch the freshly pulled helper through Popen with stdout PIPE, stderr STDOUT, line-buffered text, explicit printed lines and flushed output. Preserve argument-list invocation, working directory, helper failure propagation and the fixed safe pull cell. Add flushed stage messages before long existing prepare/snapshot/checkpoint staging and digest operations; preserve every integrity check, cache behavior and atomic copy. Do not introduce extra preprocessing, detector runs or evaluation scope. Clarify the SESSION_HOURS description as per-fit training time excluding setup/staging; retain RUN_EXTENSION, RUN_FULL_DEV and DISCONNECT_NOW defaults. Regenerate the notebook only through its builder, then add success/failure streaming and generated-cell regression tests.</action>
  <verify><automated>python -m pytest tests/test_phase3_h1.py -q -p no:cacheprovider</automated></verify>
  <done>The helper's progress is visible in Colab, failures remain nonzero and actionable, all cells compile, notebook generation is deterministic, and no extra GPU job or expensive option is enabled.</done>
</task>

<task type="auto">
  <name>Verify compatibility and document the exact resume handoff</name>
  <files>COLAB_TRAINING.md, .planning/quick/261013-fzr-fix-h1-resume-with-a-fresh-per-session-t/261013-fzr-SUMMARY.md</files>
  <action>Document the Timer carryover cause, the opt-in per-session correction, preserved full-state resume and the distinction between training allowance and total Colab/setup cost. State that a runtime already at step 862 can pull and retry the extension toward total step 1000, while a changed notebook wrapper requires opening the generated current notebook rather than copying temporary cells. Keep evaluation optional and avoid claiming model improvement or Phase 3 completion. Run targeted and full CPU regression checks, confirm git diff has no configuration/protocol/data changes, and record exact passing commands and any unavailable real-runtime integration check. Commit only implementation/test/docs files; the orchestrator owns the final STATE/summary documentation commit and push. Do not start training or modify live Drive artifacts.</action>
  <verify><automated>python -m pytest tests -q -p no:cacheprovider</automated></verify>
  <done>CPU regressions pass, generated notebook matches its builder, no frozen configuration or experiment data changes exist, implementation commits are ready, and the summary provides truthful verification and permanent-pull instructions.</done>
</task>

</tasks>

<threat_model>
## Trust boundaries and ASVS level 1 threats

| Threat | Severity | Disposition | Mitigation |
|---|---|---|---|
| Checkpoint pickle content enters the training runtime | high | mitigate | Restrict to existing user-trusted project checkpoint resolution and contract validation; introduce no new download or untrusted checkpoint source. |
| Resume correction accidentally discards learning state | high | mitigate | Reset Timer callback restoration only; retain full ckpt_path restoration and test optimizer, scheduler, loops and global step. |
| Subprocess output/arguments cross notebook-to-helper boundary | medium | mitigate | Keep argument arrays, merge stderr into streamed stdout, propagate nonzero status and avoid shell interpolation. |
| Retry consumes an unlimited GPU budget | high | mitigate | Retain positive bounded session duration, fixed targets and explicit extension/full-evaluation gates; no GPU execution during this fix. |
</threat_model>

<verification>
Targeted behavioral tests plus the full existing CPU suite; real tiny Lightning CPU resume if the available runtime supports it. Notebook generation/compilation and frozen-config diff inspection are mandatory. Do not infer live Colab acceptance from local tests.
</verification>

<source_audit>
GOAL: user-requested H1 operational fix is covered by tasks 1-3. REQ: no new roadmap requirement; full-state resumability and bounded resource use are preserved by task 1. RESEARCH: previously verified Timer elapsed carryover and hidden helper output are covered by tasks 1-2. CONTEXT: unchanged protocol/configuration, no GPU/Drive action, permanent pull workflow and disabled expensive options are covered throughout. No deferred research/model changes are included.
</source_audit>

<success_criteria>
Fresh time allowance works on opt-in resume without losing learning state; the legacy path is unchanged; notebook progress is visible; compatibility tests pass; generated notebook and documentation are ready for orchestrator commit/push and user execution.
</success_criteria>

<output>
Create .planning/quick/261013-fzr-fix-h1-resume-with-a-fresh-per-session-t/261013-fzr-SUMMARY.md with actual verification evidence. Orchestrator owns STATE changes and final push.
</output>
