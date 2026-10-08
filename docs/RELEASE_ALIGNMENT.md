# Alignment of main with the final paper

The final Overleaf manuscript was inspected at commit
`b64d77ad0d38f29b1aded73247dcd6f7fc172ee8` on 8 October 2026, with `main.tex`
as its root. Its main and transfer tables agree with the author's final README
scores. The public reference configuration and recovered historical recipes
retain their own identities. This checklist concerns implementation and
configuration alignment; historical scores do not replace final results.

| Item | Main release | Alignment work |
|---|---|---|
| Editor budget | Demo: 16 updates at 1e-6 | Final manuscript specifies 400 updates at 1e-4 |
| Planner budget | Step limit unset | Final manuscript specifies 16 updates at 1e-5 |
| Proposed edit families | Removal, replacement and addition | Match the final task mixture; the demo does not target primary extraction |
| Programmatic extraction checks | Absent from the main reward path | Final manuscript names background purity and object completeness, but gives no executable definitions or thresholds; obtain their final implementation |
| Yes/no verifier bank | Historical code is on the detector branch | Final manuscript places token-probability verification in a separate analysis variant, not as a mandatory primary-loop component; the final bank implementation still needs mapping |
| Editor target weights | Accepted targets have weight 1; rejected targets excluded | Final manuscript describes reward-weighted SFT but does not specify how reward maps to record weight; obtain the effective configuration |
| Identity/reconstruction replay | Demo replay ratio is zero | Final manuscript includes replay in the Editor objective; obtain the actual replay manifest, ratio and weights |
| Planner band-pass admission | Productive-band requirement defaults to off | Final manuscript excludes all-fail and all-pass proposals; obtain the configuration that enforces this rule |
| Editor reference regularization | Demo specifies L2 weight 0.01 and relative-delta cap 0.1 | Final manuscript specifies anchoring and a cap without their numeric values or warm-start adapter identity |
| Step1X training and inference | Public CLI supports Qwen only; Step1X paths are benchmark scorers | Final manuscript reports a second editor using its own VLM and VAE; obtain the corresponding training/feature backend |
| Final artifact mapping | Final run bundle location not supplied | Associate the final config, manifest, adapter hashes and evaluation records with each final table |

The core internal reward, rubric gates and separate Planner/Editor training
paths are present. External detector loading is absent from `main`; optional
internal VLM judging uses the editor-side component and should not be described
as an external reward model. Historical detector-assisted miner code belongs to
the separate branch.

The supplied YAML remains a demonstration configuration. The exact settings
stated in the manuscript are transcribed in
[`final_paper_settings.json`](../reproducibility/final_paper_settings.json).
That record is a manuscript settings inventory, not an executable training
configuration. Missing extraction formulas, weight/replay rules and adapter
identities cannot be recovered from the manuscript alone.

The final-paper recipe should be recovered from the final run's effective
configuration rather than inferred by changing isolated hyperparameters. The
historical detector-assisted extraction heuristic is not a substitute for the
final primary method's purity/completeness checks.
