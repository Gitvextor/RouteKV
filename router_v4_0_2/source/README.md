# Router V4.0.2

Router V4.0.2 is a same-data internal development pilot. It preserves all
V4.0.1/V4.0.0/V3 artifacts and the frozen six-action V1 meanings.

The Primary architecture is:

```text
Frozen z64 + normalized structural16
  -> Full Failure linear head

Question base80 + policy descriptor + interaction
  -> shared candidate-level CIF linear head

Structural16 + policy descriptor
  -> shared Peak-KV cost head

10-member document bootstrap
  -> cross-fitted Platt calibration
  -> one-sided risk UCB
  -> zero-CIF conservative selector
```

H2O route rate is not an acceptance gate. Zero compression-induced failures
and preservation of Fixed Full acceptable count remain reported quality
targets. A user-authorized override now allows an all-780 `research_only`
bundle after technical integrity gates pass. The override does not erase the
observed failures, enable routing, or make the bundle production-eligible.

```powershell
$py = 'E:\Users\23001\Desktop\project\codex\router-v2-env\Scripts\python.exe'
$cfg = 'G:\2026.08_KV_Cache_Research\codex\router_v4_0_2\configs\shared_candidate_risk_v1.yaml'

& $py -m pytest codex\router_v4_0_2\tests codex\router_v4_0_1\tests codex\router_v4_0_0\tests -q -p no:cacheprovider
& $py -m codex.router_v4_0_2.scripts.audit_data --config $cfg
& $py -m codex.router_v4_0_2.scripts.build_candidates --config $cfg
& $py -m codex.router_v4_0_2.scripts.prepare_nested_crossfit --config $cfg
& $py -m codex.router_v4_0_2.scripts.cuda_tiny_smoke --config $cfg
& $py -m codex.router_v4_0_2.scripts.run_nested_oof --config $cfg --device cuda
& $py -m codex.router_v4_0_2.scripts.evaluate_nested_oof --config $cfg
& $py -m codex.router_v4_0_2.scripts.render_report --config $cfg
& $py -m codex.router_v4_0_2.scripts.fit_research_candidate --config $cfg --device cuda
```

Even if gates pass, this is not blind evidence or a production Router because
the same 780 outcomes were exposed before V4.0.2 was designed.
