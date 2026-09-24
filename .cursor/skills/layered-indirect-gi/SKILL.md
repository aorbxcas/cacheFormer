---
name: layered-indirect-gi
description: >-
  Implements and benchmarks CacheFormer three-layer indirect GI (L1 I-buffer warp,
  L2 skip-frame fast term, L3 VI cache + adaptive RF refresh). Use when changing
  LayeredIndirectPipeline, skip_fast_term, hybrid/C1 scheduling, or when the user
  asks for neural-indirect performance vs CacheFormer, Direct/Indirect split, or
  skip-RF quality locks.
---

# Layered indirect GI

## Quality lock (do not violate)

- Compare speed to **CacheFormer** = every-frame RF + VI cache.
- Full resolution only (`quality_lock`, `neural_res_scale=1`).
- Default `quality_anchor=rf` (warp RF HDR with analytic-plane inverse bilinear). Do not turn on classic Direct follow or an untrained residual head for vs-CF tests.
- Pass: `speedup > 1` and `mean absL1 vs CF < 0.08`.
- Bench dynamics: `orbit_only,fov_sweep,roughness_orbit,specular_orbit,irradiance_orbit,roughness_fast,combined`.

## Code map

- Pipeline: `renderformer/c1/layered_pipeline.py`
- L2 ops: `renderformer/c1/skip_fast_term.py`
- Bench: `tools/benchmark_layered_vs_cf.py`
- Design + debate: `docs/layered_indirect_three_layer.md`
- Project hub: `docs/README.md`
- C1 residual head: `docs/C1_residual_indirect_head.md`

## Refresh policy (L3)

Force RF on empty buffer, scene fingerprint change, `rot >= max_camera_rot_deg`, or `elapsed >= max_skip_run`. Interval refresh only if `elapsed >= refresh_every` **and** `rot >= soft_rot_deg`; otherwise defer.

## Product vs CF-match

- CF-match: `quality_anchor=rf`, `direct_mode=stub`, `l2_gate_strength=0`, `l2_head_mix=0`
- Product: `quality_anchor=hybrid`, `l1_direct_follow=True`, `direct_mode=always` (nvdiffrast). Expect absL1 vs CF to fail; that is not a regression of the RF-anchor path.
