# Probing Visual Affordance in Vision-Language-Action Encoders

How does part / affordance structure evolve as a vision encoder is fine-tuned into a
Vision-Language-Action (VLA) policy? This project probes that representation across stages.

<!--📄 [Full report](LVB_final_report.pdf)-->

## Overview

Modern robot policies route vision through one of two architectural paradigms, and we ask
whether each carries a *different kind* of affordance signal:

- **Vision–Language–Action (VLA)** models (π0, π0.5) fine-tune a contrastive SigLIP encoder
  inside a PaliGemma backbone toward action prediction.
- **Generative world-model policies** (e.g. Flux-style diffusion) let language drive spatial
  generation through cross-attention.

We probe both along two complementary axes:

- **Axis 1** — linear probing + unsupervised PCA of the SigLIP encoder along the VLA
  fine-tuning trajectory (raw → PaliGemma → π0 → π0.5) on **UMD**.
- **Axis 2** — cross-attention extraction from **FLUX.1** during denoising, measuring
  verb-spatial binding on **AGD20K**.

## Method

**Axis 1 — VLA-stage probing (UMD).** Each frozen encoder is hooked at four equally-spaced
intermediate layers (bilinear-resize + channel-concat fusion → BatchNorm–1×1 Conv head),
predicting UMD's seven affordance classes (grasp/cut/scoop/contain/pound/support/w-grasp),
scored by mIoU at n=4 seeds. Six encoders (DINOv2-B/L and four SigLIP-So400m stages) at two
resolutions: **res224** (16×16 patches, the VLA operating resolution) and **resumd**
(~480×640, 35×46 patches). *Pipeline check:* DINOv2-B at UMD-native reproduces 0.666 mIoU
vs. 0.670 reported in prior work.

**Axis 1 complement — per-layer PCA.** Patch features extracted at every transformer layer
(28 SigLIP / 13 DINOv2) across three domains (web, phone, UMD); we report PC1 explained-variance
ratio and PC1/PC2/PC3 → R/G/B subspace-projection colormaps on held-out objects. Unsupervised,
so it sidesteps the supervised probe's resolution-dependent sensitivity ceiling.

**Axis 2 — verb-spatial binding (FLUX.1).** A custom `AttentionProcessor` on every
`FluxTransformerBlock.attn` records the post-softmax image-query × text-key slice for the prompt
"a person {verb} a {object}". Maps are head-averaged, aggregated across timesteps, and scored
against AGD20K GT heatmaps with KLD, SIM, and NSS. Tested on FLUX.1-schnell (4 steps, no CFG)
and FLUX.1-dev (20 steps, CFG=3.5) over n=1675 across 36 categories.

## Key results

**Axis 1 — encoder rankings depend on resolution.** At UMD-native, DINOv2 beats every SigLIP
variant (DINOv2-B at 86M params > SigLIP at 400M, +3.5 pp); at the VLA operating resolution the
encoders compress into seed noise.

| Encoder (params) | mIoU @ UMD-native | mIoU @ 224×224 |
|---|---|---|
| DINOv2-L (300M) | 0.666 | 0.377 |
| DINOv2-B (86M)  | 0.663 | 0.338 |
| SigLIP raw (400M)   | 0.628 | 0.312 |
| SigLIP PaliGemma-1  | 0.602 | 0.352 |
| SigLIP π0           | 0.550 | 0.372 |
| SigLIP π0.5         | 0.565 | 0.389 |

**The VLA trajectory reverses sign across resolutions.** Fine-tuning helps monotonically at
res224 (0.31 → 0.39, **+7.7 pp**) but is net negative at resumd (0.63 → 0.56, **−6.3 pp**) —
VLA training produces a *resolution-specialized* representation.

**PCA reveals stage-specific geometry the probe cannot see.** Three signatures emerge: DINOv2
stays distributed (PC1 ≈ 0.10–0.20); raw SigLIP / PaliGemma trace an inverted-U (mid-network
bottleneck PC1 ≈ 0.55–0.65); **π0 develops a sharp final-layer collapse (PC1 ≈ 0.83)**, while
**π0.5 does not collapse** (flat ~0.20, structurally closer to DINOv2). Subspace-projection
colormaps show only DINOv2 produces consistent part-vs-background structure; every SigLIP
variant — including π0.5 at matched PC1 — shows only object-vs-background. Feature magnitude
grows with VLA depth (std: raw 1.10 → π0 2.96 → π0.5 3.95; DINOv2 0.74). The π0-vs-π0.5
distinction is **invisible to the supervised probe** but clear under PCA.

**Axis 2 — verb-spatial binding is real but uneven.** Across 1675 samples KLD=1.86, SIM=0.25,
NSS=+0.43, all departing from the uniform-attention null in the expected direction.
Counterintuitively, **FLUX-schnell (cheap) beats FLUX-dev** on KLD/SIM (p < 10⁻⁴), and
**manipulation verbs bind *weaker*** than non-manipulation verbs (NSS +0.349 vs +0.509) — the
signal is shaped by GT-region geometry, not verb semantics.


## References

Key references from the report: π0 / π0.5 (Physical Intelligence), SigLIP (Zhai et al., ICCV 2023),
PaliGemma (Beyer et al., 2024), UMD (Myers et al., ICRA 2015), FLUX.1 (Black Forest Labs, 2024),
AGD20K (Luo et al., CVPR 2022), DINOv2 (Oquab et al., TMLR 2024). Full list in the report.


<!--## Setup -->
<!-- TODO: dependencies + install -->

<!--## Usage / reproducing the analysis -->
<!-- TODO: how to run the probes / regenerate the figures --> 

## Status
Graduate course research project, Spring 2026 (research contributor).
