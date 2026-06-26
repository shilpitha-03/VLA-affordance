# Probing Visual Affordance in Vision-Language-Action Encoders

How does part / affordance structure evolve as a vision encoder is fine-tuned into a
Vision-Language-Action (VLA) policy? This project probes that representation across stages.

## Overview

We track affordance structure across the VLA fine-tuning pipeline
**SigLIP → π0 → π0.5** to see where, and how, the encoder reorganizes its representation.

## Method

- **Per-layer PCA** across the encoder stack.
- **Subspace-projection analysis** to compare representation geometry across stages.

## Key finding

- A **stage-specific representation collapse** emerges during fine-tuning that is
  **invisible to standard supervised probing** — only the subspace analysis surfaces it.

## Repository structure
<!-- TODO: list main folders/files -->

## Setup
<!-- TODO: dependencies + install -->

## Usage / reproducing the analysis
<!-- TODO: how to run the probes / regenerate the figures -->

## Status
Graduate course research project, Spring 2026 (research contributor).
