---
title: "CortexBridge vNext: Current Evidence, Final Method Definition, and Journal Experiment Roadmap"
authors: ["Zhou et al. (working project; final author list not specified)"]
year: 2026
venue: "Target: IEEE TMI or Medical Image Analysis; not yet submitted"
doi: "Not available from provided input"
ara_version: "1.0"
domain: "Medical image analysis, cortical representation learning, privileged distillation"
keywords: ["T1 MRI", "cortical surface", "cortical ribbon", "DKT atlas", "cross-geometry correspondence", "graph learning", "knowledge distillation", "cognitive diagnosis"]
claims_summary:
  - "Affine-aware ribbon geometry is valid, but pure ribbon pooling does not show a stable classification advantage over legacy pooling."
  - "Hard surface-ribbon supervision learns strong ROI identity correspondence with only a weak subject-specific increment."
  - "Real mesh-neighbour enrichment motivates, but does not yet validate, an anatomy-prior ROI graph and MRI-only graph-state distillation."
abstract: "This evolving artifact records the implemented CortexBridge vNext system, exact completed evidence, negative results, and the falsifiable route to a journal-ready method. The current validated method retains legacy MRI ROI pooling for classification and uses high-resolution ribbon sampling only as an auxiliary surface-to-MRI correspondence anchor. A real fsaverage DKT mesh topology has been audited but has not yet entered the model. The planned core additions are an anatomy-prior disease-aware ROI graph and graph-state distillation to a strict MRI-only Student."
---

# CortexBridge vNext Research Artifact

## Overview

This artifact separates implemented facts from proposed journal components. The reader-facing Chinese synthesis is [`../CortexBridge_vNext_Journal_Method_Status_and_Experiment_Roadmap.md`](../CortexBridge_vNext_Journal_Method_Status_and_Experiment_Roadmap.md).

## Layer Index

### Cognitive Layer (`logic/`)

| File | Description |
|---|---|
| [problem.md](logic/problem.md) | Observations, gaps, insight, and assumptions |
| [claims.md](logic/claims.md) | Ten falsifiable supported/refuted/hypothesis claims |
| [concepts.md](logic/concepts.md) | Formal definitions and scope boundaries |
| [experiments.md](logic/experiments.md) | Completed verification designs and planned journal experiments |
| [solution/architecture.md](logic/solution/architecture.md) | Current and proposed component graphs |
| [solution/algorithm.md](logic/solution/algorithm.md) | Current losses and proposed graph/distillation algorithms |
| [solution/constraints.md](logic/solution/constraints.md) | Scientific and deployment boundaries |
| [solution/heuristics.md](logic/solution/heuristics.md) | Evidence-grounded design heuristics |
| [related_work.md](logic/related_work.md) | Typed internal/external dependency map |

### Physical Layer (`src/`)

| File | Description | Claims |
|---|---|---|
| [configs/training.md](src/configs/training.md) | Locked and proposed hyperparameters | C03, C06–C09 |
| [configs/model.md](src/configs/model.md) | Current and planned architecture settings | C01–C09 |
| [execution/method_spec.py](src/execution/method_spec.py) | Typed algorithm specification | C03, C05, C07, C08 |
| [environment.md](src/environment.md) | Software/hardware provenance | C01–C10 |

### Exploration Graph (`trace/`)

| File | Description |
|---|---|
| [exploration_tree.yaml](trace/exploration_tree.yaml) | Explicit research trajectory with decisions, pivots, and dead ends |

### Evidence (`evidence/`)

| File | Description |
|---|---|
| [README.md](evidence/README.md) | Index of exact evidence tables and claim bindings |

