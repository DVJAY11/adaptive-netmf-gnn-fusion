# Observations from the reproduction run (2026-11-10)

Environment: Apple M-series Mac (16 GB RAM), Python 3.13.5, torch 2.15.0,
torch-geometric 2.8.1, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 2.3.3.
Training device: MPS (auto). Sections 1–6 were written **before** the code was
changed; section 0 records the full rerun **after** the NetMF_GCN scaling fix.

## 0. Update: full rerun after the scaling fix

**Code changes**
- `NetMF_GCN` now rescales the NetMF block relative to the mean row L2 norm of the
  original features. The scale is chosen from {0.1, 0.3, 1.0} by accuracy on the
  base early-stopping validation split; test labels are not used. The chosen value is
  saved in the `netmf_scale` column, and the reported training time is the total
  for all three candidate trainings.
- The old concatenation is still run and reported as `NetMF_GCN_Unscaled`.
- `config.json` now records package versions; `requirements-lock.txt` pins the
  environment used for this run.
- The `model_comparison.png` y-axis now includes the oracle line, which was previously
  cut off, and its x-axis labels no longer overlap.
- Added a unit test for the scaling (7 tests, all passing).

`run_final_cora.sh`, `run_citeseer.sh` and `run_missing_edges.sh` were each rerun
(about 11–12 min each on MPS), overwriting the committed `results_*` folders.

**Test accuracy (%), mean ± std over 5 seeds**

| Model | Cora 5 | Cora 10 | Cora 20 | CiteSeer 5 | CiteSeer 10 | CiteSeer 20 |
|---|---:|---:|---:|---:|---:|---:|
| GCN | 72.5 ± 2.1 | 74.5 ± 3.6 | 78.0 ± 1.6 | **60.3 ± 4.9** | **64.8 ± 2.1** | 70.7 ± 1.1 |
| NetMF_GCN (scaled) | 73.1 ± 3.5 | **75.0 ± 2.3** | **78.2 ± 1.4** | 59.7 ± 5.5 | 64.3 ± 4.2 | **70.9 ± 1.8** |
| NetMF_GCN_Unscaled | 51.5 ± 1.1 | 58.3 ± 3.1 | 64.2 ± 1.3 | 34.2 ± 5.3 | 40.5 ± 1.9 | 46.9 ± 1.1 |
| GraphSAGE | 70.9 ± 2.8 | 72.8 ± 2.9 | 75.8 ± 0.5 | 56.8 ± 6.4 | 62.0 ± 4.4 | 69.7 ± 2.0 |
| NetMF (LogReg) | 53.5 ± 4.6 | 61.2 ± 4.0 | 68.4 ± 1.7 | 29.8 ± 4.6 | 35.1 ± 2.2 | 42.9 ± 1.0 |
| EqualAverage | 72.9 ± 3.2 | 74.5 ± 2.7 | 77.8 ± 0.6 | 56.1 ± 7.1 | 59.7 ± 6.3 | 65.7 ± 3.3 |
| DegreeRouter | **73.2 ± 3.0** | 74.6 ± 2.8 | 77.9 ± 0.7 | 55.4 ± 6.4 | 57.8 ± 6.4 | 63.5 ± 3.4 |
| LearnedRouter | 71.7 ± 3.0 | 73.0 ± 2.8 | 77.2 ± 1.8 | 58.9 ± 5.4 | 64.2 ± 3.1 | 70.2 ± 1.0 |
| Oracle (not deployable) | 82.3 | 83.8 | 86.5 | 72.7 | 75.7 | 79.8 |

**Cora missing edges, 20 labels/class**

| Edges removed | GCN | NetMF_GCN | EqualAverage | LearnedRouter | NetMF alone |
|---:|---:|---:|---:|---:|---:|
| 0% | 78.0 | **78.2** | 77.8 | 77.2 | 68.4 |
| 20% | 76.3 | **76.8** | 76.5 | 76.0 | 61.4 |
| 40% | 72.9 | **73.7** | 72.5 | 70.8 | 53.5 |

**What changed and what it means**
- **Rescaling fixed the NetMF_GCN baseline:** +14 points on Cora and +24 on CiteSeer at
  20 labels. Its scores now equal GCN's within noise everywhere (all differences ≤ 0.9 points,
  far below the 1.4–5.5-point standard deviations). It beats GCN on 1–4 of 5 seeds,
  depending on the configuration.
- **Validation picked a small scale every time:** 0.1 or 0.3, never 1.0. On CiteSeer it
  picked 0.1 in 13 of 15 runs. 0.1 is the edge of the candidate grid, so it is worth
  trying smaller values (e.g. 0.03) or 0.
- **The headline conclusion is now:** no method reliably beats a plain GCN. That includes
  adding NetMF features, averaging, and both routers. Every Cora gap is under about 1 point.
  On CiteSeer, the averaging and degree routers are clearly *worse* (5–7 points at
  20 labels) because NetMF alone is weak there (43%).
- **Missing edges:** scaled NetMF_GCN is the best method at every edge-removal rate,
  but only by 0.2–0.9 points, all within one standard deviation. This is the place to
  check with more seeds if you want to claim a robustness benefit.
- **The learned router still almost ignores NetMF:** mean NetMF weight ≤ 0.02 on Cora
  and in every missing-edge setting. CiteSeer is the exception, at 0.08 with 10 labels
  and 0.20 with 20. It beats GCN on at most 1 of 5 seeds in any configuration.
- **The oracle headroom is unchanged:** 8.5–9.8 points above GCN on Cora and 9.1–12.4 on CiteSeer.
- The README results tables and the "NetMF concatenation performs substantially
  worse" sentence are now out of date.

## 0b. Why the learned router fails (diagnostic, 2026-11-10)

Script: rebuilds NetMF, GCN and GraphSAGE exactly as the pipeline does (MPS, seeds
42-46, budgets 5 and 20), computes the router's "all" inputs, and fits logistic-regression
detectors on the validation nodes the router may use (router-train + router-stop).

**Naive test: detect "expert right, GCN wrong".** Test AUC is high (NetMF 0.72-0.91,
GraphSAGE 0.91-0.96), but switching to the expert on flagged nodes, with the threshold
tuned on validation, changes accuracy by -1.6 to +0.2 points (mean per setting). These
AUCs are inflated: such nodes require disagreement, and disagreement is an input.

**Controlled test: among nodes where the expert disagrees with GCN, predict which is right.**

| Dataset, labels | Expert | Disagree share | Expert right | GCN right | Neither | AUC |
|---|---|---:|---:|---:|---:|---:|
| Cora 5 | NetMF | 43.9% | 15.4% | 58.3% | 26.3% | 0.65 |
| Cora 5 | GraphSAGE | 16.9% | 30.8% | 39.5% | 29.7% | 0.61 |
| Cora 20 | NetMF | 28.7% | 22.6% | 56.3% | 21.0% | 0.64 |
| Cora 20 | GraphSAGE | 13.3% | 28.8% | 46.6% | 24.6% | 0.61 |
| CiteSeer 5 | NetMF | 67.3% | 11.1% | 56.3% | 32.6% | 0.58 |
| CiteSeer 5 | GraphSAGE | 23.2% | 25.4% | 41.6% | 33.0% | 0.56 |
| CiteSeer 20 | NetMF | 52.5% | 10.3% | 63.2% | 26.6% | 0.62 |
| CiteSeer 20 | GraphSAGE | 15.8% | 32.5% | 38.3% | 29.2% | 0.54 |

Per-seed AUC ranges from 0.44 to 0.74. Conclusion: the router's inputs carry little
information about which expert is right, and GCN wins most disagreements, so no
reweighting of these inputs can recover the oracle headroom. New signals are needed
(e.g. neighbourhood label agreement, feature-space distance to training nodes).

## 0c. Missing edges with ten seeds (2026-11-10)

`results_missing_edges_10seeds/`, seeds 42–51, Cora, 20 labels. Seeds 42–46 reproduce the
five-seed run in 179 of 180 rows; one GCN row differs by 0.2 points (one test node), most
likely MPS floating-point nondeterminism while another job shared the GPU.

| Edges removed | GCN | NetMF_GCN | EqualAverage | LearnedRouter |
|---:|---:|---:|---:|---:|
| 0% | 78.60 ± 1.8 | 78.53 ± 3.0 | 78.87 ± 1.6 | 78.17 ± 2.0 |
| 20% | 77.64 ± 2.0 | 77.64 ± 1.4 | 77.69 ± 2.1 | 76.69 ± 1.9 |
| 40% | 74.43 ± 2.7 | 74.57 ± 1.5 | 74.40 ± 3.3 | 73.12 ± 3.7 |

Paired difference from GCN (points; paired t-test p):

| Method | 0% | 20% | 40% |
|---|---:|---:|---:|
| NetMF_GCN | −0.07 (0.94) | +0.00 (1.00) | +0.14 (0.81) |
| EqualAverage | +0.27 (0.53) | +0.05 (0.86) | −0.03 (0.93) |
| LearnedRouter | −0.43 (0.28) | −0.95 (0.035) | −1.31 (0.032) |

The five-seed hint that NetMF_GCN is more robust does not hold up. The learned router is
lower than GCN on 10/10 seeds at 40% removal. With nine tests, its p-values would not survive
a Bonferroni correction, so describe this as consistent but modest.

## 0d. PubMed (2026-11-10)

**NetMF for large graphs.** `netmf.py` now has `compute_netmf_embedding_eigen`, the
large-window NetMF from the NetMF paper: top-256 eigenpairs of D^-1/2 A D^-1/2, filtered by
(1/T) sum_r lambda^r, dense log(max(M,1)) built in float32 row blocks, then randomized SVD.
On PubMed it takes about 16 s with 2.8 GB peak memory (smoke run). Selected with
`--netmf-method` (`auto` = eigen above 10,000 nodes; Cora and CiteSeer still use exact and
keep their cache names). A unit test was added (8 tests, all passing).

Validation against exact NetMF (NetMF logistic regression, seeds 42-46):

| Dataset | Labels | Exact | Eigen, rank 256 | Eigen, rank 512 |
|---|---:|---:|---:|---:|
| Cora | 5 | 53.5 | 58.0 | 57.4 |
| Cora | 20 | 68.4 | 70.4 | 70.6 |
| CiteSeer | 5 | 29.8 | 36.9 | 36.2 |
| CiteSeer | 20 | 42.9 | 48.4 | 48.2 |

The eigen version is better, consistent with the NetMF paper (truncation removes noise).
Consequence: PubMed's NetMF is not computed the same way as Cora's and CiteSeer's. Rerunning
Cora and CiteSeer NetMF_GCN with `--netmf-method eigen` would make them comparable; not done.

**Results.** `results_pubmed/` (seeds 42-46, about 8 min on MPS) and
`results_pubmed_10seeds/` (seeds 42-51; seeds 42-46 identical, 180/180 rows).

| Method (10 seeds) | 5 labels | 10 labels | 20 labels |
|---|---:|---:|---:|
| GCN | 67.13 ± 5.7 | 74.09 ± 2.4 | 76.35 ± 2.0 |
| NetMF_GCN | 68.51 ± 4.5 | 74.53 ± 2.6 | 77.67 ± 1.8 |
| NetMF_GCN_Unscaled | 60.54 ± 2.1 | 64.29 ± 3.3 | 68.12 ± 2.9 |
| GraphSAGE | 64.90 ± 3.8 | 71.95 ± 2.8 | 75.06 ± 2.8 |
| NetMF | 60.44 ± 3.7 | 67.92 ± 2.1 | 70.84 ± 2.7 |
| EqualAverage | 67.17 ± 5.2 | 74.31 ± 2.4 | 76.92 ± 2.3 |
| DegreeRouter | 67.00 ± 5.2 | 74.23 ± 2.4 | 76.66 ± 2.4 |
| LearnedRouter | 67.01 ± 5.6 | 74.08 ± 2.8 | 76.41 ± 1.6 |
| Oracle | 79.61 | 85.30 | 85.35 |

Paired NetMF_GCN minus GCN: +1.38 (p = 0.16), +0.44 (p = 0.30), +1.32 at 20 labels
(t p = 0.040, Wilcoxon p = 0.020, 7/10 wins). Pooled over 30 runs: +1.05 points, median
+0.7, 18 wins / 10 losses / 2 ties, Wilcoxon p = 0.010. With five seeds the gains looked larger
(+3.0, +1.1, +1.6) and none were significant; ten seeds shrank them. Treat this as a small but
consistent gain, the only positive fusion result in the project.

Learned router: ties GCN everywhere (-0.12 to +0.06 points); NetMF weight 0.10 / 0.02 / 0.02
at 5 / 10 / 20 labels. Selected NetMF scales: 0.1 in 16 runs, 0.3 in 12, 1.0 in 2. PubMed has
no router runs at the 300-epoch cap except 2 for the degree-only router (five-seed run).

## What was run

| Step | Result |
|---|---|
| Created `.venv` and installed `requirements.txt` | Installed cleanly |
| `python -m unittest discover -s tests -v` | 6/6 tests pass |
| `./run_smoke.sh` | Completed in about 50 s, including the Cora download |
| Full five-seed Cora run (`run_final_cora.sh` settings) written to a scratch directory | Completed; committed results were not overwritten |
| Diagnostic: NetMF_GCN with different NetMF feature scales (Cora and CiteSeer, budgets 5 and 20, seeds 42–46, CPU) | See section 1 |
| Determinism checks (CPU vs MPS, repeated processes) | See section 3 |

CiteSeer and the missing-edge experiment were not rerun end to end.

---

## 1. The NetMF_GCN baseline is weak because of feature scaling (most important)

`run_experiments.py` concatenates the row-normalized bag-of-words features with the
L2-normalized NetMF embedding without rescaling either block:

| | Cora | CiteSeer |
|---|---:|---:|
| Mean row L2 norm, original features `x` | 0.255 | 0.179 |
| Mean row L2 norm, NetMF embedding | 1.000 | 0.986 |

The NetMF block has about 4–5× the magnitude of the word features, so the GCN
mostly learns from NetMF. As a result, NetMF_GCN performs about as well as a GCN
given **only** the NetMF embedding:

GCN test accuracy (%), mean of 5 seeds:

| Input features | Cora, 5 labels | Cora, 20 labels | CiteSeer, 5 labels | CiteSeer, 20 labels |
|---|---:|---:|---:|---:|
| `x` only (plain GCN) | 72.1 | 77.1 | 60.1 | 71.8 |
| `[x, 0]` (sanity check) | 72.0 | 77.6 | 58.0 | 71.2 |
| NetMF only | 47.2 | 63.1 | 30.0 | 42.1 |
| **`[x, NetMF]` as in the repo** | **52.4** | **63.9** | **34.4** | **47.0** |
| `[x, NetMF × ‖x‖]` | 62.4 | 72.1 | 47.4 | 61.7 |
| `[x, NetMF × 0.3‖x‖]` | 71.7 | **78.5** | 58.4 | 70.5 |
| `[x, NetMF × 0.1‖x‖]` | 72.3 | 78.1 | 59.1 | 71.3 |

Standard deviations are 1–2 points at 20 labels and 3–5 points at 5 labels.

**Implications**
- The README states that "NetMF feature concatenation performs substantially worse."
  That is an artifact of the unscaled features, not a property of the method. When
  scaled properly, NetMF_GCN matches GCN, and on Cora with 20 labels it is about
  1 point higher, which is within noise.
- This is the baseline that answers the research question ("is the router better
  than just adding NetMF to the GCN?"). It should be fixed, for example by rescaling
  the NetMF block or standardizing both blocks, and the scale should be tuned on
  validation data before results are reported.
- (Numbers here are from CPU training, so plain GCN is about 1 point lower than in the
  MPS runs. Compare rows within this table, not against the README tables.)

## 2. The learned router mostly ignores NetMF

Mean NetMF weight assigned by the learned routers (averaged over all nodes and seeds):

| Run | Router | 5 labels | 10 labels | 20 labels |
|---|---|---:|---:|---:|
| Committed Cora | LearnedRouter (all features) | 0.131 | 0.109 | 0.0002 |
| Rerun Cora | LearnedRouter (all features) | 0.0001 | 0.016 | 0.0001 |
| Committed CiteSeer | LearnedRouter (all features) | 0.001 | 0.072 | 0.093 |

In most configurations the router learns to mix only GCN and GraphSAGE and assigns
NetMF almost no weight. In the Cora rerun with 20 labels, the weights for low-, medium-
and high-degree nodes are about NetMF 0.00, GCN 0.40–0.52, GraphSAGE 0.48–0.60.

- Even so, NetMF is uniquely correct on about 4–5% of Cora test nodes (oracle
  analysis), so the router does not detect the cases where NetMF should be trusted.
- The router is trained on about 150 validation nodes (30% of 500). That small set
  probably explains why it falls back to the safest experts.
- The router-weight plot (`router_weights.png`) changes a lot between environments
  (0.13 vs 0.0001 NetMF weight at 5 labels), so it should not be presented as a
  stable finding.

## 3. Reproducibility

- **Deterministic within an environment:** repeated runs with the same seed give
  identical results on CPU and on MPS, both within one process and across separate processes.
- **Not reproducible across environments:** compared with the committed
  `results_final_cora/raw_results.csv`, only 20 of 165 rows match exactly. Most per-run
  differences are within ±3 points, but some are larger (MLP up to ±5.5 points, one
  LearnedRouterConfidence run +19 points). The cause is most likely different library
  versions or hardware. `config.json` does not record package versions, so this
  cannot be confirmed. **Suggestion:** save `torch`, `torch_geometric`, `numpy`,
  `scipy` and `sklearn` versions in `config.json`, and pin exact versions in
  `requirements.txt` for the final runs.
- **The ranking at 5 labels per class depends on the environment:**

  | Cora, 5 labels | Committed | Rerun |
  |---|---:|---:|
  | GCN | **72.28** | 72.48 |
  | EqualAverage | 71.90 | 72.88 |
  | DegreeRouter | 71.6 | **73.16** |
  | LearnedRouter | 71.52 | 71.66 |

  In the committed run, EqualAverage beats GCN on 1 of 5 seeds; in the rerun, on 3 of 5.
  Differences under about 1 point should be described as "no meaningful difference," not
  as a win or loss for any method.
- Means at 20 labels are stable: GCN 78.26 → 78.02, LearnedRouter 77.42 → 77.20,
  oracle 86.3 → 86.5. The overall conclusion (the learned router does not beat GCN;
  the oracle shows about 8 points of headroom) holds.

## 4. Degree analysis does not support the degree-router hypothesis

Cora rerun, 20 labels, test accuracy (%) by degree tercile:

| Model | Low degree | Medium | High |
|---|---:|---:|---:|
| GCN | 74.3 | 79.7 | 81.7 |
| GraphSAGE | 70.7 | 77.9 | 80.8 |
| LearnedRouter | 72.8 | 78.9 | 81.9 |
| NetMF | 59.4 | 72.6 | 76.8 |

NetMF is **weakest** on low-degree nodes, the opposite of what `degree_router` assumes
(it gives NetMF weight 0.5 on low-degree nodes). The README's limitations section already
says this; the table above is direct evidence.

## 5. Training details

- **The learned router sometimes reaches the 300-epoch cap without early stopping:**
  6/45 router runs (Cora), 7/45 (CiteSeer), 5/45 (missing edges) ended with
  `best_epoch == 300`. Raising `epochs` or the learning rate for those runs could change
  their results.
- **CiteSeer has 48 isolated nodes**, and their NetMF embedding is all zeros (hence the
  mean row norm of 0.986). NetMF logistic regression can only guess the prior for them.
- **Peak memory is not measured on MPS:** `peak_memory_mb` is 0.00 for almost every
  model in the rerun, because the RSS-difference method does not capture GPU allocations.
  The memory columns should be removed from any report generated on a Mac.
- Mean training time per run on MPS (Cora): GCN 3.2 s, GraphSAGE 9.5 s, NetMF_GCN 2.3 s,
  learned router 1.8–3.6 s, NetMF logistic regression < 0.1 s. NetMF preprocessing for
  Cora takes under 1 s.

## 6. Smaller code notes

- `run_experiments.py:425` prints the LearnedRouter accuracy as `raw_rows[-2]`. It is correct
  now but breaks silently if rows are added or reordered.
- `val_accuracy` for the learned routers is computed on the full validation set, which
  includes the router's own training nodes. Test numbers are unaffected, but this
  validation score is optimistic.
- MLP probabilities are computed but not used in any fusion method.
- The default `--seeds` in `parse_args` is `[42, 43, 44]`; the shell scripts override it
  with five seeds.
- `netmf.py` builds a dense n×n matrix. This is fine for Cora and CiteSeer, but the
  estimated peak memory for PubMed (about 10 GB or more) would likely exceed free Colab
  RAM (about 12.7 GB) together with the SVD copies. Use an eigen-approximation (large-graph
  NetMF) before running PubMed; see the discussion on running on Colab.

## Suggested next steps (in priority order)

1. ~~Fix the NetMF_GCN feature scaling and rerun all three experiments.~~ Done; see section 0.
2. ~~Record package versions.~~ Done (`config.json` and `requirements-lock.txt`).
3. Update the README tables and conclusions, describing differences of about 1 point as ties.
4. If the router remains part of the project, give it more training data (for example,
   cross-fitting expert predictions on the training nodes) or regularize it toward equal
   weights, since it currently collapses to GCN/GraphSAGE.
5. Implement the large-graph NetMF approximation before running PubMed.



## 0e. Pairwise Oracle Analysis (5 Seeds, 3 Datasets)

Seeds 42–46, 20 labels/class, 1,000 test nodes per run.

| Combination | Cora | CiteSeer | PubMed |
|---|---:|---:|---:|
| GCN + NetMF | 84.30 ± 0.84 | 76.68 ± 1.27 | 83.94 ± 1.85 |
| GCN + GraphSAGE | 82.24 ± 2.11 | 75.48 ± 2.00 | 81.68 ± 1.14 |
| NetMF + GraphSAGE | 83.06 ± 1.27 | 74.54 ± 1.48 | 83.14 ± 1.50 |
| All three | 86.48 ± 1.51 | 79.70 ± 1.82 | 86.00 ± 1.76 |

### Key findings
- GCN + NetMF had higher oracle accuracy than GCN + GraphSAGE in 13 of 15 runs.
- NetMF provides complementary predictions despite its weaker individual accuracy.
- The three-model oracle offers approximately 8.3–8.4 percentage points of headroom over the best individual model.
- Higher individual model accuracy does not necessarily imply greater fusion value.
- The challenge remains learning when to trust each expert without using test labels.

### Limitations
Oracle accuracy is a theoretical upper bound, not achievable model performance. Results use five seeds and one label budget, so broader generalization requires additional evaluation.


## 0f. Weighted Prediction Averaging (2026-11-10)

**Single-seed experiment:** Cora, seed 42, 20 labels/class. Fixed probability weights were
searched in steps of 0.1 using only the 500-node saved validation mask; the selected weights
were then evaluated once on the 1,000-node test mask. No models were retrained.

| Method | Selected weights (GCN / NetMF / GraphSAGE) | Validation | Test |
|---|---:|---:|---:|
| Weighted GCN + NetMF | 0.4 / 0.6 / 0.0 | 78.8% | **78.5%** |
| Weighted GCN + GraphSAGE | 1.0 / 0.0 / 0.0 | 77.4% | 78.3% |
| Weighted all three | 0.4 / 0.6 / 0.0 | 78.8% | **78.5%** |
| GCN | 1.0 / 0.0 / 0.0 | 77.4% | 78.3% |
| EqualAverage | 1/3 / 1/3 / 1/3 | 76.4% | 77.1% |
| LearnedRouter | node-dependent | 75.8%* | 76.9% |

The best validation-selected fixed average is 0.4 GCN + 0.6 NetMF. It reaches 78.5% test
accuracy, 0.2 percentage points above GCN, 1.4 points above equal averaging, and 1.6 points
above the learned router in this seed. Adding GraphSAGE does not help the validation search:
the selected three-model candidate gives it zero weight. Likewise, the GCN + GraphSAGE search
selects pure GCN. These small single-seed differences do not establish a general improvement;
multiple seeds are required.

\* The learned router's validation accuracy is not directly comparable to validation-tuned
averaging. It was fit on 149 router-training nodes and early-stopped on 103 router-stop nodes,
all of which are included in the 500-node validation mask used to report 75.8%. Its held-out
test accuracy is comparable. The fixed-weight validation score is also a selection score rather
than performance on an untouched split.
