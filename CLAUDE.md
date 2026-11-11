# What Claude did

This project was extended with help from Claude (Anthropic's AI assistant, through Claude Code) in one working session on 10 November 2026. This file records what Claude changed, ran and found, so a reader can tell AI-assisted work apart from the team's original work.

The team wrote the original pipeline (`netmf.py`, `models.py`, `fusion.py`, `run_experiments.py`, `plot_results.py`, the tests and the run scripts), the proposal and the intermediate presentation. Everything below was done by Claude at the team's request and reviewed by the team.

## Summary

- Found and fixed a feature-scaling bug that made the NetMF + GCN baseline look 14–24 points worse than GCN.
- Reran every experiment with the fix, then added PubMed (with a scalable NetMF) and ten-seed reruns for the close comparisons.
- Ran two diagnostic studies that explain why the learned router does not beat GCN.
- Wrote up the results and drafted the report story, and extended the intermediate deck into the final presentation.
- Cleaned up the code for sharing, checking that outputs did not change.

## Commits

All work is on the branch `netmf-scaling-fix`, in five commits on top of the team's `main`:

| Commit | Change |
|---|---|
| `c67c4f5` | Rescale NetMF features in NetMF + GCN; record package versions; regenerate Cora, CiteSeer and missing-edge results |
| `2d332bb` | Add the router diagnostic, the report story and corrected oracle figures |
| `bb9ee0b` | Add the ten-seed missing-edge experiment |
| `3859007` | Add PubMed with an eigen-approximation NetMF |
| `700395f` | Clean up code and repository layout for sharing |

## Code changes

| File | Change |
|---|---|
| `run_experiments.py` | NetMF + GCN rescales the NetMF block to 0.1, 0.3 or 1× the word features' row norm, chosen on validation accuracy; the old version still runs as `NetMF_GCN_Unscaled`. Adds `--netmf-method` and `--netmf-rank`. Records package versions in `config.json`. The main loop was split into small functions. |
| `netmf.py` | Adds `compute_netmf_embedding_eigen`, the large-window NetMF approximation from the NetMF paper (top 256 eigenpairs). It handles PubMed in about 16 s and 2.8 GB; the exact version would need well over 10 GB. |
| `plot_results.py` | Fixes the oracle line being cut off and overlapping labels; adds a colour for the unscaled baseline. |
| `tests/test_fusion.py` | Adds tests for feature scaling and the eigen NetMF (8 tests in total). |
| `run_pubmed.sh` | New PubMed run script. |
| `diagnostics/` | New: `scaling_check.py` and `predictability_check.py`, with saved results. |
| `requirements-lock.txt` | New: exact package versions behind the committed results. |

Removed: `run_intermediate.sh` and the placeholder report and slide template generators, along with their output files.

## Experiments run

All runs used an Apple Silicon Mac (16 GB) with PyTorch on MPS.

1. **Reproduction of the original Cora results.** Means matched, but only 20 of 165 rows matched exactly. Runs are deterministic within one environment, so the difference comes from package versions.
2. **Feature-scaling diagnostic** (Cora and CiteSeer). Showed that unscaled concatenation, not NetMF itself, caused the weak baseline.
3. **Full reruns with the fix:** Cora, CiteSeer and Cora missing edges, 5 seeds each.
4. **Router-predictability diagnostic.** Checks whether the router's inputs can tell which expert is right.
5. **Missing edges with 10 seeds**, to test NetMF + GCN's small lead from the 5-seed run.
6. **PubMed** with 5 seeds and then 10 seeds, after validating the eigen NetMF against the exact version on Cora and CiteSeer.

## Main findings

- **No method reliably beats GCN on Cora or CiteSeer.** Every gap is within about 1 point.
- **On PubMed, NetMF + GCN beats GCN by about 1 point:** +1.3 at 20 labels (p = 0.02–0.04 over 10 seeds) and +1.05 pooled (p = 0.01). This is the only positive fusion result, and part of it may come from the eigen NetMF.
- **The experts make different mistakes.** An oracle is 8.5–12.5 points above GCN.
- **The learned router cannot use that headroom.** Among nodes where an expert disagrees with GCN, its inputs predict the right one with an AUC of only 0.54–0.65, and GCN wins most of those disagreements. The router learns to give NetMF almost no weight.
- **NetMF adds no robustness to missing edges** (+0.1 points at 40% removed, p = 0.81). The router falls 1.3 points behind GCN there (p = 0.03).
- **The degree hypothesis is backwards.** NetMF is weakest on low-degree nodes on all three datasets.

Full numbers are in `README.md` and `docs/OBSERVATIONS.md`.

## Documents and presentation

- `docs/OBSERVATIONS.md`: a record of every run and diagnostic, including problems found along the way.
- `docs/FINAL_REPORT_AND_SLIDE_CONTENT.md`: filled in with results, a draft abstract, the conclusion and a slide outline.
- `README.md`: results tables, method notes, diagnostics and limitations updated.
- `presentation/final_presentation.pptx`: the team's intermediate deck (slides 1–12, unchanged) plus 14 new slides with final results, charts and speaker notes.

## How the work was checked

- All 8 unit tests pass, and the code is lint-clean under ruff (rules E, F, W, B, I).
- After the code clean-up, a smoke run produced exactly the same results as before it.
- The rewritten diagnostics reproduce their earlier results.
- Close comparisons use 10 seeds and paired tests. Where 5-seed results did not hold up at 10 seeds, the docs say so.
- Every figure in the deck and the docs was taken from the result CSVs. Claude corrected one figure it had initially misstated: the oracle gap was first given as "8–9 points".

## Not done

- Cora and CiteSeer were not rerun with the eigen NetMF, so PubMed's NetMF is computed differently from the other two datasets.
- NetMF scales below 0.1 were not tested, although CiteSeer chose 0.1 in 13 of 15 runs.
- GPU memory was not measured, because MPS does not report it.
- Proposal slides 6 and 7 in the deck still mention 40 labels per class and GPU memory. The new slides explain both changes.
