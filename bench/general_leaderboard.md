| system | POPE | MME | NLVR2 (2 images) | A-OKVQA (4-way) | Food-101 (20-way) | mean acc [95% CI] | errors stated at p>=0.99 | ms |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct frozen, prompt + logit readout | 0.900 / 0.96 (ECE 0.072) | 0.903 / 0.97 (ECE 0.076) | 0.897 / 0.96 (ECE 0.085) | 0.903 (ECE 0.084) | 0.950 (ECE 0.037) | 0.911 [0.897-0.925] | 49 of 134 (37%) | 185.1 |
| V5b (ours, trained on screen state judgments) | 0.860 / 0.73 (ECE 0.094) | 0.890 / 0.77 (ECE 0.033) | 0.917 / 0.79 (ECE 0.016) | 0.890 / 0.90 (ECE 0.080) | 0.937 / 0.91 (ECE 0.046) | 0.899 [0.883-0.913] | 22 of 152 (14%) | 268.9 |
| V7 (ours, general mix + screens) | 0.913 / 0.81 (ECE 0.054) | 0.927 / 0.81 (ECE 0.022) | 0.933 / 0.89 (ECE 0.023) | 0.903 / 0.91 (ECE 0.056) | 0.937 / 0.96 (ECE 0.052) | 0.923 [0.909-0.937] | 15 of 116 (13%) | 111.6 |
| V8 (ours, V7 + smoothing 0.05; superseded) | 0.910 / 0.79 (ECE 0.026) | 0.923 / 0.83 (ECE 0.024) | 0.927 / 0.91 (ECE 0.028) | 0.893 / 0.91 (ECE 0.082) | 0.937 / 0.92 (ECE 0.052) | 0.918 [0.903-0.932] | 17 of 123 (14%) | 76.7 |

Cells: accuracy / AUROC (yes-no sources) (ECE). CI: 2,000-resample item bootstrap over all 1,500 items. Errors stated at p>=0.99: wrong answers given with at least 0.99 confidence (the Solomon card's high-confidence-error column). ms: mean per request on one H100, batch 1; ours packs all questions of an image in one forward, the frozen baseline is one forward per question.
