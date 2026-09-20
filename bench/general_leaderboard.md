| system | POPE | MME | NLVR2 (2 images) | A-OKVQA (4-way) | Food-101 (20-way) | mean acc | ms |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct frozen, prompt + logit readout | 0.900 / 0.96 (ECE 0.072) | 0.903 / 0.97 (ECE 0.076) | 0.897 / 0.96 (ECE 0.085) | 0.903 (ECE 0.084) | 0.950 (ECE 0.037) | 0.911 | 185.1 |
| V3 (ours, trained on web + desktop grounding) | 0.860 / 0.67 (ECE 0.085) | 0.883 / 0.70 (ECE 0.044) | 0.917 / 0.78 (ECE 0.076) | 0.890 / 0.90 (ECE 0.079) | 0.937 / 0.91 (ECE 0.045) | 0.897 | 263.9 |
| V5b (ours, trained on screen state judgments) | 0.860 / 0.73 (ECE 0.094) | 0.890 / 0.77 (ECE 0.033) | 0.917 / 0.79 (ECE 0.016) | 0.890 / 0.90 (ECE 0.080) | 0.937 / 0.91 (ECE 0.046) | 0.899 | 268.9 |

Cells: accuracy / AUROC (yes-no sources) (ECE). ms: mean per request on one H100, batch 1; ours packs all questions of an image in one forward, the frozen baseline is one forward per question.
