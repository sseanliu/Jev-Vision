| system | skip acc (pos/neg) | effect acc (pos/neg) | done acc (pos/neg) | ECE (skip/effect/done) | AUROC | sel@90 | ms |
|---|---|---|---|---|---|---|---|
| Jev 1.13 (text only, candidate table) | 0.870 (0.55/0.92) n=300 | 0.733 (0.66/0.98) n=300 | 0.887 (0.89/0.89) n=300 | 0.105/0.096/0.049 | 0.902/0.904/0.936 | 0.900/0.774/0.907 | 238 (API p50, serial, one question) |
| Claude Sonnet 5 (screenshots, verbal probability) | 0.850 (0.31/0.93) n=200 | 0.477 (0.33/0.94) n=130 | 0.780 (0.35/0.93) n=200 | 0.119/0.301/0.105 | 0.769/0.721/0.819 | 0.867/0.521/0.806 | n/a |
| V5b (ours, 8B, screenshots + candidate table) | 0.936 (0.79/0.96) n=807 | 0.967 (0.97/0.96) n=390 | 0.896 (0.83/0.92) n=806 | 0.042/0.020/0.055 | 0.972/0.991/0.956 | 0.972/0.989/0.932 | 157.0 |
| rules: URL/DOM change for effect, constant no for skip/done | 0.865 (0.00/1.00) n=807 | 0.892 (0.86/1.00) n=390 | 0.758 (0.00/1.00) n=806 | 0.035/0.008/0.142 | 0.500/0.930/0.500 | 0.866/0.883/0.756 | 0.0 |

Pixel track (same episodes, raw screenshot + one marker, no candidate table):

| system | skip acc (pos/neg) | effect acc (pos/neg) | done acc (pos/neg) | ECE (skip/effect/done) | AUROC | sel@90 | ms |
|---|---|---|---|---|---|---|---|
| V5b, pixel track (zero-shot: raw screenshot + marker, no table) | 0.931 (0.89/0.94) n=807 | 0.908 (0.97/0.69) n=390 | 0.902 (0.88/0.91) n=806 | 0.039/0.080/0.046 | 0.972/0.978/0.958 | 0.963/0.946/0.941 | 182 |
