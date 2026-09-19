# Our model as the provider in Cua's jev-use example

Cua's `libs/cua-driver/examples/jev-use` runs a bounded agent: Cua Driver owns capture, candidates, execution and
verification; the provider only picks a candidate id. `s1_adapter.py` implements the same `system_one(state, questions)`
surface the example uses for TypeSafe Jev, backed by `model/serve.py`, so the example's request construction and
candidate validation are untouched.

Setup (macOS): install Cua Driver, grant Accessibility + Screen Recording, Chrome in /Applications
(strip Finder xattrs if the strict codesign check fails: `xattr -dr com.apple.FinderInfo "/Applications/Google Chrome.app"`).

```
cp s1_adapter.py <cua>/libs/cua-driver/examples/jev-use/python/
cd <cua>/libs/cua-driver/examples/jev-use && git apply <this dir>/cua_example.patch   # adds --provider s1 and verify_setup --s1
export S1_SERVER=http://127.0.0.1:8811     # tunnel to the pod running serve.py
uv run --frozen python verify_setup.py --s1 --output-dir runs/s1-demo
```

Evidence of the first run (2026-09-19, V3 checkpoint on an L40S through an ssh tunnel) is in
`results/vision/harness/jev_use_s1_demo/`: the model chose `type-verification-value` (p 0.998) then `submit-form`
(p 0.99993), the fixture server independently observed the submitted token, decisions took 249 ms and 127 ms end to end.
