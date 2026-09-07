# Lightning.AI Runbook — run the skip-robust co-train on free GPU

> The cheapest-decisive next experiment (E87 continuation). Fits the free tier (LoRA on a 1.5B base
> runs on a 16GB T4). ~hours. Pre-registration: `cotrain_skiprobust_preregistration.yaml`.

## 0. One-time: get the code onto Lightning

On your Mac, push the branch (it's 4 commits ahead of origin/main):
```bash
cd /Users/christienantonio/aurelius
git push origin spike/first-light-readiness
```
On the Lightning Studio (a fresh GPU Studio), in the terminal:
```bash
git clone https://github.com/Aurelien033/Aurelius.git && cd Aurelius
git checkout spike/first-light-readiness
pip install -q torch transformers peft datasets accelerate safetensors numpy
huggingface-cli login            # only if a dataset/model needs auth; Qwen base is open
```

## 1. Environment smoke (validate the Studio works) — ~2 min
```bash
python docs/training/training_readiness_spike.py
```
Expect: loss drops ~10.9 → ~7 (it tokenizes wikitext; downloads gpt2 tokenizer). If this passes, the
training stack works on the GPU.

## 2. Co-train smoke gate (REQUIRED before the budgeted run) — ~5 min
```bash
python docs/training/cotrain_skiprobust.py --smoke
```
Expect: `print_trainable_parameters` shows ~0.5–1% trainable, `decoder layers located: 28`, and loss
dropping over 20 steps under mixed skip-k. If loss does NOT drop, STOP — fix before spending GPU.

## 3. The real co-train — ~hours on free GPU
```bash
python docs/training/cotrain_skiprobust.py --steps 2000 --out checkpoints/cotrain-skiprobust
```
Adapters + `cotrain_result.json` land in `checkpoints/cotrain-skiprobust/`. Push or download them.

## 4. Eval = the E87 chessboard on the co-trained model (the actual result)
The gym + E87 split are regenerable (deterministic) OR upload them (~12 MB). To regenerate the clean
families used by the eval:
```bash
# F2 (gym-v0.1-FL) + F3 (gym-v0.3) build scripts live in the Research folder; copy them in, or upload
# ~/Desktop/"AI:ML Research"/{gym-v0.1-FL,gym-v0.3,e87_receipt/e87_split.json} to the Studio.
```
Then run the chessboard with the adapters loaded (small edit to e87_chessboard.py: after loading the
base, `model = PeftModel.from_pretrained(model, "checkpoints/cotrain-skiprobust")`). Compare to the
frozen-base E87 numbers (dense 73 / random 30 / learned 20 / entropy 2). **H-CT-0/1/2** are decided by
whether the dense-minus-skip cost shrinks and whether routing now beats random.

## What to bring back
`cotrain_result.json` + the chessboard analysis JSON. Hand them to me with "verify" — same re-score
discipline as every prior run (independent reproduction before any record updates).

## Notes
- GPU fit: LoRA + 1.5B bf16 ≈ 3–4 GB weights + adapters + activations → fits 16 GB (T4). If you have
  an A10G/L4/A100, raise `--batch` (8–16) and it finishes faster.
- 100M from-scratch (`configs/config_100m.yaml`) is a separate, bigger run — do the co-train FIRST
  (it answers whether skip-native training is even worth it).

## 4b. Eval is now ONE command (pre-built, no hand-editing)
```bash
python docs/training/eval_cotrained.py --adapters checkpoints/cotrain-skiprobust --k 1 2 4
```
Loads the base + your LoRA adapters, runs the held-out E87 test set, and prints CO-TRAINED ceiling/avg
at each k next to the FROZEN-base numbers. Writes `cotrain_eval.json`. The decisive comparisons:
- **H-CT-0 (skipping got cheaper):** co-trained avg-random-skip HIGHER than frozen at the same k.
- **H-CT-2 (no capability loss):** co-trained dense ≈ frozen 73%.
Bring `cotrain_result.json` + `cotrain_eval.json` back with "verify".
Note: the gym + e87_split must be present on Lightning (regenerate via the build scripts or upload ~12MB).
