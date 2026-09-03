import json, math

LAT  = "results/statebridge_gpqa_8B_20260821_103752.json"
SOLO = "results/statebridge_gpqa_8B_20260824_105733.json"

def key(r): return r.get("global_idx", r.get("idx"))
def load(p): return {key(r): r for r in json.load(open(p))["results"]}
def judger(r): return next((t for t in reversed(r["trace"]) if t["role"] == "judger"), {})
def done(r):   return bool(judger(r).get("gen_info", {}).get("hit_eos"))

lat, solo = load(LAT), load(SOLO)
common = sorted(set(lat) & set(solo))
print(f"items in both: {len(common)}\n")

for name, D in [("4-agent latent", lat), ("solo judger", solo)]:
    rs = [D[i] for i in common]
    n, c = len(rs), sum(r["correct"] for r in rs)
    fin  = sum(done(r) for r in rs)
    cc   = sum(r["correct"] for r in rs if done(r))
    print(f"{name:>15}: scored {c}/{n} = {c/n:5.1%} | completed {fin}/{n} = {fin/n:5.1%} "
          f"| among completed {cc}/{fin} = {cc/max(fin,1):5.1%}")

def mcnemar(idxs, label):
    b = sum(1 for i in idxs if lat[i]["correct"] and not solo[i]["correct"])
    c = sum(1 for i in idxs if solo[i]["correct"] and not lat[i]["correct"])
    print(f"\n{label} ({len(idxs)} items)")
    la = sum(lat[i]["correct"] for i in idxs); so = sum(solo[i]["correct"] for i in idxs)
    print(f"  latent {la}/{len(idxs)} = {la/len(idxs):.1%}   solo {so}/{len(idxs)} = {so/len(idxs):.1%}")
    print(f"  discordant: latent-only {b}, solo-only {c}")
    if b + c:
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        print(f"  McNemar chi2 = {chi:.3f}, p ~ {math.erfc(math.sqrt(chi/2)):.3f}")

mcnemar(common, "ALL ITEMS, paired")
mcnemar([i for i in common if done(lat[i]) and done(solo[i])], "BOTH ARMS COMPLETED")