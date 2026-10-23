# onnx2ckpt.py <policy.onnx> <template_ckpt.pt> <out.pt> [--check-onnx <our.onnx> <our_ckpt.pt>]
# rebuilds an rsl_rl checkpoint from a Pollen-exported ONNX actor (Sub mean, Div std+eps, 4 Gemm + Elu).
# actor weights + obs normalizer come from the ONNX; critic/optimizer/iter come from the template (unused at play;
# for a BC expert only actor_state_dict is read). eps is measured from one of OUR exports vs its own checkpoint.
import sys, torch, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper

def onnx_inits(p):
    m = onnx.load(p)
    return {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

src, tmpl, out = sys.argv[1:4]
eps = 1e-2
if "--check-onnx" in sys.argv:
    i = sys.argv.index("--check-onnx")
    our_onnx, our_ckpt = sys.argv[i + 1], sys.argv[i + 2]
    I = onnx_inits(our_onnx)
    div = [v for k, v in I.items() if "Div" in k][0]
    std = torch.load(our_ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]["obs_normalizer._std"].numpy()
    d = div - std
    eps = float(np.median(d))
    print(f"eps measured on our export: median {eps:.6g}, spread {d.min():.3g}..{d.max():.3g}")
I = onnx_inits(src)
div = [v for k, v in I.items() if "Div" in k][0]
c = torch.load(tmpl, map_location="cpu", weights_only=False)
a = c["actor_state_dict"]
std = torch.tensor(div - eps, dtype=a["obs_normalizer._std"].dtype)
a["obs_normalizer._mean"] = torch.tensor(I["obs_normalizer._mean"], dtype=a["obs_normalizer._mean"].dtype)
a["obs_normalizer._std"] = std
a["obs_normalizer._var"] = std ** 2
for k in ["mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias", "mlp.4.weight", "mlp.4.bias", "mlp.6.weight", "mlp.6.bias"]:
    assert tuple(I[k].shape) == tuple(a[k].shape), k
    a[k] = torch.tensor(I[k], dtype=a[k].dtype)
c["actor_state_dict"] = a
torch.save(c, out)
# equivalence check: torch forward of the rebuilt actor vs onnxruntime on random obs
x = np.random.default_rng(0).normal(size=(256, 61)).astype(np.float32)
sess = ort.InferenceSession(src)
y_onnx = np.concatenate([sess.run(None, {"obs": x[j:j + 1]})[0] for j in range(len(x))])
h = (torch.tensor(x) - a["obs_normalizer._mean"]) / (a["obs_normalizer._std"] + eps)
for li in (0, 2, 4):
    h = torch.nn.functional.elu(h @ a[f"mlp.{li}.weight"].T + a[f"mlp.{li}.bias"])
y = (h @ a["mlp.6.weight"].T + a["mlp.6.bias"]).numpy()
print(f"rebuilt {out}: max |torch - onnx| = {np.abs(y - y_onnx).max():.2e} over 256 random obs")
