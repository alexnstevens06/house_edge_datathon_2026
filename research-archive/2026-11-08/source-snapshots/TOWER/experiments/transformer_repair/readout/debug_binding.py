import os, sys
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ["HE_POS"] = "lag"
READOUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(READOUT, "..", "source", "src"))
sys.path.insert(0, READOUT)
import numpy as np, torch
import test_readout as T
from house_edge.data import Episode
from house_edge.windows import Example, WindowCollator

for arch in ("dense", "sparse"):
    model, loss, acc, episodes = T.fit_toy(arch, "self")
    print(f"== {arch} loss={loss:.4f} acc={acc:.3f}")
    ep = episodes[0]
    print("phase rows t=0,1,2:", ep.obs[:3].tolist())
    print("machine_types:", ep.machine_types.tolist())
    t, coll = 70, WindowCollator(episodes, arch, T.WINDOW)
    model.eval()
    with torch.no_grad():
        base = torch.softmax(model.forward(coll([Example(0, t, None)])), -1)[0]
    D = np.zeros((8, 8))
    for c in range(8):
        obs2 = ep.obs.copy()
        obs2[:t, c] = (obs2[:t, c].astype(np.int64) + 2) % 5
        ep2 = Episode(ep.episode_id, ep.machine_ids, ep.machine_types, obs2, ep.cutoff)
        with torch.no_grad():
            pert = torch.softmax(model.forward(
                WindowCollator([ep2], arch, T.WINDOW)([Example(0, t, None)])), -1)[0]
        D[:, c] = (pert - base).abs().sum(-1).numpy()
    print("delta[row, perturbed_col]:")
    print(np.array2string(D, precision=2, suppress_small=True))
    print("argmax per row:", D.argmax(1).tolist(), " max per col:", D.max(0).round(2).tolist())
