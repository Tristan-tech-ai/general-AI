import importlib
import platform

MODS = ["torch", "torchaudio", "transformers", "soundfile", "numpy", "scipy",
        "sklearn", "pandas", "matplotlib", "tqdm", "pyloudnorm"]

print("Python", platform.python_version())
for m in MODS:
    try:
        mod = importlib.import_module(m)
        print("  OK   %-14s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("  FAIL %-14s %s: %s" % (m, type(e).__name__, e))

try:
    import torch
    print("\nCUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device         :", torch.cuda.get_device_name(0))
        print("Capability     :", torch.cuda.get_device_capability(0))
        free, total = torch.cuda.mem_get_info()
        print("VRAM free/total: %.1f / %.1f GiB" % (free / 1024**3, total / 1024**3))
        # uji matmul nyata di sm_120
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        c = (a @ a).float().sum().item()
        torch.cuda.synchronize()
        print("bf16 matmul OK :", not (c != c))  # cek bukan NaN
except Exception as e:
    print("torch check failed:", type(e).__name__, e)
