# Diagnostic only: why does MinkowskiEngine's setup.py fail on this image?
# Run setup.py DIRECTLY so the traceback is not buried under pip's caller boilerplate.
import subprocess, sys, torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("python", sys.version)

def sh(label, cmd, cwd=None):
    print(f"\n{'='*70}\n[{label}]\n{'='*70}", flush=True)
    r = subprocess.run(["bash","-lc",cmd], capture_output=True, text=True, cwd=cwd)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    print(out[-4000:] if out else "(no output)")
    print(f"--- rc={r.returncode}")
    return r.returncode

sh("clone", "rm -rf /tmp/ME && git clone -q --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git /tmp/ME && echo cloned")
sh("setup.py head", "sed -n '40,80p' /tmp/ME/setup.py")
sh("setup.py egg_info DIRECT (torch visible)", "cd /tmp/ME && python setup.py egg_info 2>&1 | tail -40")
sh("openblas", "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y libopenblas-dev >/dev/null 2>&1; echo ok")
rc = sh("build --no-build-isolation", "cd /tmp/ME && MAX_JOBS=2 python -m pip install --no-build-isolation . 2>&1 | tail -45")
if rc == 0:
    sh("import check", "python -c 'import MinkowskiEngine as ME; print(\"ME\", ME.__version__)'")
