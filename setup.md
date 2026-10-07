# MIL-Adapter setup on Rorqual (Compute Canada / Digital Research Alliance)

Repo: https://github.com/cvblab/MIL-Adapter (Meseguer, del Amor, Naranjo — MedIA'26)

## 1. Environment installation

Alliance clusters use `module` + `virtualenv`, not conda, and compute nodes have **no internet access** — only login nodes do. All package installs happen on a login node.

```bash
ssh <username>@rorqual.alliancecan.ca

# work in your project space, not $HOME, since the repo + a venv can get large
cd ~/projects/def-<pi>/$USER   # replace def-<pi> with your allocation
git clone https://github.com/cvblab/MIL-Adapter.git
cd MIL-Adapter

module load python/3.11 cuda
module load gcc arrow/25.0.0
# check `module avail python` for exact versions on Rorqual              
 # check `module avail cuda`; needed if torch build requires it as a separate module
```

Modules to load:
```bash
module load python/3.11 cuda
module load gcc arrow/25.0.0
```

**Check wheel availability BEFORE creating/activating the venv.** `avail_wheels` uses `pip`'s internal API, and it breaks (`ImportError: cannot import name 'BuildEnvironment' from 'pip._internal.build_env'`) if it picks up a `pip` that's been upgraded inside an activated venv instead of the system pip from the loaded module. Run these checks first, in a clean shell, before touching the venv:

```bash
avail_wheels "torch*"
avail_wheels "torchvision*"
avail_wheels "torchaudio*"
avail_wheels -r requirements.txt
```

> **Troubleshooting**: if you hit the `BuildEnvironment` ImportError, it means `avail_wheels` picked up a mismatched pip (usually because a venv with an upgraded pip is active). Fix: `deactivate` (or open a fresh shell), confirm `which avail_wheels` resolves outside your venv, and re-run. If it persists, `module unload python && module load python/3.11` to reset PATH, then try again.

Now create the venv and install:

```bash
virtualenv --no-download ~/envs/mil-adapter
source ~/envs/mil-adapter/bin/activate
pip install --no-index --upgrade pip
```

**PyTorch**: the repo pins `torch==2.1.0+cu121` via the PyTorch cu121 wheel index, but Alliance clusters ship their own pre-built wheels optimized for their GPUs/interconnect, and that exact cu121 build is unlikely to be in the local wheelhouse. Install from the wheelhouse (checked above) rather than pinning the exact PyPI version:

```bash
pip install --no-index torch torchvision torchaudio
```

If the code depends on behavior specific to torch 2.1.0, check compatibility after install (`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`) — run this inside an interactive GPU job (`salloc --gpus-per-node=1 --time=0-00:30 --account=def-<pi>`), since login nodes have no GPU.

**Remaining dependencies** (`requirements.txt`: matplotlib, pandas, seaborn, scikit-learn, tqdm, transformers, datasets, huggingface_hub==0.26.5):

```bash
pip install --no-index -r requirements.txt
```

Manual install

```bash
pip install --no-index matplotlib pandas seaborn scikit-learn tqdm transformers "huggingface_hub==0.26.5"
pip install --no-index datasets
pip install --no-index --no-build-isolation datasets
```

Anything `avail_wheels` didn't cover earlier, install normally from PyPI on the login node (internet is available there):

```bash
pip install <missing-package>
```

Check install
```bash
module load python/3.11 cuda gcc arrow/25.0.0
source ~/envs/mil-adapter/bin/activate

pip check   # flags any broken/missing dependency chains

python - <<'EOF'
import importlib

packages = {
    "torch": None,
    "torchvision": None,
    "torchaudio": None,
    "matplotlib": None,
    "pandas": None,
    "seaborn": None,
    "sklearn": "scikit-learn",   # import name differs from package name
    "tqdm": None,
    "transformers": None,
    
    "huggingface_hub": None,
    "pyarrow": None,             # confirms the Arrow module is providing it correctly
    "datasets": None
}

for import_name, display_name in packages.items():
    label = display_name or import_name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"OK   {label:<16} {version}")
    except Exception as e:
        print(f"FAIL {label:<16} {e}")
EOF
```

## 2. Data download

MIL-Adapter's pre-extracted patch embeddings are distributed via a SharePoint folder link in the README (manual/browser download, no public API — it's a `:f:` SharePoint folder share, which requires the interactive web app to zip and serve the folder; a plain `wget`/`curl` against that URL just returns the HTML shell, not the data). Compute nodes have no internet either way, so the transfer has to land on Rorqual via a login node or Globus. Recommended path:

1. Download the embeddings locally on your own machine from the SharePoint link (browser) into one local folder, e.g. `~/mil-adapter-data/`.
2. Transfer to Rorqual with **Globus** (recommended for large transfers — multi-GB embedding sets):
   - Install **Globus Connect Personal** on your own machine (globus.org/globus-connect-personal) and sign in with your Globus account (Alliance accounts are a supported identity provider — log in via "Digital Research Alliance of Canada" / your home institution in the Globus login screen).
   - In Globus Connect Personal, add `~/mil-adapter-data/` (or its parent) to the list of folders it's allowed to share, then start it so your machine appears as a Globus endpoint (e.g. named after your computer).
   - Go to the Globus web app (app.globus.org) → **File Manager**.
   - In the left panel, set the collection to your own machine's Globus Connect Personal endpoint and navigate to `~/mil-adapter-data/`.
   - In the right panel, set the collection to **`rorqual.alliancecan.ca`** (Rorqual's Globus endpoint — search for "Rorqual" in the collection search) and navigate to your destination, e.g. `~/projects/def-<pi>/$USER/data/`.
   - Select the files/folders on the left, click **Start** to transfer left → right. Globus runs the transfer in the background (resumable, checksum-verified), and emails you when it completes — you can close the browser tab.
   - Check progress any time under **Activity** in the Globus web app, or `globus task list` / `globus task show <task-id>` if you have the Globus CLI installed.
3. For smaller/quick transfers, `rsync`/`scp` to a login node works too:
   ```bash
   rsync -avP "/mnt/c/Users/natgi/Desktop/PhD project/Baselines/Datasets/DDBB_MILAdapter.zip" natgill@rorqual.alliancecan.ca:/project/rrg-josedolz/natgill/data
   ```
4. Store the data under `$PROJECT` (persistent, backed up) rather than `$HOME`. Use `$SCRATCH` only for transient checkpoints/logs (auto-purged, ~60 days). At job start, if I/O bound, copy the working subset to `$SLURM_TMPDIR` for best performance.

Expected layout per the repo's `--folder` argument: `<folder>/<project>/<encoder>/*.npy`, e.g. `data/NSCLC/CONCH/`.

## 3. Running a job

Check imports
```batch
python -c "
from utils.adapters import ZSMIL, TaskRes, CLIPAdapter, TIPAdapter
from utils.trainer import validate_model, train_model
from utils.utils import set_random_seeds, plot_confmx, get_project_data, load_data, fewshot_sampling
print('All imports OK')
"
```

Sbatch job:
```bash
#!/bin/bash
#SBATCH --account=rrg-josedolz
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

module load python/3.11 cuda
module load gcc arrow/25.0.0
source ~/envs/mil-adapter/bin/activate

cd /project/rrg-josedolz/natgill/baselines/MIL-Adapter

python main.py --folder /project/rrg-josedolz/natgill/data \
  --project NSCLC \
  --encoder CONCH \
  --aggregator ABMIL \
  --adapter TaskRes \
  --init ZS \
  --k_shots 4 \
```

Submit with `sbatch job.sh`; check GPU type names available on Rorqual with `sinfo` or `/opt/software/slurm/...` docs, since specifiers vary by cluster (e.g. `h100`, `a100`, `l40s`).

## Check job progress

Check if it's queued/running:
```bash
squeue -u $USER
```

Add --start to see the estimated start time if it's still queued:
```bash
squeue -u $USER --start
```

Watch live output while it runs — your #SBATCH --output=%x-%j.out writes stdout/stderr to a file named <job-name>-<jobid>.out in the submission directory:
```bash
ls *.out                
tail -f few_shot-<jobid>.out
```

Check GPU utilization while it's running (useful given the whole job depends on that GPU actually being used):
```bash
srun --jobid=<jobid> --pty nvidia-smi
```

After it finishes (or if it disappears from squeue unexpectedly), check what actually happened:
```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
```
State tells you COMPLETED vs FAILED/TIMEOUT/OUT_OF_MEMORY, Elapsed tells you real runtime (useful for right-sizing --time next round), and MaxRSS tells you actual peak memory used (useful for right-sizing --mem given the whole-dataset-in-RAM loading we flagged).
If it fails, the .out file will usually have the Python traceback — that's the first thing to paste back here if something goes wrong.

To cancel it if needed: 
```bash
scancel <jobid>.
```

## Open items to confirm on Rorqual specifically
- Exact `module avail python` / `module avail cuda` versions (not verified live — alliancecan.ca docs pages returned bot-blocked errors during research).
- Rorqual's GPU type and correct `--gpus-per-node` specifier.
- Whether the SharePoint embeddings download works via any programmatic method, or is strictly manual.
- Exact Globus endpoint name for Rorqual (confirm in app.globus.org's collection search — Alliance clusters sometimes list endpoints per site/cluster, e.g. under Calcul Québec).

## Troubleshooting log
- `avail_wheels` → `ImportError: cannot import name 'BuildEnvironment' from 'pip._internal.build_env'`: caused by running `avail_wheels` after upgrading `pip` inside an activated venv. Fix: run `avail_wheels` checks before creating/activating the venv (now reflected in step 1 above), or `deactivate` and reset via `module unload python && module load python/3.11` if it happens again.