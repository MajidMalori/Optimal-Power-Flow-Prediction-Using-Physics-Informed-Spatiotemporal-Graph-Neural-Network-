# Setup Guide for Collaborators

This guide walks you through setting up this project on a new machine from scratch. It covers Python environment setup, dependency installation, Weights & Biases authentication, and running the full pipeline.

---

## Prerequisites

- Python 3.10 or higher
- `pip` (comes with Python)
- `git`
- NVIDIA GPU with CUDA 12.x drivers (optional — CPU-only training is supported but slow)

---

## Step 1: Clone and Enter the Project

```bash
git clone <repository-url>
cd spatio_temporal_nn
```

---

## Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Every time you open a new terminal to work on this project, you need to activate the environment again with `source .venv/bin/activate`.

---

## Step 3: Install Dependencies

**If you have an NVIDIA GPU:**

```bash
pip install -r requirements-gpu.txt
```

**If you are on CPU only (no GPU):**

```bash
pip install -r requirements-cpu.txt
```

This installs PyTorch, PyTorch Geometric, Lightning, pandapower, W&B, and all other dependencies. It may take a few minutes.

---

## Step 4: Set Up Weights & Biases (W&B)

W&B is used to track training experiments (loss curves, hyperparameters, model comparisons). This step is required before training.

### 4.1 Create a W&B Account

If you do not already have one, go to [https://wandb.ai/authorize](https://wandb.ai/authorize) and sign up. You can use your Google or GitHub account.

### 4.2 Get Your API Key

1. Log into [wandb.ai](https://wandb.ai).
2. Go to [https://wandb.ai/authorize](https://wandb.ai/authorize).
3. You will see a long string like `a1b2c3d4e5f6...`. This is your API key. Copy it.

### 4.3 Log In from the Terminal

Run this command in your terminal (with the virtual environment activated). This command is how you "prove" who you are to W&B without needing to type a username:

```bash
wandb login
```

It will print:

```
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
```

Paste the API key you copied and press Enter. You will NOT see the key as you type — this is normal (it is hidden for security, like a password).

If successful, it will print:

```
wandb: Appending key for api.wandb.ai to your netrc file: /home/<your-user>/.netrc
```

**This only needs to be done once.** The key is saved to `~/.netrc` and all future W&B commands will use it automatically. You will not be asked again unless you delete the `.netrc` file or run `wandb login` again.

### 4.4 When Does Authentication Happen?

| Scenario | What happens | Do you need to enter a key? |
|----------|-------------|:---------------------------:|
| First time ever on this machine | `wandb login` prompts for key | Yes, once |
| Every subsequent training run | W&B reads key from `~/.netrc` automatically | No |
| Training in offline mode (default) | W&B does not contact the server at all | No |
| Running `make sync` or `wandb sync` | W&B reads key from `~/.netrc` | No |
| After deleting `~/.netrc` or on a new machine | `wandb login` prompts for key again | Yes, once |

**In short:** You paste the key once. After that, everything is automatic. No passwords, no prompts.

### 4.5 Offline vs Online Mode

By default, this project trains in **offline mode**. This means:

- W&B does **not** need an internet connection during training.
- All logs are saved locally to `logs/wandb_logs/`.
- Training runs will not appear on the W&B dashboard until you sync.

To sync your offline runs to the cloud after training:

```bash
make sync
```

If you want live logging (results appear on the W&B dashboard in real time), use the `--online` flag:

```bash
python scripts/train.py --case 33 --models all --online
```

The project is configured to log to a shared research team defined in `configs/training.yaml`:

```yaml
logger:
  project: "powerflow-pinn"
  entity: "benardmarfoadjei-knust-engineering-research-project-keep-"
```

**⚠️ IMPORTANT for Collaborators:**
If you have **not** been invited to the team yet (Section 4.7), you MUST change the `entity` to your own W&B username or leave it blank. If you don't, the code will give you an "Access Denied" error because it tries to save data to a bucket you don't own.

### 4.7 Joining the W&B Team

**You cannot add yourself to the team.** Only the team admin (the project owner) can invite you. Ask them to:

1. Go to [wandb.ai](https://wandb.ai) → **Team Settings** (the team name is shown in the `entity` field above).
2. Click **Invite new member**.
3. Enter your email address or W&B username and send the invitation.
4. You will receive an email with a link to accept the invitation.

Once you accept, your training runs will automatically appear under the shared team dashboard when you use online mode or sync your offline runs.

---

## Step 5: Verify the Setup

Run the sanity check. This runs a minimal version of the full pipeline (24 timesteps, 1 epoch) for all three power system cases in parallel:

```bash
bash verify_setup.sh
```

This will:
1. Generate data using pandapower (Case 33, 57, 118).
2. Preprocess and normalize the data.
3. Train all 7 model architectures for 1 epoch each.
4. Run benchmark evaluation.
5. Run uncertainty analysis.

If everything is set up correctly, you will see `SANITY CHECK COMPLETE` at the end. The full run takes approximately 5–15 minutes depending on hardware.

---

## Step 6: Run a Full Training

Once the sanity check passes, you can run full training:

```bash
# Generate full dataset (10,008 timesteps per case)
python scripts/generate_data.py --case all --timesteps 10008

# Preprocess
python scripts/preprocess_data.py --case all

# Train all models on Case 33
python scripts/train.py --case 33 --models all --epochs 200
```

Or use the parallel pipeline to run everything at once:

```bash
bash run_pipeline.sh
```

See `README.md` for the full list of Makefile targets and configuration options.

---

## Troubleshooting

### `wandb: ERROR Run directory not writable`

Make sure the `logs/wandb_logs/` directory exists and is writable:

```bash
mkdir -p logs/wandb_logs
```

### `ModuleNotFoundError: No module named 'torch_geometric'`

PyTorch Geometric requires specific PyTorch version compatibility. Reinstall it:

```bash
pip install torch-geometric==2.7.0
```

### `wandb: Network error (ConnectionError), entering offline mode`

This is fine. If you are training in offline mode (the default), W&B does not need internet. Your runs are saved locally. Sync them later with `make sync`.

### `RuntimeError: CUDA out of memory`

Reduce the batch size in `configs/training.yaml`:

```yaml
data:
  batch_size: 16   # Was 32
```

### `pandapower` convergence warnings

These are expected for some edge cases at extreme renewable penetration levels. The data generator handles non-convergence automatically (it retries with curtailment or skips the timestep).

---

## Summary of Key Commands

| Command | What it does |
|---------|-------------|
| `source .venv/bin/activate` | Activate the Python environment |
| `wandb login` | One-time W&B authentication |
| `bash verify_setup.sh` | Verify the full pipeline works |
| `bash run_pipeline.sh` | Run the full parallel pipeline |
| `make train-33` | Train all models on Case 33 |
| `make eval-33` | Evaluate trained models on Case 33 |
| `make sync` | Upload offline W&B logs to the cloud |
| `make clean-all` | Delete all generated data, logs, and reports |
