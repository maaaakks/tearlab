# Reproducibility

The source is portable relative to the checkout root. Parent weights and output
directories are explicit training arguments; generated files go in `artifacts/`.
The calibration JSON is included despite the default data ignore rule.

Environment used for the existing results and checkout checks:

- Windows, Python3.11.9; NVIDIA RTX4070.
- PyTorch2.13.0+cu130, CUDA13.0; NumPy2.4.6; W&B0.28.2.
- Native physics configuration `data/physics_v2/calibration_r5.json`.
- CPU reference tests do not compile native code or initialize CUDA.

These are observed versions, not a claim that every supported platform/version
was tested. No clean-environment package installation has been performed.
Native CUDA/CPU build logs contain source digests. Checkpoints must have SHA256
sidecars. Training output snapshots retain sources, parent hashes, seeds and
offline W&B artifact references. Only the latest weights are bundled; ancestor
weights from the published experiments remain external inputs, so the single
bundled checkpoint does not reproduce every historical continuation result.

The full raw evidence remains in the original research checkout. This branch
contains readable protocols/results and the latest checkpoint instead of generated logs, intermediate model binaries,
copied source snapshots or machine-specific launchers. Old history is retained;
cleaning the current tree does not rewrite previous commits.
