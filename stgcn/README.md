# ST-GCN Implementation

The ST-GCN is kept in a separate directory because its input tensor has a fundamentally different shape from the other models in this project:

- Other models (SVM, 1D-CNN, BiLSTM): input shape `(batch, timesteps, channels)` = `(N, 100, 30)`
- ST-GCN: input shape `(batch, nodes, features, timesteps)` = `(N, 5, 6, 36)`

The ST-GCN treats each IMU sensor as a graph node (5 nodes: ankle, pocket, waist, neck, wrist) with 6 features per node (3-axis accelerometer + 3-axis gyroscope). The spatial relationships between nodes are encoded by three adjacency matrices following the spatial-configuration partitioning scheme of Yan et al. (2018).

The ST-GCN uses 2-second windows (36 timesteps at 18 Hz), whereas the other models use 5.6-second windows. This follows the window-size ablation of Yan et al. (2023), who identified 2 seconds as optimal for this architecture on the UP-Fall dataset.

## Files

- `monique.py` — ST-GCN training under 15-fold LOSO cross-validation. Reports both 11-class activity classification accuracy and binary fall/non-fall metrics.
- `moniquedataprep.py` — graph-structured windowing that reads the raw UP-Fall CSV and produces `(N, 5, 6, 36)` tensors for the ST-GCN.

## Running

Requires the raw UP-Fall CSV in `../data/data.csv`.

```bash
python moniquedataprep.py   # generates graph-structured windowed data
python monique.py           # runs 15-fold LOSO and saves loso_results_*.json
```

## Output

Results are saved as `loso_results_YYYYMMDD_HHMMSS.json` in the ST-GCN output directory. This file contains per-subject accuracy (11-class and binary) along with the full classification report and confusion matrix.