To run the benchmarks and save results for all combinations of estimators and embeddings, you can use the following commands, which will require you to run `pip install hydra-core`.

Benchmarking can be parallelized using Hydra's launchers (e.g. joblib or ray).

```bash
# Essential DEG
python bench_essential_deg.py --config-name config_essential_deg --multirun
python bench_essential_deg.py --config-name config_essential_deg_baseline --multirun

# Essential LFC
python bench_essential_lfc.py --config-name config_essential_lfc --multirun
python bench_essential_lfc.py --config-name config_essential_lfc_baseline --multirun

# Norman LFC
python bench_norman_lfc.py --config-name config_norman_lfc --multirun
python bench_norman_lfc.py --config-name config_norman_lfc_baseline --multirun

# Sciplex DEG
python bench_sciplex_deg.py --config-name config_sciplex_deg --multirun
python bench_sciplex_deg.py --config-name config_sciplex_deg_baseline --multirun

# Sciplex LFC
python bench_sciplex_lfc.py --config-name config_sciplex_lfc --multirun
python bench_sciplex_lfc.py --config-name config_sciplex_lfc_baseline --multirun

# Tahoe DEG
python bench_tahoe_deg.py --config-name config_tahoe_deg --multirun
python bench_tahoe_deg.py --config-name config_tahoe_deg_baseline --multirun

# Tahoe LFC
python bench_tahoe_lfc.py --config-name config_tahoe_lfc --multirun
python bench_tahoe_lfc.py --config-name config_tahoe_lfc_baseline --multirun
```