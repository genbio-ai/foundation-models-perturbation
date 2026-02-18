This folder contains code to perform embedding fusion. To run fusion on the various datasets presented in the benchmark, use the following commands:

```
python main.py --config-name essential_lfc_full +model_name="essential_lfc_full"
python main.py --config-name essential_lfc_simple +model_name="essential_lfc_simple"
python main.py --config-name essential_deg_simple +model_name="essential_deg_simple"
python main.py --config-name tahoe_deg_simple +model_name="tahoe_deg_simple"
python main.py --config-name sciplex_deg_simple +model_name="sciplex_deg_simple"
```

By default this runs fold 0 across all cell lines. You can modify the config files in `config/` or pass config variables in command-line (Hydra handles argument parsing)