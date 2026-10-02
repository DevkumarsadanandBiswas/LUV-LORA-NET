from pathlib import Path
def write_readme(out):
    out = Path(out); s = out / "summary_all.md"; L = []
    L.append("# Lightweight Unified U-Net-ViT + LoRA + Data Consistency: multi-modal 2D/3D reconstruction benchmark\n")
    L.append("Forward model y = M.FFT(x) + noise (random phase-encode lines, 30% kept, 8% centre, sigma 0.02), simulated for every dataset (no paired raw k-space exists for X-ray/ultrasound). Identical masks/noise for every model. Metrics on the held-out test split only.\n")
    L.append("Files: summary_all.{md,csv,tex} | win_table.csv | stats_ours_vs_baselines.csv | efficiency_table.{md,csv} | comparison_bar.png | global_metric_heatmap.png | per_model_across_datasets.png | ours_vs_best_baseline.png | flops_chart.png | time_per_scan_chart.png | params_chart.png | psnr_vs_cost.png | global_qualitative_comparison.png | <dataset>/comparison_all_models_*.png | <dataset>/boxplot_metrics.png | <dataset>/training_curves_all_models.png | <dataset>/<model>/<dataset>_<model>_qualitative.png | <dataset>/ablation/*\n")
    L.append("FLOPs = torch FlopCounterMode (conv/matmul/attention, 2 x MAC) + analytic 5 N log2 N per FFT call, batch 1, one scan. Time = median of 30 forward passes after 5 warm-ups, batch 1, CUDA-synchronised, model's native precision.\n")
    L.append("## Results\n"); L.append(s.read_text() if s.exists() else "_not available_"); L.append("\n![bar](comparison_bar.png)\n")
    out.mkdir(parents=True, exist_ok=True); (out / "README.md").write_text("\n".join(L))
