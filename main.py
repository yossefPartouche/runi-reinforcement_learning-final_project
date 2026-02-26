from datetime import timedelta
from typing import Dict

import torch

from src.experiments import calibration_experiments, exp_set_1
from src.experiment_runner import Experiment
from src.utils import analyze_inference, plot_training_curves, save_experiment_report, set_random_seed, get_device, plot_milestone_progress


def run_single_experiment(config: Dict, exp_name: str, device: torch.device) -> None:
    """Runs one full experiment according to config"""
    print(f"--- Starting Experiment: {exp_name} ---")

    set_random_seed(config["seed"])
    
    exp = Experiment(config=config, exp_name=exp_name, device=device)

    # training (note @timer decorator on train(), adds runtime to output)
    train_metrics, train_time = exp.train()
    plot_training_curves(log_dir=exp.results_dir)
    plot_milestone_progress(log_dir=exp.results_dir)

    # inference (note @timer decorator on evaluate(), adds runtime to output)
    inference_metrics, inference_time = exp.evaluate()
    analyze_inference(log_dir=exp.results_dir)

    # collect training + inference metrics
    experiment_metrics = train_metrics | inference_metrics 
    timings = {
        "train": str(timedelta(seconds=int(train_time))),
        "inference": str(timedelta(seconds=int(inference_time))),
    }

    # genereate experiment report
    save_experiment_report(
        log_dir=exp.results_dir, 
        config=config, 
        metrics=experiment_metrics,
        timings=timings
    )
    
    print(f"--- Finished: {exp_name} ---\n")


def main():

    device = get_device()

    # define exp set:
    experiments = calibration_experiments
    for exp in experiments:
        run_single_experiment(exp["config"], exp["name"], device=device)


if __name__ == "__main__":
    main()