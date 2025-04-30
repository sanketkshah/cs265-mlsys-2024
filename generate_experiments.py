import itertools
from typing import Dict, List

model_names: List[str] = [
    "torchbenchmark.models.hf_Bert.Model",
    "Resnet18",
    "Resnet50",
]

model_batch_sizes: Dict[str, List[int]] = {
    "torchbenchmark.models.hf_Bert.Model": [10 * i for i in range(1, 11)]
    + [2**i for i in range(13)],
    "Resnet18": [500 * i for i in range(1, 11)] + [2**i for i in range(13)],
    "Resnet50": [100 * i for i in range(1, 11)] + [2**i for i in range(13)],
}

parameters_to_sweep = {
    ("model_names", "batch_size"): [
        (model_name, batch_size)
        for model_name in model_names
        for batch_size in model_batch_sizes[model_name]
    ],
}

if __name__ == "__main__":
    experiments = []
    for ((model_name, batch_size),) in itertools.product(*parameters_to_sweep.values()):
        experiment_name = (
            f"results/log_{model_name}_{batch_size}.txt"  # add _no_ac for no ac
        )

        experiments.append(
            f"python benchmarks.py "
            f"--model={model_name} "
            f"--batch_size={batch_size} "
            f"> {experiment_name}"
        )

    with open("experiments.txt", "w") as f:
        for experiment in experiments:
            f.write(experiment + "\n")
