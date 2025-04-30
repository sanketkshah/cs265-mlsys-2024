import argparse
import importlib
import statistics as stats
from datetime import datetime
from typing import Any, List

import pandas as pd
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchbenchmark.util.model import BenchmarkModel
from torchvision.models import resnet18, resnet50

from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile

torch.manual_seed(0)  # Set seed for reproducibility

model_names: List[str] = [
    "torchbenchmark.models.hf_Bert.Model",
    "Resnet18",
    "Resnet50",
]


class Experiment:
    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        assert model_name in model_names, (
            f"Model {model_name} not found in model names {model_names}"
        )
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size

        if self.model_name == "torchbenchmark.models.hf_Bert.Model":
            pos = model_name.rfind(".")
            module = importlib.import_module(model_name[:pos])
            model_class = getattr(module, model_name[(pos + 1) :])

            model: BenchmarkModel = model_class(
                "train", "cuda", batch_size=batch_size, extra_args=extra_args
            )
            self.model: nn.Module = model.model
            self.example_inputs = model.example_inputs

            def bert_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = model(**example_inputs).loss
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = bert_train_step
            self.optimizer: optim.Optimizer = model.optimizer

        elif self.model_name in ["Resnet18", "Resnet50"]:
            inp = torch.randn(self.batch_size, 3, 224, 224, device=dev)
            num_classes = 10
            target = torch.randint(0, num_classes, (self.batch_size,), device=dev)
            self.example_inputs = (inp, target)
            with torch.device(dev):
                self.model = resnet18() if self.model_name == "Resnet18" else resnet50()

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-2, fused=True, capturable=True
            )
            self.train_step = resnet_train_step

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def init_opt_states(self):
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        # Build the graph profiler
        warm_up_iters, profile_iters = 2, 3
        graph_profiler = GraphProfiler(gm)

        # Perform static analysis of the graph
        with torch.no_grad():
            for _ in range(warm_up_iters):
                graph_profiler.run(*args)
            graph_profiler.reset_stats()

            for _ in range(profile_iters):
                graph_profiler.run(*args)
            graph_profiler.aggregate_stats()
            graph_profiler.print_stats(
                to_file=f"results/graph_prof_before_checkpoint_{self.model_name}_{self.batch_size}.txt"
            )

        # Use the static data to decide which nodes to checkpoint
        # Get the memory limit from CUDA device
        mem_limit = (
            torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            // 4
        )

        print(f"Memory limit: {mem_limit / 1024 / 1024 / 1024} GB")
        recomps = graph_profiler.run_checkpoint_selection(mem_limit)

        # Modify the graph based on the checkpointing decision
        graph_profiler.apply_checkpointing(recomps)

        # Perform static analysis of the graph again
        with torch.no_grad():
            for _ in range(warm_up_iters):
                graph_profiler.run(*args)
            graph_profiler.reset_stats()

            for _ in range(profile_iters):
                graph_profiler.run(*args)
            graph_profiler.aggregate_stats()
            graph_profiler.print_stats(
                to_file=f"results/graph_prof_after_checkpoint_{self.model_name}_{self.batch_size}.txt"
            )

        return gm

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


if __name__ == "__main__":
    # Get the model name and batch size from the command line
    # use argparse to get the model name and batch size
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    # Run the experiment for each model and batch size
    print(f"Running experiment for {args.model} with batch size {args.batch_size}")
    exp = Experiment(args.model, args.batch_size)
    exp.init_opt_states()
    compiled_fn = compile(
        exp.train_step, exp.graph_transformation
    )  # for no ac exp.graph_transformation --> lambda gm, args: gm
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
    torch.cuda.synchronize()

    num_iters = 3
    run_times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
        end_event.record()
        torch.cuda.synchronize()
        run_times.append(start_event.elapsed_time(end_event))
    run_time = stats.mean(run_times)
    peak_memory = torch.cuda.max_memory_allocated()

    # Save results to dataframe
    result = pd.DataFrame(
        [
            {
                "Model": args.model,  #  + " (No AC)" for no ac
                "Batch Size": args.batch_size,
                "Run Time": run_time,
                "Peak Memory": peak_memory,
            }
        ]
    )

    # Save results to CSV
    result.to_csv(
        f"results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False
    )
