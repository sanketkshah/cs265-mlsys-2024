from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.models import hf_Bert, resnet50, resnet152
from typing import List, Dict, Any
import torch.nn as nn
import torch
import torch.optim as optim
import importlib
from graph_tracer import SEPFunction, compile

model_names: List[str] = [
    "torchbenchmark.models.hf_Bert.Model",
    "torchbenchmark.models.resnet50.Model",
    "torchbenchmark.models.resnet152.Model",
]

actual_model_names: List[str] = [
    "hf_Bert",
    "resnet50",
    "resnet152",
]

model_batch_sizes: Dict[str, int] = {
    "torchbenchmark.models.hf_Bert.Model": 32,
    "torchbenchmark.models.resnet50.Model": 256,
    "torchbenchmark.models.resnet152.Model": 64,
}

class Experiment:
    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        pos = model_name.rfind(".")
        module = importlib.import_module(model_name[:pos])
        model_class = getattr(module, model_name[(pos + 1) :])

        model: BenchmarkModel = model_class(
            "train", "cuda", batch_size=batch_size, extra_args=extra_args
        )
        self.model = model.model
        self.model_type = type(model)

        self.batch_size = batch_size
        self.example_inputs = model.example_inputs

        if self.model_type == hf_Bert.Model:

            def bert_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = model(**example_inputs).loss
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()

            self.train_step = bert_train_step
            self.optimizer = model.optimizer

        elif self.model_type in (resnet50.Model, resnet152.Model):
            self.loss_fn = model.loss_fn
            self.example_inputs = model.example_inputs[0]

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                output = model(example_inputs)
                target = torch.rand_like(output)
                loss = self.loss_fn(output, target)
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()

            self.optimizer = model.opt
            self.train_step = resnet_train_step

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")
        

if __name__ == "__main__":

    exp = Experiment(model_names[1], model_batch_sizes[model_names[1]])
    exp.run()
    compiled_fn = compile(exp.train_step, lambda x, y : x)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)


