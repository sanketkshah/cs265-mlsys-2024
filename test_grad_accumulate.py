import torch
import torchvision.models as models
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode


class MyDispatch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        print(str(func.__name__))
        res = func(*args, **kwargs)
        return res


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super(MyModule, self).__init__()
        self.t = torch.nn.Parameter(torch.randn(3))
        self.t.requires_grad = True
        self.register_parameter("test", self.t)

    def forward(self, s: torch.Tensor):
        z = torch.nn.functional.linear(s, self.t)
        return z.sum()


def f(t: torch.Tensor, s: torch.Tensor):
    z = s + t
    z.sum().backward()


# FakeTensorMode(allow_non_fake_inputs=True) as fake_mode,
if __name__ == "__main__":
    with MyDispatch():
        s = torch.randn(3, 3, device="cuda")
        # t = torch.randn(3, device='cuda', requires_grad=True)
        # f(t, s)
        mod = MyModule().cuda()
        mod(s).backward()
