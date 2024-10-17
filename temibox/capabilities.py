import torch
from abc import ABCMeta, abstractmethod
from typing import List, Any, Protocol, runtime_checkable


class CudaCapable(metaclass=ABCMeta):
    r"""
    Represents a capability to use CUDA-compatible GPU
    """

    def __init__(self):
        self._cuda = False

    def _change_device(self, cuda: bool = False):
        if cuda and not torch.cuda.is_available():
            return

        models = self.get_cuda_components()

        for i, model in enumerate(models):
            if model is None:
                continue

            if cuda:
                models[i] = model.cuda()
            else:
                models[i] = model.cpu()

        self._cuda = cuda

    @abstractmethod
    def get_cuda_components(self) -> List[Any]:
        r"""
        Returns a list of cuda-capable components

        :return: list of cuda-capable components
        """
        raise NotImplementedError("capability method not implemented")

    def to_active_device(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Transfers the tensor to the components active device,
        i.e. if the component is loaded onto the GPU, the tensor
        will be copied to the GPU.

        :param tensor: torch Tensor

        :return: torch Tensor
        """

        if self.is_cuda:
            tensor = tensor.cuda()
        else:
            tensor = tensor.cpu()

        return tensor

    @property
    def is_cuda(self) -> bool:
        r"""
        Returns whether the current device is set to GPU or not

        :return: True if the component is loaded onto the GPU
        """
        if not hasattr(self, "_cuda"):
            self._cuda = False

        return self._cuda

    def set_cuda_mode(self, on: bool = True) -> None:
        r"""
        Sets cuda mode

        :param on: transfers the component onto the GPU if True, else to the CPU

        :return: None
        """

        if self.get_cuda_components():
            self._change_device(on)


class InferenceCapable(metaclass=ABCMeta):
    r"""
    Represents a capability to switch between training and inference modes
    """

    def __init__(self):
        self._inference = False

    @abstractmethod
    def get_inferential_components(self) -> List[Any]:
        r"""
        Returns a list of inference-capable components

        :return: list of inference-capable components
        """
        raise NotImplementedError("capability method not implemented")

    def zero_grad(self) -> None:
        r"""
        Zeroes out the gradient

        :return: None
        """
        for m in self.get_inferential_components():
            m.zero_grad()

    def _change_mode(self, eval_mode: bool):
        self._eval_mode = eval_mode

        components = self.get_inferential_components()

        for i, component in enumerate(components):

            self._inference = eval_mode

            if component is None:
                continue

            if eval_mode:
                components[i] = component.eval()
            else:
                components[i] = component.train()

            for param in component.parameters():
                param.requires_grad_(not eval_mode)

    @property
    def is_inference(self) -> bool:
        r"""
        Returns whether the component is in inference mode

        :return: True if component is in inference mode
        """
        if not hasattr(self, "_inference"):
            self._inference = False

        return self._inference

    def set_inference_mode(self, on: bool) -> None:
        r"""
        Sets inference and cuda modes

        :param on: turns inference mode on if True, else off

        :return: None
        """

        if self.get_inferential_components():
            self._change_mode(on)


class ParameterCapable(metaclass=ABCMeta):
    r"""
    Represents a parametrizable component
    """

    @abstractmethod
    def get_training_parameters(self) -> Any:
        r"""
        Returns all the trainable parameters

        :return: trainable parameters
        """
        raise NotImplementedError("capability method not implemented")


@runtime_checkable
class NeuralModel(Protocol):
    r"""
    Represents a component with the method `to_active_device`
    """

    def to_active_device(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

@runtime_checkable
class BinaryNeuralModel(NeuralModel, Protocol):
    r"""
    Represents a NeuralModel with the method `forward(document_embeddings)`
    """
    def forward(self,
                document_embeddings: torch.Tensor) -> torch.Tensor:
        pass


@runtime_checkable
class MultilabelBinaryNeuralModel(NeuralModel, Protocol):
    r"""
    Represents a NeuralModel with the method `forward(document_embeddings, target_embeddings)`
    """
    def forward(self,
                document_embeddings: torch.Tensor,
                target_embeddings: torch.Tensor) -> torch.Tensor:
        pass

@runtime_checkable
class MultilabelNeuralModel(NeuralModel, Protocol):
    r"""
    Represents a NeuralModel with the method `forward(document_embeddings)`
    """
    def forward(self,
                document_embeddings: torch.Tensor) -> torch.Tensor:
        pass

@runtime_checkable
class MultinomialNeuralModel(NeuralModel, Protocol):
    r"""
    Represents a NeuralModel with the method `forward(usecase_name, document_embeddings)`
    """
    def forward(self,
                usecase_name: str,
                document_embeddings: torch.Tensor) -> torch.Tensor:
        pass