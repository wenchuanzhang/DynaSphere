import math as scalar_math
import numpy as array_engine
import torch as tensor_engine

_HAS_SCALED_BESSEL = False
try:
    import torch.special as special_engine
    if hasattr(special_engine, "ive"):
        _HAS_SCALED_BESSEL = True
        _scaled_bessel_operator = special_engine.ive
except Exception:
    _HAS_SCALED_BESSEL = False

_fallback_message_emitted = False


def evaluate_scaled_bessel(order_tensor: tensor_engine.Tensor, argument_tensor: tensor_engine.Tensor) -> tensor_engine.Tensor:
    global _fallback_message_emitted
    if _HAS_SCALED_BESSEL:
        return _scaled_bessel_operator(order_tensor, argument_tensor)
    from scipy.special import ive as scipy_scaled_bessel
    if not _fallback_message_emitted:
        print("tensor_engine.special.ive not found; falling back to scipy.special.ive (no gradient wrt kappa).")
        _fallback_message_emitted = True
    scalar_order = float(order_tensor.detach().cpu().item()) if tensor_engine.is_tensor(order_tensor) else float(order_tensor)
    numpy_arguments = argument_tensor.detach().cpu().numpy()
    numpy_outputs = scipy_scaled_bessel(scalar_order, numpy_arguments)
    numpy_outputs = array_engine.asarray(numpy_outputs, dtype=array_engine.float64)
    return tensor_engine.tensor(numpy_outputs, device=argument_tensor.device, dtype=argument_tensor.dtype)


def log_modified_bessel(order_tensor: tensor_engine.Tensor, argument_tensor: tensor_engine.Tensor) -> tensor_engine.Tensor:
    safe_arguments = tensor_engine.clamp(argument_tensor, min=1e-8)
    return tensor_engine.log(evaluate_scaled_bessel(order_tensor, safe_arguments)) + safe_arguments


def mean_resultant_ratio(concentration_tensor: tensor_engine.Tensor, ambient_dimension: int) -> tensor_engine.Tensor:
    bessel_order = tensor_engine.tensor(ambient_dimension / 2.0 - 1.0, dtype=concentration_tensor.dtype, device=concentration_tensor.device)
    numerator_values = evaluate_scaled_bessel(bessel_order + 1.0, tensor_engine.clamp(concentration_tensor, min=1e-8))
    denominator_values = evaluate_scaled_bessel(bessel_order, tensor_engine.clamp(concentration_tensor, min=1e-8))
    return numerator_values / (denominator_values + 1e-20)


def log_normalization_constant(concentration_tensor: tensor_engine.Tensor, ambient_dimension: int) -> tensor_engine.Tensor:
    bessel_order = tensor_engine.tensor(ambient_dimension / 2.0 - 1.0, dtype=concentration_tensor.dtype, device=concentration_tensor.device)
    return bessel_order * tensor_engine.log(tensor_engine.clamp(concentration_tensor, min=1e-8)) - (ambient_dimension / 2.0) * scalar_math.log(2.0 * scalar_math.pi) - log_modified_bessel(bessel_order, concentration_tensor)


def log_unit_sphere_area(ambient_dimension: int) -> float:
    return scalar_math.log(2.0) + (ambient_dimension / 2.0) * scalar_math.log(scalar_math.pi) - tensor_engine.lgamma(tensor_engine.tensor(ambient_dimension / 2.0)).item()


def uniform_direction_divergence(concentration_tensor: tensor_engine.Tensor, ambient_dimension: int) -> tensor_engine.Tensor:
    log_constant = log_normalization_constant(concentration_tensor, ambient_dimension)
    resultant_ratio = mean_resultant_ratio(concentration_tensor, ambient_dimension)
    log_area = log_unit_sphere_area(ambient_dimension)
    return log_constant + concentration_tensor * resultant_ratio + log_area
