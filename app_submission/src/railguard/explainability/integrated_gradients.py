"""Optional Captum Integrated Gradients hook."""


def integrated_gradients(model, inputs, target=None, **kwargs):
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError("Captum is optional; install railguard-ai[explain]") from exc
    return IntegratedGradients(model).attribute(inputs, target=target, **kwargs)

