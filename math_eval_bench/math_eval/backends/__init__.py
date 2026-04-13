from .base import Backend, GenerationResult


def create_backend(backend_type: str, **kwargs) -> Backend:
    if backend_type == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(**kwargs)
    elif backend_type == "api":
        from .api_backend import APIBackend
        return APIBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
