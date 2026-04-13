"""vLLM offline inference backend."""

from typing import List, Optional

from vllm import LLM, SamplingParams

from .base import Backend, GenerationResult


class VLLMBackend(Backend):
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        seed: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        enable_thinking: bool = False,
    ):
        self.enable_thinking = enable_thinking
        llm_kwargs = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if seed is not None:
            llm_kwargs["seed"] = seed

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop or [],
        )

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [
            GenerationResult(
                text=o.outputs[0].text,
                prompt=o.prompt,
                finish_reason=o.outputs[0].finish_reason,
            )
            for o in outputs
        ]

    def generate_chat(self, messages_list: List[List[dict]]) -> List[GenerationResult]:
        chat_kwargs = {}
        if self.enable_thinking:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        outputs = self.llm.chat(messages_list, self.sampling_params, **chat_kwargs)
        return [
            GenerationResult(
                text=o.outputs[0].text,
                prompt=getattr(o, "prompt", str(messages_list[i])),
                finish_reason=o.outputs[0].finish_reason,
            )
            for i, o in enumerate(outputs)
        ]

    def generate_chat_n(
        self, messages_list: List[List[dict]], n: int
    ) -> List[List[GenerationResult]]:
        """Generate n samples per prompt using vLLM's native n parameter."""
        params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k,
            max_tokens=self.sampling_params.max_tokens,
            stop=self.sampling_params.stop,
            n=n,
        )
        chat_kwargs = {}
        if self.enable_thinking:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        outputs = self.llm.chat(messages_list, params, **chat_kwargs)
        return [
            [
                GenerationResult(
                    text=out.text,
                    prompt=getattr(o, "prompt", str(messages_list[i])),
                    finish_reason=out.finish_reason,
                )
                for out in o.outputs
            ]
            for i, o in enumerate(outputs)
        ]
