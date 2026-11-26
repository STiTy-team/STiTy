import uuid

from core.utils import cache
from core.utils.langs import CODE_TO_NAME

from . import translators
from .base import Translator

GPU_MEMORY_UTILIZATION = 0.2
MAX_MODEL_LEN = 2048


async def load_model(path: str) -> tuple:
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=path,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    return engine, tokenizer, SamplingParams(temperature=0.0, max_tokens=256)


@translators.register("zero-local-llm")
class ZeroLocalLLMTranslation(Translator):
    SETTINGS = {"model": ("model", str)}

    async def load(self) -> None:
        path = self.settings["model"]
        self.engine, self.tokenizer, self.params = await cache.load(
            ("zero-local-llm", path), lambda: load_model(path)
        )

    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str | None = None,
        context: list[str] | None = None,
    ) -> tuple[str, str]:
        request = (
            f"Translate the following text into {CODE_TO_NAME.get(target_lang, target_lang)}. "
            f"Output only the translation.\n\n{text}"
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": request}], add_generation_prompt=True, tokenize=False
        )
        async for out in self.engine.generate(prompt, self.params, request_id=str(uuid.uuid4())):
            pass
        return out.outputs[0].text.strip(), source_lang or ""
