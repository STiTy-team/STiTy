"""Translation backends.

Each one attaches somewhere different -- a module global, a constructor kwarg, or
a monkey-patched module function -- so they share one `attach()` surface rather
than pretending to share a type.

  none    config.no_translation = True
  google  set_google_translate_api_key(), plus trans_guard for retry/failure counts
  local   set_local_translator(make_translator(...))
  gpt     handler kwarg gpt_translator=, plus config.enable_gpt_translation

`verify()` exists because init_model() only warns when a GPT translator was asked
for and OPENAI_API_KEY is absent -- it then falls back to Google and the run looks
successful. A benchmark that silently measures a different backend than the one
named in its config is worse than one that fails.
"""
import logging
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

from .. import registry
from ..errors import BenchConfigError


@dataclass
class Attachment:
    name: str
    fingerprint: str
    handler_kwargs: dict = field(default_factory=dict)
    config_overrides: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    verify: Callable[[object], None] = lambda server: None
    teardown: Callable[[], None] = lambda: None
    stats: Callable[[], dict] = dict


class TranslationBackend:
    def __init__(self, options: dict, ctx):
        self.options = options
        self.ctx = ctx

    def attach(self, server_module) -> Attachment:
        raise NotImplementedError


@registry.register("translation", "none")
class NoTranslation(TranslationBackend):
    def attach(self, server_module) -> Attachment:
        return Attachment(
            name="none",
            fingerprint="none",
            config_overrides={"no_translation": True},
            provenance={"backend": "none"},
        )


@registry.register("translation", "google")
class GoogleTranslation(TranslationBackend):
    def attach(self, server_module) -> Attachment:
        key = self.options.get("api_key") or os.environ.get("GOOGLE_TRANSLATE_API_KEY")
        backend = "v2" if key else "gtx"
        use_context = bool(self.options.get("context", False))
        server_module.set_google_translate_api_key(key, local_translation=False)

        guard = None
        try:
            import evaluation.ast.trans_guard as trans_guard
            trans_guard.install(
                server_module,
                backend=backend,
                retries=int(self.options.get("retries", 3)),
                timeout=float(self.options.get("timeout", 10.0)),
                backoff=float(self.options.get("backoff", 0.5)),
            )
            guard = trans_guard
        except Exception as e:  # noqa: BLE001 - instrumentation is optional
            logging.getLogger("bench").warning(
                "trans_guard not installed (%s); translation failure counts unavailable", e)

        def stats() -> dict:
            if guard is None:
                return {}
            snapshot = getattr(guard, "snapshot", None)
            return snapshot() if callable(snapshot) else {}

        def verify(server) -> None:
            if getattr(server_module, "LOCAL_TRANSLATOR", None) is not None:
                raise BenchConfigError(
                    "translation 'google' was requested but a local translator is "
                    "installed on the server module")

        return Attachment(
            name="google",
            fingerprint=f"google:{backend}:ctx={use_context}",
            config_overrides={"no_translation": False, "google_context": use_context},
            provenance={"backend": backend, "context": use_context,
                        "api_key_present": bool(key)},
            verify=verify,
            stats=stats,
        )


@registry.register("translation", "local")
class LocalTranslation(TranslationBackend):
    def attach(self, server_module) -> Attachment:
        from core.translator.local_translator import make_translator

        model = self.options.get("model", "google/madlad400-3b-mt")
        device = self.options.get("device")
        translator = make_translator(model_name=model, device=device)
        translator.load()

        counter = {"calls": 0, "failed": 0}
        inner_translate = translator.translate

        async def counting_translate(text, target_code, source_code=None, context=None):
            counter["calls"] += 1
            try:
                return await inner_translate(text, target_code, source_code, context=context)
            except Exception:
                counter["failed"] += 1
                raise

        translator.translate = counting_translate
        server_module.set_local_translator(translator)

        def verify(server) -> None:
            if getattr(server_module, "LOCAL_TRANSLATOR", None) is None:
                raise BenchConfigError(
                    "translation 'local' was requested but set_local_translator did not take")

        def teardown() -> None:
            server_module.set_local_translator(None)

        # trans_guard is deliberately not installed: _guarded_translate replaces
        # google_translate_async wholesale and never consults LOCAL_TRANSLATOR, so
        # the two are mutually exclusive.
        return Attachment(
            name="local",
            fingerprint=f"local:{model}:{device or 'auto'}",
            config_overrides={"no_translation": False},
            provenance={"backend": "local", "model": model, "device": device or "auto"},
            verify=verify,
            teardown=teardown,
            stats=lambda: dict(counter),
        )


class _MeteredClient:
    """Proxies the OpenAI client so every call's usage is recorded.

    GPTTranslator.correct_and_translate reads only the message content and drops
    resp.usage, so this is the only place the tokens can be captured. Everything
    except chat.completions.create falls through to the real client.
    """

    def __init__(self, inner, on_usage):
        self._inner = inner

        async def create(**kwargs):
            resp = await inner.chat.completions.create(**kwargs)
            on_usage(kwargs.get("model", ""), getattr(resp, "usage", None))
            return resp

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def __getattr__(self, name):
        return getattr(self._inner, name)


@registry.register("translation", "gpt")
class GptTranslation(TranslationBackend):
    def attach(self, server_module) -> Attachment:
        from core.translator import GPTTranslator

        model = self.options.get("model", "gpt-5.4-mini")
        context_window = int(self.options.get("context_window", 5))
        key = self.options.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise BenchConfigError(
                "translation 'gpt' needs an API key, but OPENAI_API_KEY is unset. "
                "The server would warn and silently fall back to Google Translate, "
                "so this run would measure a backend it does not name."
            )
        usage_log = self.ctx.usage_log

        def on_usage(used_model: str, usage) -> None:
            if usage_log is not None:
                usage_log.record(model=used_model or model, usage=usage,
                                 item=self.ctx.recorder.item)

        translator = GPTTranslator(api_key=key, model=model, max_context=context_window)
        translator._client = _MeteredClient(translator._client, on_usage)

        def verify(server) -> None:
            if server.gpt_translator is not translator:
                raise BenchConfigError(
                    "translation 'gpt' was requested but the server is holding a "
                    f"different translator ({type(server.gpt_translator).__name__}). "
                    "init_model falls back to Google when the key is missing."
                )

        return Attachment(
            name="gpt",
            fingerprint=f"gpt:{model}:ctx={context_window}",
            handler_kwargs={"gpt_translator": translator},
            config_overrides={"no_translation": False, "enable_gpt_translation": True,
                              "translation_model": model, "context_window": context_window,
                              "api_key": key},
            provenance={"backend": "gpt", "model": model, "context_window": context_window},
            verify=verify,
        )


# Per-backend option surface. Unknown options raise rather than being dropped:
# a `context_window: 10` that silently does nothing still produces a number, which
# is the worst outcome.
OPTIONS = {
    "none": set(),
    "google": {"api_key", "context", "retries", "timeout", "backoff"},
    "local": {"model", "device"},
    "gpt": {"model", "context_window", "api_key", "max_retries"},
}

# option -> StreamingConfig field, for options the server config carries directly
CONFIG_OPTIONS = {
    "gpt": {"model": ("translation_model", str),
            "context_window": ("context_window", int)},
    "google": {"context": ("google_context", bool)},
    "none": {},
    "local": {},
}


def config_contribution(name: str, options: dict) -> dict:
    """StreamingConfig fields this backend contributes. Raises on unknown options.

    Validated at config-parse time, so --dry-run catches a typo before the model
    loads. Importing this module stays cheap -- every heavy import is inside
    attach().
    """
    registry.get("translation", name)  # raises with the available list
    accepted = OPTIONS.get(name, set())
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise BenchConfigError(
            f"stity.translation: unknown option(s) {unknown} for {name!r} "
            f"(accepted: {sorted(accepted) or 'none'})"
        )
    out: dict = {"no_translation": name == "none",
                 "enable_gpt_translation": name == "gpt"}
    for key, (field, cast) in CONFIG_OPTIONS.get(name, {}).items():
        if key in options and options[key] is not None:
            out[field] = cast(options[key])
    return out


def build(name: str, options: dict, ctx) -> TranslationBackend:
    return registry.get("translation", name)(options, ctx)
