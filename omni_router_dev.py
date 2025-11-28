#%%
#!/usr/bin/env python3
"""Development harness for the Arch Router tutorial (router-only)."""
from __future__ import annotations

import argparse
import ast
import copy
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from PIL import Image

try:  # Optional heavy deps; validated lazily when needed
    import torch  # type: ignore
except ImportError:  # pragma: no cover - surfaced by runtime checks
    torch = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from huggingface_hub import InferenceClient  # type: ignore
except ImportError:  # pragma: no cover
    InferenceClient = None  # type: ignore

# ---------------------------------------------------------------------------
# Router prompt + route configuration
# ---------------------------------------------------------------------------
DEFAULT_ROUTER_MODEL = "katanemo/Arch-Router-1.5B"
DEFAULT_ROUTE_CONFIG_PATH = Path(__file__).with_name("route_config.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs")
``
TASK_INSTRUCTION = """
You are a routing specialist. Your job is to:
1. Read the conversation in <conversation>.
2. Pick the single best route from <routes>.
3. Recommend the single best Hugging Face model `id` to execute for that route.

Each route entry provides:
- `name`: canonical route identifier.
- `description`: how to recognize that route.
- Optional `models`: auto-learned history of model ids previously used for this route (may be empty). Feel free to reuse them or introduce a new model id when appropriate. New ids will be recorded automatically.
- Even when a route's `models` list is empty, you MUST still choose a real, public Hugging Face repository id in the format `organization/model_name` (for example `organization/model_name`).

<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
""".strip()

FORMAT_PROMPT = """
Return exactly ONE JSON object with this schema:
{
    "route": "<route_name from routes>",
    "model": "<huggingface repo id you recommend>",
    "reason": "<very short explanation>"
}

Rules:
1. If the user request is irrelevant or already satisfied, output {"route": "other", "model": "none", "reason": "..."}.
2. Otherwise, the `route` MUST be one of the names listed in <routes>. The `model` may reuse a prior entry from that route's `models` array or introduce a brand-new Hugging Face repo id if needed, but it must contain a `/` (organization/model_name).
3. Do not add extra keys, markdown, or text. The response must be valid JSON.
""".strip()

def load_route_config(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load the route configuration JSON from disk."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Route config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Route config must be a JSON array")
    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Route entry at index {idx} must be an object")
        normalized.append(_normalize_route_entry(entry))
    return normalized


def load_default_route_config() -> List[Dict[str, Any]]:
    """Helper to load the default config that ships with the repo."""
    return load_route_config(DEFAULT_ROUTE_CONFIG_PATH)


def _normalize_route_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    entry.setdefault("description", "")
    models = entry.get("models", [])
    normalized_models: List[Dict[str, Any]] = []
    if isinstance(models, list):
        for item in models:
            model_entry: Optional[Dict[str, Any]] = None
            if isinstance(item, dict) and "id" in item:
                model_entry = dict(item)
            elif isinstance(item, str) and item:
                model_entry = {"id": item}
            if not model_entry:
                continue
            normalized_id = _normalize_model_id(model_entry.get("id"))
            if not _is_valid_model_id(normalized_id):
                continue
            model_entry["id"] = normalized_id
            normalized_models.append(model_entry)
    entry["models"] = normalized_models
    return entry


def save_route_config(path: Union[str, Path], data: Sequence[Dict[str, Any]]) -> None:
    resolved = Path(path)
    with resolved.open("w", encoding="utf-8") as fh:
        json.dump(list(data), fh, ensure_ascii=False, indent=2)


def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _normalize_model_id(value: Optional[Any]) -> Optional[str]:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None


def _is_valid_model_id(model_id: Optional[str]) -> bool:
    return bool(model_id and "/" in model_id)


def _last_known_model(route_config: Sequence[Dict[str, Any]], route_name: Optional[str]) -> Optional[str]:
    if not route_name:
        return None
    for route in route_config:
        if route.get("name") != route_name:
            continue
        models = route.get("models") or []
        if isinstance(models, list) and models:
            entry = models[-1]
            if isinstance(entry, dict):
                normalized = _normalize_model_id(entry.get("id"))
                if _is_valid_model_id(normalized):
                    return normalized
            elif isinstance(entry, str):
                normalized = _normalize_model_id(entry)
                if _is_valid_model_id(normalized):
                    return normalized
    return None


def resolve_model_choice(
    route_config: Sequence[Dict[str, Any]],
    decision: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], str]:
    decision = decision or {}
    route_name = decision.get("route")
    model_id = _normalize_model_id(decision.get("model"))
    if _is_valid_model_id(model_id):
        return model_id, "router"
    known = _last_known_model(route_config, route_name)
    if _is_valid_model_id(known):
        return known, "history"
    return None, "missing"


def record_model_choice(
    route_config: List[Dict[str, Any]],
    *,
    route_name: Optional[str],
    model_id: Optional[str],
    reason: Optional[str] = None,
) -> bool:
    if not route_name or route_name == "other":
        return False
    model_id = _normalize_model_id(model_id)
    if not _is_valid_model_id(model_id):
        return False
    timestamp = _utc_timestamp()
    for route in route_config:
        if route.get("name") != route_name:
            continue
        models = route.setdefault("models", [])
        if not isinstance(models, list):
            models = []
            route["models"] = models
        for entry in models:
            if isinstance(entry, dict) and entry.get("id") == model_id:
                entry["last_used"] = timestamp
                if reason:
                    entry["last_reason"] = reason
                return True
        model_entry: Dict[str, Any] = {"id": model_id, "first_seen": timestamp, "last_used": timestamp}
        if reason:
            model_entry["last_reason"] = reason
        models.append(model_entry)
        return True
    return False


def persist_model_choice(
    path: Union[str, Path],
    route_config: List[Dict[str, Any]],
    *,
    route_name: Optional[str],
    model_id: Optional[str],
    reason: Optional[str] = None,
) -> bool:
    updated = record_model_choice(route_config, route_name=route_name, model_id=model_id, reason=reason)
    if updated:
        save_route_config(path, route_config)
    return updated


@dataclass
class SpecialistResult:
    success: bool
    route: Optional[str]
    model: Optional[str]
    message: str
    artifact_path: Optional[Path] = None


class SpecialistExecutor:
    """Executes router-selected models via the Hugging Face Inference API."""

    def __init__(
        self,
        *,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        timeout: int = 120,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        stop_iteration_retries: int = 1,
    ) -> None:
        self.hf_token = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.dry_run = dry_run
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clients: Dict[str, Any] = {}
        self.available = InferenceClient is not None
        self.stop_iteration_retries = max(0, stop_iteration_retries)

    def execute(
        self,
        *,
        route_name: Optional[str],
        model_id: Optional[str],
        user_prompt: str,
        enriched_prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> SpecialistResult:
        route_name = route_name or "unknown"
        prompt = (enriched_prompt or user_prompt or "").strip()
        if not model_id or model_id == "none":
            return SpecialistResult(False, route_name, model_id, "Router did not provide a specialist model; skipping.")
        if "/" not in model_id:
            return SpecialistResult(
                False,
                route_name,
                model_id,
                "Model id must be a Hugging Face repo identifier like 'org/model'.",
            )
        if self.dry_run:
            return SpecialistResult(True, route_name, model_id, f"[dry-run] Would have sent prompt of length {len(prompt)} to {model_id}.")
        if not self.available:
            return SpecialistResult(False, route_name, model_id, "huggingface_hub is not installed; cannot execute specialist models.")
        try:
            client = self._client_for(model_id)
        except Exception as exc:  # pragma: no cover - network setup errors
            return SpecialistResult(False, route_name, model_id, f"Failed to create inference client: {exc}")

        task = self._task_hint(route_name)
        attempts = self.stop_iteration_retries + 1
        last_stop_iteration: Optional[BaseException] = None

        def _run_task() -> SpecialistResult:
            if task == "image_generation":
                message, artifact = self._run_image_generation(client, prompt, route_name)
                return SpecialistResult(True, route_name, model_id, message, artifact)
            if task == "image_understanding" and image_url:
                caption = self._run_image_understanding(client, image_url)
                return SpecialistResult(True, route_name, model_id, caption)
            text = self._run_text_generation(client, prompt)
            return SpecialistResult(True, route_name, model_id, text)

        for attempt in range(attempts):
            try:
                return _run_task()
            except StopIteration as exc:
                last_stop_iteration = exc
                if attempt + 1 < attempts:
                    continue
                break
            except Exception as exc:  # pragma: no cover - remote inference errors
                detail = str(exc) or repr(exc)
                return SpecialistResult(False, route_name, model_id, f"Specialist call failed: {detail}")

        stop_msg = self._format_stop_iteration_message(model_id)
        detail = str(last_stop_iteration) if last_stop_iteration else "StopIteration"
        return SpecialistResult(False, route_name, model_id, f"{stop_msg} (details: {detail})")

    @staticmethod
    def _format_stop_iteration_message(model_id: Optional[str]) -> str:
        base = "Specialist call stopped before producing output"
        if not model_id:
            return base + "."
        return (
            f"{base}. This typically means '{model_id}' is not available on the Inference API endpoint or requires a Hugging Face token with the correct permissions. "
            "Try selecting a different model or provide an HF token via --hf-token."
        )

    def _client_for(self, model_id: str) -> Any:
        if model_id not in self._clients:
            if InferenceClient is None:
                raise RuntimeError("huggingface_hub is not installed")
            self._clients[model_id] = InferenceClient(model=model_id, token=self.hf_token, timeout=self.timeout)
        return self._clients[model_id]

    def _run_text_generation(self, client: Any, prompt: str) -> str:
        response = client.text_generation(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            return_full_text=False,
        )
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict) and "generated_text" in response:
            return str(response["generated_text"]).strip()
        return json.dumps(response, ensure_ascii=False)

    def _run_image_generation(self, client: Any, prompt: str, route_name: str) -> Tuple[str, Path]:
        image = client.text_to_image(prompt)
        if not isinstance(image, Image.Image):
            raise RuntimeError("text_to_image did not return a PIL image")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{route_name}.png"
        path = self.output_dir / filename
        image.save(path)
        return f"Generated image saved to {path}", path

    def _run_image_understanding(self, client: Any, image_url: str) -> str:
        image = self._download_image(image_url)
        if image is None:
            raise RuntimeError(f"Unable to download image from {image_url}")
        result = client.image_to_text(image=image)
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict) and "generated_text" in result:
            return str(result["generated_text"]).strip()
        return json.dumps(result, ensure_ascii=False)

    def _download_image(self, url: str) -> Optional[Image.Image]:
        try:
            with urlopen(url, timeout=15) as resp:  # type: ignore[arg-type]
                data = resp.read()
        except (URLError, HTTPError):
            return None
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None

    @staticmethod
    def _task_hint(route_name: Optional[str]) -> str:
        mapping = {
            "image_generation": "image_generation",
            "image_understanding": "image_understanding",
        }
        return mapping.get(route_name or "", "text_generation")

DEFAULT_MULTI_MODAL_QUERIES = [
    "Hello! How are you today?",
    "Tell me a joke about computers.",
    "Summarize the following article: Artificial intelligence in healthcare is growing rapidly...",
    "Who was the first woman to win a Nobel Prize in Physics?",
    "Write a Python function to check if a number is prime.",
    "I have this code that's failing, can you fix it?\npython\nx = [1,2,3]\nprint(x[::-1]",
    "Generate an image of a tropical beach at sunset with palm trees.",
    {
        "text": "Here's a photo of my garden. What kind of flower is the one with purple petals?",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Phlox_paniculata_%27Younique_Lilac%27.jpg",
    },
]


def format_router_prompt(route_config: Sequence[Dict[str, Any]], conversation: Sequence[Dict[str, Any]]) -> str:
    """Render the router prompt with the current route policy and conversation."""
    return (
        TASK_INSTRUCTION.format(
            routes=json.dumps(route_config, ensure_ascii=False, indent=2),
            conversation=json.dumps(list(conversation), ensure_ascii=False, indent=2),
        )
        + "\n\n"
        + FORMAT_PROMPT
    )


def _extract_route_json(text: str) -> Dict[str, Any]:
    """Robustly extract a JSON dict that contains a `route` key."""
    text = text.strip()
    if not text:
        return {}
    parsers = (json.loads, ast.literal_eval)
    for parser in parsers:
        try:
            obj = parser(text)
        except Exception:
            continue
        if isinstance(obj, dict) and "route" in obj:
            return obj
    for match in re.finditer(r"\{[\s\S]*?\}", text):
        snippet = match.group(0)
        for parser in parsers:
            try:
                obj = parser(snippet)
            except Exception:
                continue
            if isinstance(obj, dict) and "route" in obj:
                return obj
    route_match = re.search(r"\broute\b\s*:\s*(['\"])([^'\"]+)\1", text)
    if route_match:
        return {"route": route_match.group(2)}
    return {}


class MultiModalOmniRouter:
    """Wrapper around Arch-Router with hardened JSON parsing."""

    def __init__(
        self,
        model_name: str = DEFAULT_ROUTER_MODEL,
        *,
        route_config: Optional[Sequence[Dict[str, Any]]] = None,
        debug: bool = False,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.route_config = list(route_config or load_default_route_config())
        self.debug = debug
        self.last_prompt: str = ""
        self.last_raw_response: str = ""
        self.last_decision: Dict[str, Any] = {}
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise RuntimeError(
                "`torch` and `transformers` must be installed to instantiate MultiModalOmniRouter."
            )
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.generation_kwargs = {
            "max_new_tokens": 192,
            "do_sample": False,
            "temperature": 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if generation_kwargs:
            self.generation_kwargs.update(generation_kwargs)

    def predict_route(self, conversation: Sequence[Dict[str, Any]]) -> str:
        route_prompt = format_router_prompt(self.route_config, conversation)
        self.last_prompt = route_prompt
        messages = [{"role": "user", "content": route_prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = self._torch.ones_like(input_ids)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.generation_kwargs,
            )
        prompt_length = input_ids.shape[1]
        generated_only = generated_ids[0, prompt_length:]
        response = self.tokenizer.decode(generated_only, skip_special_tokens=True).strip()
        self.last_raw_response = response
        if self.debug:
            print("[router raw]", response)
        obj = _extract_route_json(response)
        self.last_decision = obj if isinstance(obj, dict) else {}
        route = obj.get("route") if isinstance(obj, dict) else None
        if isinstance(route, str):
            route = route.strip().strip("\"'\n\r\t ")
        configured = {route_info["name"] for route_info in self.route_config}
        configured_lower = {name.lower(): name for name in configured}
        if isinstance(route, str) and route.lower() in configured_lower:
            return configured_lower[route.lower()]
        return "other"

    def route_conversation(
        self,
        user_message: str,
        conversation_history: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        image_url: Optional[str] = None,
        image_data: Optional[Union[Image.Image, str]] = None,
    ) -> Tuple[str, str]:
        history = list(conversation_history or [])
        user_entry = {"role": "user", "content": user_message}
        if image_url or image_data:
            image_info = "[Image attached]"
            if image_url:
                image_info += f" (URL: {image_url})"
            user_entry["content"] = f"{user_message}\n{image_info}"
        history.append(user_entry)
        route = self.predict_route(history)
        return route, user_entry["content"]


def print_query_result(
    router: MultiModalOmniRouter,
    query: str,
    *,
    image_url: Optional[str] = None,
    specialist: Optional[SpecialistExecutor] = None,
) -> Tuple[str, Optional[SpecialistResult]]:
    route, enriched = router.route_conversation(query, image_url=image_url)
    print(f"Query: {query}")
    print(f"Predicted Route: {route}")
    decision = router.last_decision or {}
    model = decision.get("model")
    resolved_model, model_source = resolve_model_choice(router.route_config, decision)
    if resolved_model and resolved_model != model:
        decision["model"] = resolved_model
        router.last_decision = decision
        print(f"[router] substituted model '{resolved_model}' from {model_source} fallback")
    model = resolved_model
    if model:
        print(f"Chosen Model: {model}")
    else:
        print("Chosen Model: [none available]")
    reason = decision.get("reason")
    if reason:
        print(f"Router Reason: {reason}")
    if router.last_raw_response:
        print("Raw Router Output:")
        print(router.last_raw_response)
    specialist_result: Optional[SpecialistResult] = None
    if specialist:
        specialist_result = specialist.execute(
            route_name=decision.get("route"),
            model_id=model,
            user_prompt=query,
            enriched_prompt=enriched,
            image_url=image_url,
        )
        _print_specialist_result(specialist_result)
    print("-" * 80)
    return route, specialist_result


def _print_specialist_result(result: Optional[SpecialistResult]) -> None:
    if not result:
        return
    header = "Specialist Output" if result.success else "Specialist Error"
    print(header + ":")
    if result.message:
        print(result.message)
    if result.artifact_path:
        print(f"Artifact saved to: {result.artifact_path}")


def run_router_demo(
    router: MultiModalOmniRouter,
    queries: Sequence[str],
    *,
    specialist: Optional[SpecialistExecutor] = None,
    decision_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    print("\n=== Router Demo ===")

    successes = 0
    for entry in queries:
        query = entry.get("text", "") if isinstance(entry, dict) else entry
        image_url = entry.get("image_url") if isinstance(entry, dict) else None
        route, _ = print_query_result(router, query, image_url=image_url, specialist=specialist)
        successes += int(route != "other")
        if decision_callback:
            decision_callback(router.last_decision or {})
    print(f"Non-'other' routes: {successes}/{len(queries)}")

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omni router development harness")
    parser.add_argument("--prompt", help="Single user prompt to route", default=None)
    parser.add_argument("--image-url", help="Optional image URL to attach to the prompt", default=None)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in router-only multi-query demo after routing the prompt",
    )
    parser.add_argument("--list-routes", action="store_true", help="Print the current route configuration and exit")
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL, help="Override the router checkpoint")
    parser.add_argument(
        "--route-config",
        default=None,
        help=f"Path to JSON route config (defaults to {DEFAULT_ROUTE_CONFIG_PATH.name} next to this script)",
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for specialist inference (falls back to env)")
    parser.add_argument("--skip-specialists", action="store_true", help="Skip executing router-selected models")
    parser.add_argument(
        "--specialist-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to request from specialist text models",
    )
    parser.add_argument(
        "--specialist-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for specialist text models",
    )
    parser.add_argument("--debug-router", action="store_true", help="Print raw router generations")
    parser.add_argument("--max-new-tokens", type=int, default=192, help="Router generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Router sampling temperature")
    parser.add_argument("--no-banner", action="store_true", help="Suppress usage banner when no action is requested")
    args = parser.parse_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.route_config).expanduser() if args.route_config else DEFAULT_ROUTE_CONFIG_PATH
    try:
        route_config = load_route_config(config_path)
    except (OSError, ValueError) as exc:
        print(f"Failed to load route config from {config_path}: {exc}")
        return 1
    if args.list_routes:
        for route in route_config:
            print(f"- {route['name']}: {route['description']}")
        return 0
    if not args.prompt and not args.demo:
        if not args.no_banner:
            print("Nothing to do. Pass --prompt or --demo. Use --help for options.")
        return 0
    router = MultiModalOmniRouter(
        model_name=args.router_model,
        route_config=route_config,
        debug=args.debug_router,
        generation_kwargs={"max_new_tokens": args.max_new_tokens, "temperature": args.temperature},
    )

    specialist: Optional[SpecialistExecutor] = None
    if not args.skip_specialists:
        specialist = SpecialistExecutor(
            hf_token=args.hf_token,
            max_new_tokens=args.specialist_max_new_tokens,
            temperature=args.specialist_temperature,
            output_dir=DEFAULT_OUTPUT_DIR,
            dry_run=False,
        )

    def persist_decision(decision: Dict[str, Any]) -> None:
        updated = persist_model_choice(
            config_path,
            route_config,
            route_name=decision.get("route"),
            model_id=decision.get("model"),
            reason=decision.get("reason"),
        )
        if updated:
            router.route_config = copy.deepcopy(route_config)
            route_name = decision.get("route")
            model_id = decision.get("model")
            print(f"[router] recorded model '{model_id}' for route '{route_name}' in {config_path}")
    if args.prompt:
        route, enriched = router.route_conversation(args.prompt, image_url=args.image_url)
        print("\n=== Single Prompt Routing ===")
        print("Prompt:", args.prompt)
        if args.image_url:
            print("Image URL:", args.image_url)
        print("Router content fed:")
        print(enriched)
        print("Predicted route:", route)
        decision = router.last_decision or {}
        if decision:
            reason = decision.get("reason")
            resolved_model, source = resolve_model_choice(router.route_config, decision)
            if resolved_model and resolved_model != decision.get("model"):
                decision["model"] = resolved_model
                router.last_decision = decision
                print(f"[router] substituted model '{resolved_model}' from {source} fallback")
            if resolved_model:
                print("Chosen model:", resolved_model)
            else:
                print("Chosen model: [none available]")
            if reason:
                print("Router reason:", reason)
        specialist_result = None
        if specialist:
            specialist_result = specialist.execute(
                route_name=router.last_decision.get("route"),
                model_id=router.last_decision.get("model"),
                user_prompt=args.prompt,
                enriched_prompt=enriched,
                image_url=args.image_url,
            )
            _print_specialist_result(specialist_result)
        persist_decision(router.last_decision or {})
    if args.demo:
        run_router_demo(router, DEFAULT_MULTI_MODAL_QUERIES, specialist=specialist, decision_callback=persist_decision)
    return 0

#%%
if __name__ == "__main__":  # pragma: no cover
    main()

# %%
