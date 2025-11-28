# HuggingChat Omni Router – Working Notes

_Last updated: 2025-11-16_

This repo hosts an experimental harness for the Arch Router tutorial. It routes user prompts to specialized Hugging Face models and executes them via the Inference API. These notes summarize the current architecture, recent progress, operating procedures, and how to resume work in a fresh environment or chat session.

## Project snapshot

| Component | Purpose |
| --- | --- |
| `omni_router_dev.py` | Main CLI for routing prompts, running the multi-query demo, and persisting router decisions. |
| `route_config.json` | Source of truth for routes, descriptions, and auto-learned specialist model IDs (persisted across runs). |
| `outputs/` | Destination directory for generated artifacts (e.g., images). |

## Current architecture

- **Router model**: `katanemo/Arch-Router-1.5B`, loaded via `torch`/`transformers` when the CLI runs.
- **Prompt contract**: Router must emit a strict JSON object `{"route", "model", "reason"}`. `_extract_route_json` sanitizes partial/markdown responses.
- **Model selection flow**:
  1. Router proposes `{route, model}`.
  2. `resolve_model_choice` validates the suggestion; if missing/invalid, it falls back to the most recent valid model stored for that route.
  3. `persist_model_choice` updates `route_config.json` with timestamps and reasons when a new model is accepted.
- **Specialist execution**: `SpecialistExecutor` calls the Hugging Face Inference API for text, image generation, or image understanding tasks. It retries `StopIteration` failures once and surfaces actionable error messages.

## Recent progress (this session)

1. **Config hygiene** – Moved all route metadata into `route_config.json` and scrubbed invalid entries (e.g., `"image_generation"` as a model ID).
2. **Model validation** – Normalization now rejects anything that isnt a proper `org/model` repo before persisting or executing.
3. **Prompt/schema rewrite** – Router instructions emphasize picking real repo IDs even when no history exists; the schema is enforced during parsing.
4. **Decision persistence** – Router decisions (with reasons) are automatically logged back to `route_config.json` so history can guide future runs.
5. **Specialist hardening** – Added StopIteration retries plus clearer failure guidance about HF token scopes and unavailable endpoints.
6. **Documentation** – Captured this working log so future sessions can resume with full context.

## Outstanding work

- **Rerun end-to-end demo**: `python omni_router_dev.py --demo --hf-token <token>` once the router model is installed. Confirm each route yields a valid specialist, specialists execute successfully, and the updated IDs persist.
- **Iterate on failures** (if any): adjust router prompting, seed `route_config.json` manually with safe defaults, or swap specialist models per route.

## Rules & procedures

1. **No hardcoded specialist IDs** – Only the router and persisted history determine which model to call.
2. **Strict JSON contract** – Router output must be machine-parseable; malformed responses trigger history-based fallback.
3. **Valid model IDs only** – Must contain a slash (`org/model`). `route_name` or placeholders are rejected before execution/persistence.
4. **Always persist decisions** – Successful `{route, model}` pairs are recorded with timestamps and reasons.
5. **HF token management** – Provide a valid token via `--hf-token` or `HF_API_TOKEN`/`HF_TOKEN` env vars before specialist runs.
6. **Specialist retries** – `StopIteration` is retried once automatically; lingering failures prompt manual model/token changes.
7. **Todo discipline** – Maintain the structured todo list (in chat or an external tracker), with only one item in progress at a time and immediate status updates.
8. **Verification requirement** – After modifying runnable code, run at least a lightweight command (e.g., `--list-routes` or targeted demo slice) to confirm the build path still works.

## How to run

### Environment

- Python 3.10+ with `torch`, `transformers`, `huggingface_hub`, and Pillow installed. (`.venv/` or `venv/` directories are available if you want to reuse them.)
- GPU strongly recommended for `katanemo/Arch-Router-1.5B`.

### Quick commands

```bash
# List the current routes and stored specialist history
python omni_router_dev.py --list-routes

# Route a single prompt (router only)
python omni_router_dev.py --prompt "Tell me a joke about computers."

# Attach an image URL for multimodal routing
python omni_router_dev.py --prompt "What flower is this?" \
  --image-url "https://example.com/photo.jpg"

# Run the full multi-query demo with specialists (requires model download + HF token)
python omni_router_dev.py --demo --hf-token <your_hf_token>
```

### Notes

- The demo processes `DEFAULT_MULTI_MODAL_QUERIES` (text + image cases). Keep an eye on console output for `[router] recorded model ...` messages indicating history persistence.
- `route_config.json` is safe to edit manually—just ensure each `models[].id` stays in `org/model` format.
- Generated images are saved under `outputs/` with timestamped filenames.

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `Specialist call failed: StopIteration` | Model unavailable on HF Inference or missing token scope | Provide a valid `--hf-token`, switch to a public model, or seed `route_config.json` with a known-good repo for that route. |
| Router keeps returning `other` | Prompt instructions insufficient or model not downloaded | Re-run after verifying the router checkpoint is available; consider nudging prompt context. |
| Invalid model saved (no `/`) | Manual edit or bad router output before validation | Delete the offending entry from `route_config.json` and re-run; validators now block new bad values. |

## Picking up in a new chat

1. **Skim this README** to internalize the status, rules, and pending tasks.
2. **Check the todo list** (currently: rerun demo). Update it as you work to maintain continuity across sessions.
3. **Inspect `route_config.json`** if you need to seed or verify specialist IDs before a demo run.
4. **Run `--list-routes`** to ensure dependencies are installed and the config loads cleanly before attempting the heavy demo.
5. **Proceed with the outstanding plan** (rerun demo, adjust specialists, add docs/tests) and record any new insights back into this file.

Thats everything needed to continue seamlessly, even if the conversation context resets.
