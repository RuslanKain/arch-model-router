#%%
# HuggingChat Omni Router Tutorial Dev


import os
import json
import re
import ast
from typing import Any, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import requests
from io import BytesIO
from IPython.display import display as ipy_display, Markdown as md
import base64

# In[21]:


# Initialize the routing model
model_name = "katanemo/Arch-Router-1.5B" 
# Arch-Router-1.5B is the routing model on Hugging Face at the time of writing, it may change in the future
# Please visit https://huggingface.co/katanemo/Arch-Router-1.5B for more information
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype="auto", 
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# In[22]:


# Prompts for routing (tightened for strict JSON output)
TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""

FORMAT_PROMPT = """
Your task is to decide which route best suits the user's latest intent in <conversation></conversation>.
Rules:
1. If the user's latest request is irrelevant or already fulfilled, return {"route": "other"}.
2. Otherwise, analyze the route descriptions and choose the best match using the exact route name from <routes>.
3. Return ONLY one JSON object. Do not include any additional text, formatting, or explanation.
4. The JSON schema is exactly: {"route": "route_name"}
"""

# In[23]:


# Route configuration (7-route policy from research) -- descriptions tuned for discrimination
route_config = [
    {
        "name": "general_chat",
        "description": "Open-ended small talk, casual conversation, or broad assistance not asking for facts, summary, or code.",
    },
    {
        "name": "summarization",
        "description": "Requests to 'summarize', 'condense', 'tl;dr', or produce a shorter version of provided text preserving key ideas.",
    },
    {
        "name": "knowledge_qa",
        "description": "Short factual questions seeking a specific answer (science, history, people, places, definitions). Usually ends with '?'.",
    },
    {
        "name": "code_generation",
        "description": "Requests to write new code, functions, algorithms, or boilerplate from scratch.",
    },
    {
        "name": "code_debugging",
        "description": "Provided broken code or error messages asking for fixes, explanations, or corrections.",
    },
    {
        "name": "image_generation",
        "description": "User wants to create/generate an image from a textual description. Output is an image.",
    },
    {
        "name": "image_understanding",
        "description": "User supplies or references an image and asks for description, objects, or interpretation of its content.",
    },
]

# In[24]:


def format_prompt(route_config: List[Dict[str, Any]], conversation: List[Dict[str, Any]]):
    """Format the prompt for the routing model"""
    return (
        TASK_INSTRUCTION.format(
            routes=json.dumps(route_config), 
            conversation=json.dumps(conversation)
        )
        + FORMAT_PROMPT
    )

# In[26]:


# Early direct demo of the routing model (robust JSON extraction)
demo_conversation = [{"role": "user", "content": "Can you summarize the key points of the article I provided?"}]
prompt = format_prompt(route_config, demo_conversation)
messages = [{ 'role': 'user', 'content': prompt }]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)
with torch.no_grad():
    gen_ids = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=192, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
prompt_len = input_ids.shape[1]
gen_only = gen_ids[0, prompt_len:]
raw_response = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
print('Raw model output:', raw_response)
# Robust extraction: attempt to parse first JSON-like brace block containing 'route'
def extract_route_obj(text: str):
    text = text.strip()
    # direct JSON attempt
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'route' in obj: return obj
    except Exception: pass
    # scan for smallest brace block containing 'route'
    for m in re.finditer(r'\{[^{}]*\}', text):
        block = m.group(0)
        if 'route' not in block: continue
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and 'route' in obj: return obj
        except Exception: continue
    # permissive regex fallback
    rm = re.search(r'\broute\b\s*:\s*(["])([^"]+)', text)
    if rm: return { 'route': rm.group(2) }
    return {}
route_obj = extract_route_obj(raw_response)
print('Parsed route object:', route_obj if route_obj else '[none found]')
print('Routed to:', route_obj.get('route','other'))

# In[7]:


class MultiModalOmniRouter:
    def __init__(self, model_name="katanemo/Arch-Router-1.5B", debug: bool = False, route_config: List[Dict[str, Any]] = None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Ensure pad_token_id is set to avoid warnings; if missing, mirror eos
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Allow passing a custom route_config; default to global route_config
        self.route_config = route_config if route_config is not None else globals().get("route_config", [])
        self.debug = debug
        # Keep last prompt/raw for inspection
        self.last_prompt: str = ""
        self.last_raw_response: str = ""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract a JSON object with a 'route' field from arbitrary text.
        Handles strict JSON, first JSON-like block, Python dicts with single quotes, and regex fallback.
        """
        text = text.strip()
        # Fast path: strict JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "route" in obj:
                return obj
        except Exception:
            pass
        # Try python-literal dict (single quotes)
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict) and "route" in obj:
                return obj
        except Exception:
            pass
        # Look for the first {...} region that parses as JSON
        for match in re.finditer(r"\{[\s\S]*?\}", text):
            snippet = match.group(0)
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict) and "route" in obj:
                    return obj
            except Exception:
                # Try python dict inside braces
                try:
                    obj = ast.literal_eval(snippet)
                    if isinstance(obj, dict) and "route" in obj:
                        return obj
                except Exception:
                    continue
        # Permissive: capture "route": 'name' or "name" without requiring full JSON
        m = re.search(r"\broute\b\s*:\s*(['\"])([^'\"]+)\1", text)
        if m:
            return {"route": m.group(2)}
        return {}

    def predict_route(self, conversation: List[Dict[str, Any]]):
        """Predict the best route for a given conversation."""
        route_prompt = format_prompt(self.route_config, conversation)
        self.last_prompt = route_prompt
        messages = [{"role": "user", "content": route_prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        # Build attention mask explicitly to avoid warning
        attention_mask = torch.ones_like(input_ids)

        # Generate deterministically for routing stability
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract the generated response
        prompt_length = input_ids.shape[1]
        generated_only = generated_ids[0, prompt_length:]
        response = self.tokenizer.decode(generated_only, skip_special_tokens=True).strip()
        self.last_raw_response = response
        if self.debug:
            print("[Router raw output]", repr(response))

        # Parse the JSON response robustly
        obj = self._extract_json(response)
        route = obj.get("route") if isinstance(obj, dict) else None
        if isinstance(route, str):
            route = route.strip().strip("\"'\n\r\t ")
        
        # Validate against configured routes (case-insensitive)
        configured = {r["name"] for r in self.route_config}
        configured_lower = {name.lower(): name for name in configured}
        if isinstance(route, str) and route.lower() in configured_lower:
            return configured_lower[route.lower()]

        # If parsing/validation fails, default to 'other'
        return "other"

    def route_conversation(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]] = None,
        image_url: str = None,
        image_data: Union[Image.Image, str] = None,
    ):
        """Route a user message to the appropriate handler with multi-modal support"""
        if conversation_history is None:
            conversation_history = []

        # Create conversation entry with potential image reference
        current_conversation = conversation_history.copy()

        # Handle different types of input
        user_entry = {"role": "user", "content": user_message}

        # Add image information if provided
        if image_url or image_data:
            image_info = "[Image attached]"
            if image_url:
                image_info += f" (URL: {image_url})"
            user_entry["content"] = f"{user_message}\n{image_info}"

        current_conversation.append(user_entry)

        # Get the predicted route
        route = self.predict_route(current_conversation)

        return route, user_entry["content"]

# In[8]:


# Initialize the multi-modal router (enable debug to see raw outputs)
router = MultiModalOmniRouter(debug=False, route_config=route_config)

# In[9]:


# Test queries aligned to 7-route policy
multi_modal_queries = [
    # General chat / small talk
    "Hello! How are you today?",
    "Tell me a joke about computers.",

    # Summarization
    "Summarize the following article: Artificial intelligence in healthcare is growing rapidly...",

    # Knowledge Q&A
    "Who was the first woman to win a Nobel Prize in Physics?",

    # Code generation
    "Write a Python function to check if a number is prime.",

    # Code debugging
    "I have this code that's failing, can you fix it?\npython\nx = [1,2,3]\nprint(x[::-1]",

    # Image generation
    "Generate an image of a tropical beach at sunset with palm trees.",

    # Image understanding (assume image provided out-of-band)
    "Here's a photo of my garden. What kind of flower is the one with purple petals?",
]

# In[10]:


# Debug flags you can toggle per run
SHOW_RAW = True
SHOW_PROMPT = False


def print_query_result(router: MultiModalOmniRouter, query: str, show_raw: bool = SHOW_RAW, show_prompt: bool = SHOW_PROMPT):
    route, content = router.route_conversation(query)
    print(f"Query: {query}")
    print(f"Predicted Route: {route}")
    if show_prompt:
        print("-- Prompt sent to router --")
        print(router.last_prompt)
    if show_raw:
        print("-- Raw model response --")
        print(router.last_raw_response)
    print("-" * 80)
    return route


print("Testing Omni Router with multi-modal queries:\n")
print("-" * 80)

ok = 0
for query in multi_modal_queries:
    route = print_query_result(router, query)
    ok += int(route != "other")

print(f"Finished. Routed to non-'other' for {ok}/{len(multi_modal_queries)} queries.")

# In[11]:


# Simple routing evaluation: distribution and 'other' rate
from collections import Counter as _Counter
route_preds = []
for q in multi_modal_queries:
    r, _ = router.route_conversation(q)
    route_preds.append(r)
dist = _Counter(route_preds)
print("Route distribution:")
for k, v in dist.items():
    print(f"  {k}: {v}")
print(f"'other' rate: {dist.get('other', 0)}/{len(route_preds)} ({100.0*dist.get('other',0)/max(1,len(route_preds)):.1f}%)")

# In[12]:


# Validation metrics for parsing/compliance
def evaluate_routing(router: MultiModalOmniRouter, queries: List[str]):
    stats = {"raw_has_brace": 0, "parsed_ok": 0, "route_in_config": 0, "other": 0}
    results = []
    for q in queries:
        r, _ = router.route_conversation(q)
        raw = router.last_raw_response or ""
        has_brace = ("{" in raw and "}" in raw)
        stats["raw_has_brace"] += int(has_brace)
        # Try parse again using internal method
        obj = router._extract_json(raw) if raw else {}
        parsed_ok = (isinstance(obj, dict) and "route" in obj)
        stats["parsed_ok"] += int(parsed_ok)
        # route in config
        cfg = {x["name"] for x in router.route_config}
        in_cfg = (r in cfg)
        stats["route_in_config"] += int(in_cfg)
        stats["other"] += int(r == "other")
        results.append({"query": q, "raw": raw, "route": r, "parsed_ok": parsed_ok, "in_cfg": in_cfg})
    print("Compliance summary:")
    print(stats)
    return results
 
# Run evaluation (optional)
# eval_results = evaluate_routing(router, multi_modal_queries)

# In[13]:


# Create a more sophisticated multi-modal system

class AdvancedMultiModalSystem:
    def __init__(self, hf_token=None):
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        self.hf_token = os.environ.get("HF_TOKEN")
        self.router = MultiModalOmniRouter(route_config=route_config)
        self.conversation_history = []

        # Route -> model_id mapping based on capability
        self.route_to_model = {
            "general_chat": CHAT_MODELS["general_chat"],
            "summarization": GENERATION_MODELS["summarization"],
            "knowledge_qa": CHAT_MODELS["knowledge_qa"],
            "code_generation": GENERATION_MODELS["code_generation"],
            "code_debugging": CHAT_MODELS["code_debugging"],
            "image_generation": VISION_MODELS["image_generation"],
            "image_understanding": VISION_MODELS["image_understanding"],
            "other": GENERATION_MODELS["other"],
        }

    def run_specialist(self, route: str, user_message: str, image_url: str = None) -> Dict[str, Any]:
        model_id = self.route_to_model.get(route, self.route_to_model["other"])
        # Chat/instruct routes
        if route in CHAT_MODELS:
            content, text_meta = run_chat_with_fallback(model_id, user_message, history=self.conversation_history)
            return {"type": "text", "model": model_id, "content": content, "text_meta": text_meta}
        # Summarization
        if route == "summarization":
            content, text_meta = run_summarize_with_fallback(model_id, user_message)
            return {"type": "text", "model": model_id, "content": content, "text_meta": text_meta}
        # Plain code generation route (use text generation)
        if route == "code_generation":
            content, text_meta = run_text_with_fallback(model_id, user_message)
            return {"type": "text", "model": model_id, "content": content, "text_meta": text_meta}
        # Vision routes
        if route == "image_generation":
            try:
                img = hf_image_generate(model_id, user_message)
                return {"type": "image", "model": model_id, "image": img}
            except Exception as e:
                return {"type": "error", "model": model_id, "error": str(e)}
        if route == "image_understanding":
            if image_url:
                caption, meta = hf_image_caption(model_id, image_url)
                return {"type": "text", "model": model_id, "content": caption, "caption_meta": meta}
            else:
                return {"type": "error", "model": model_id, "error": "image_url required for image_understanding"}
        # Fallback
        content, text_meta = run_chat_with_fallback(model_id, user_message, history=self.conversation_history)
        return {"type": "text", "model": model_id, "content": content, "text_meta": text_meta}

    def process_message(self, user_message: str, image_url: str = None, image_data: Union[Image.Image, str] = None):
        route, full_content = self.router.route_conversation(
            user_message,
            self.conversation_history,
            image_url=image_url,
            image_data=image_data,
        )
        result = self.run_specialist(route, user_message, image_url=image_url)

        # Update history for chat continuity
        self.conversation_history.append({"role": "user", "content": full_content})
        if result.get("type") == "text":
            self.conversation_history.append({"role": "assistant", "content": result.get("content", "")})
        else:
            self.conversation_history.append({"role": "assistant", "content": f"[{result.get('type')}] from {result.get('model')}"})
        return route, result

# In[14]:


# Baseline conformance check (aligns with HF model card example)
hf_example_routes = [
    {
        "name": "code_generation",
        "description": "Generating new code snippets, functions, or boilerplate based on user prompts or requirements",
    },
    {
        "name": "bug_fixing",
        "description": "Identifying and fixing errors or bugs in the provided code across different programming languages",
    },
    {
        "name": "performance_optimization",
        "description": "Suggesting improvements to make code more efficient, readable, or scalable",
    },
    {
        "name": "api_help",
        "description": "Assisting with understanding or integrating external APIs and libraries",
    },
    {
        "name": "programming",
        "description": "Answering general programming questions, theory, or best practices",
    },
 ]
 
hf_example_conversation = [
    {
        "role": "user",
        "content": "fix this module 'torch.utils._pytree' has no attribute 'register_pytree_node'. did you mean: '_register_pytree_node'?",
    }
 ]
 
# Use a temporary router with only the HF example routes
tmp_router = MultiModalOmniRouter(debug=False, route_config=hf_example_routes)
tmp_route, _ = tmp_router.route_conversation(hf_example_conversation[-1]["content"])
print("[HF example baseline route]", tmp_route)

# ## Domain–Action preference routing (primer)
#  
# Arch-Router frames routing as mapping each query to a (domain, action) pair, then applying your preferences to pick the best model. For quick demos, we can combine (domain, action) into a single route name like `programming__bug_fixing`. Keep the list concise and the descriptions discriminative; the model works best when route options are well-separated.

# ## What “Predicted Route” means
#  
# - It’s the route name emitted by Arch-Router as a tiny JSON object, e.g., `{"route": "code_generation"}`.
# - We validate that the route exists in your `route_config`; if not, we display `other`.
# - Toggle `SHOW_PROMPT` and `SHOW_RAW` above to see the exact prompt sent and the raw model output per query. This is the fastest way to debug “why other?”.

# In[ ]:


# Test with actual image processing (optional)

def test_with_sample_image():
    """Test the router with a sample image"""
    # Sample image URL (you can replace with any image)
    sample_image_url = "https://images.unsplash.com/photo-1518770660439-4636190af475?w=800"
    
    # Download and display the image
    try:
        response = requests.get(sample_image_url)
        image = Image.open(BytesIO(response.content))
        print("Sample image loaded successfully!")
        display(image)
    except:
        print("Could not load sample image. Using placeholder.")
        image = None
    
    # Test routing with image context
    queries_with_images = [
        "What can you see in this image?",
        "Describe the scene in this photo",
        "What type of devices are shown here?",
        "This image shows my code error. How do I fix it?",
    ]
    
    print("\nTesting with image context:\n")
    print("-" * 80)
    
    for query in queries_with_images:
        route, content = router.route_conversation(
            query, 
            image_url=sample_image_url if image else None
        )
        
        print(f"Query: {query}")
        print(f"Content with image context: {content}")
        print(f"Predicted Route: {route}")
        
        if "image" in route or "visual" in route:
            print(f"→ Correctly routed to image processing")
            
        print("-" * 80)

# Run the image test (uncomment to execute)
test_with_sample_image()

# In[ ]:


print_query_result(router, "Write a Python function to check if a number is prime.", show_raw=True, show_prompt=True)

# In[ ]:




# ## Setup: tokens and execution mode
# 
# We'll use the Hugging Face Inference API for quick, zero-setup execution. Set your token in an environment variable named `HF_TOKEN`. If it's not available, the demo will try to run locally where possible (text-only), and skip image features.
# 
# - Text generation/summarization: Inference API (or local model if available)
# - Image generation: Inference API (stable-diffusion)
# - Image understanding: Inference API (BLIP image captioning)
# 
# Security note: never hardcode tokens in notebooks that will be shared.

# In[ ]:


# Configure token and execution helpers
# Paste a token here for Colab/demo convenience (leave as empty string to rely on environment):
PASTED_HF_TOKEN = ""  # <-- REPLACE with your token for a full demo
# Security note: Do NOT commit real production tokens. For public demos use a scoped, revocable token.

import os
HF_TOKEN = os.environ.get("HF_TOKEN") or (PASTED_HF_TOKEN if PASTED_HF_TOKEN and not PASTED_HF_TOKEN.endswith("X") else None)
USE_API = bool(HF_TOKEN)

from huggingface_hub import InferenceClient
from typing import Dict, List
import traceback

client_cache: Dict[str, InferenceClient] = {}

def get_client(model_id: str) -> InferenceClient:
    if model_id not in client_cache:
        client_cache[model_id] = InferenceClient(model=model_id, token=HF_TOKEN) if USE_API else None
    return client_cache[model_id]

# Model ids (grouped by capability)
# NOTE: We separate chat-oriented models (use conversational task) from plain generation models.
CHAT_MODELS = {
    "general_chat": "mistralai/Mistral-7B-Instruct-v0.2",
    "knowledge_qa": "mistralai/Mistral-7B-Instruct-v0.2",
    # Code debugging often benefits from an instruct/chat model
    "code_debugging": "mistralai/Mistral-7B-Instruct-v0.2",
}
GENERATION_MODELS = {
    "code_generation": "bigcode/starcoder2-7b",
    "summarization": "facebook/bart-large-cnn",
    "other": "mistralai/Mistral-7B-Instruct-v0.2",
}
VISION_MODELS = {
    "image_generation": "stabilityai/stable-diffusion-2-1",
    # Use a widely supported captioning model instead of BLIP large to reduce errors
    "image_understanding": "nlpconnect/vit-gpt2-image-captioning",
}

# Consolidated maps for downstream access
TEXT_MODELS = {**CHAT_MODELS, **GENERATION_MODELS}
IMAGE_MODELS = VISION_MODELS

# HTTP headers and fallback image URLs for robust fetching
HTTP_IMAGE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://www.google.com",
}
FALLBACK_IMAGE_URLS = [
    # Stable demo images
    "https://picsum.photos/id/237/800/600",  # dog
    "https://placekitten.com/800/600",       # kitten placeholder
]

# Debug flag for caption path
CAPTION_DEBUG = True


def build_chat_prompt(history: List[Dict[str, str]], user_prompt: str) -> str:
    """Flatten a chat history + new user turn into a single instruction-style prompt."""
    lines: List[str] = []
    for turn in history or []:
        role = turn.get('role')
        content = turn.get('content','').strip()
        if not content: continue
        if role == 'user': lines.append(f'User: {content}')
        elif role == 'assistant': lines.append(f'Assistant: {content}')
    lines.append(f'User: {user_prompt.strip()}')
    lines.append('Assistant:')  # model should complete this
    return '

def hf_chat(model_id: str, prompt: str, history: list = None, max_new_tokens: int = 256) -> str:
    """Simplified chat wrapper using text_gener# InferenceClient canversational signattre: clieni.conversational(messages=[...])
o       out n; formats history inline."""
    if not USE_API:
        return '[HF_TOKEN not set; skipping remote call]'
    client = get_client(model_id)
    chat_prompt = build_chat_prompt(history or [], prompt)
    try:
        out = client.text_generation(chat_prompt, max_new_tokens=max_new_tokens, temperature=0.7)
        if isinstance(out, str):
            return out
        return str(out)
    except Exception as e:
        return f'[chat error] {e}'

def hf_text(model_id: str, prompt: str, max_new_tokens: int = 256) -> str:
    if not USE_API:
        return '[HF_TOKEN not se
t; skipping remote call]'
    client = get_client(model_id)
    try:
        out = client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=0.2)
        return out if isinstance(out, str) else str(out)
    except Exception as e:
        return f'[text error] {e}'

def _extract_summary_text(s: str) -> str:
    # Try to pull summary_text='...' out of repr-like strings
    m = re.search(r"summary_te(o tr _
= get_client(model_id)
    try:
        out = client.summarization(text)
        s = out if isinstance(out, str) else str(out)
        return _extract_summary_text(s)
    except Exception as e:
        return f"[sumarizatoreturn error] {e}"

def hf_image_generate(model_itr, ret USE_API:
        raise RuntimeError("HF_TOKEN not set; cannot ge
nerate image")
    client = get_client(model_id)
    try:
        img_bytes = client.text_to_image(prompt)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"image generation error: {type(e).__name__}: {e}")

# ---------- Generic text fallbacks with provenance ----------
def run_chat_with_fallback(primary_model: str, prompt: str, hie[thotw.o pta:ou
t, str):
        return out
    if isinstance(out, list) and out:
        last = out[-1]
        if isinstance(last, dict):
            return last.get("generated_text") or last.get("caption") or str(last)
        return str(last)
    if isinstance(out, dict):
        return out.get("generated_text") or out.get("caption") or str(out)
    try:
        return "".join(token.get("generated_text", "") if isinstance(token, dict) else str(token) for token in out)
    except Exception# As a las: 
esort, if it's an iterator/generator, tr  to join tokens
    try       return str(out)

def _fetch_image_bytes_with_fallback(primary_url: str) -> bytes:
    """Fetch image bytes with robust headers and domain-agnostic fallbacks."
""
    urls: List[str] = [primary_url] + FALLBACK_IMAGE_URLS
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=30, headers=HTTP_IMAGE_HEADERS)
            resp.raise_for_status()
            if CAPTION_DEBUG and CAPTION_VERBOSE:
                print(f"[caption] fetched image from: {url}")
            return resp.content
        except Exception st_err = e
            continue
    raise last_err or RuntimeError("Unknown network error while fetching image")

def _encode_image_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format=fmt, qualit
y=95)
    return buf.getvalue()

def _attempt_caption(client: InferenceClient, inputs: List, model_tag: str, attempt_ix: int, meta: Dict) -> str:
    """Single attempt over m
ultiperyt representations; accumulate meta and return caor ''.""    try:
        Try capt o in twi=hnge_to_text(inps;)handl  Sto Ite     c =ion_outputfirst non-empty (out)
      if clast_err = None
    ap and cap.strip():
                meta["success_model"] = model_tag
    , wait_for_model=True            meta[tion"attempts"].append({"model": model_tag, "attempt"CAPTION_DEBUGus": "ok", "len": prinn(f"c}apti n  {urn}cgotatype={type(out).__name__}, normalized })
={    str( extionc)e"pt StopIteratiifoisinstanc (capeion, st
) a d    tion.strip():       meta["  _mr turnpct "ton})
            continue
        excet Exception"pes, errt=t,: f"error:{ty)f"CAPTION_DEBUG:
} nn    return "mrina(fi[c_ttimn]D{> N}nS o I CraTOon_D:t
ing AIxt inputVRBOSE:
        # Alread  y prg excvr  Ex ep if" as [ca       ptio lastterr = en(meta['a'])}")
    elseD
 UG  aiot oltort Counter
        status{oug}n rrorunttype(r).__(af"__}: {e['stsor a in ep'c]n in ep("[ciaosasm_err a:"n   N fr:k, v in sreun.)      tggl       perfce]m{type(lass__m
[e_ni(mL__}:r]lm.tIerret::
   ieurn "e
      of_ ca"ts"].apen(lle.        "requested_model": mo "s   try:
        img_bth_fallback(image_url)
    except Exception as e:
        meta["feror"] = f"ch and de{ode tteype(ee robustly (with head)rs + fallbacks).__name__}:{e}"
        return f"[image caption error] NetworkError: {type(e).__name__}: {e}", mepy)nvert("RGB")
        img_bth_fallback(image_url)
        meta["decode_error"] = f"{type
name__}:{e}"
        return f"[image caption error] InvalidImage: {type(e).__name__}: {e}", m_) g_bytes = _encode_image_bytes(img, "JPEG")
    inputs = [img, png_bytes, jpeg_

    # Prepareytultiple encodings so work around b]ckend quirks # trimmed representations

    # Remote cycles
    if USE_API:
        for mid in CAPTION_MODELS_
PIOR # Tri with ehemp   ary mo el firEt    cclnent== get_cl e_t(model_id)n, miprimary_captiond=,_try_taption(ctempt,,[img,mpne_byaes, jpeg)bytes,  mg_byt s],  ag=f"pri ary:{mo el_id}")
captii isinstance(primary_caption, stn) and:primary_c
p ion and no  pri ary_ca tion.startswi h("[image capt on error]"):
        retur  prima y_c ptio 

    # Fallback to an alternate captioning modelt    alt_modele=m"Salesforce/blip-image-captioning-base"
pts(alt_clientm=eget_client(alt_model)
    alt_            ryturn captalt_ion, met[amg, p
g_bytes, j eg_by e # iLg_bytes]alt g=f"alf:{ala_lod l}" 
W_  LOCisinstanAe(Llt_ca:
   , str) and alt_captionlandonotctltpcippion.seare(withC"[iAagP capTion error]"O:N_MODELS_g, mrnnrotrcrptvon

    # Rd(urn sh) "omt inforta
ive error we h ve   reiftprimary_caption:
urn "[image capprimnryerromiony  caifpfltpcrerion:
        r"turn al,_c ptionetany  caifpfltpcrerion:
        r"turn al,_c ptioneta

# In[ ]:


# Example usage (provide your HF token via environment or directly here)
# Option A: export HF_TOKEN in your environment before starting the notebook
# Option B: pass a token string to the constructor (not recommended for shared notebooks)
system = AdvancedMultiModalSystem()  # Set HF_TOKEN environment variable or pass hf_token="your_token"

# Example (image understanding) if token is set:
# Use a URL likely to succeed; falls back internally if blocked
sample_img_url = "https://picsum.photos/id/237/800/600"
response_route, result = system.process_message(
    "Describe this image",
    image_url=sample_img_url
)
print("Route:", response_route)
print("Model:", result.get("model"))
print("Output:", result.get("content") if result.get("type") == "text" else "[image]")
if "caption_meta" in result:
    print("Caption provenance meta (truncated):")
    meta = result["caption_meta"]
    print({k: meta[k] for k in meta if k != "attempts"})
    print(f"Attempts logged: {len(meta.get('attempts', []))}")

# ## End-to-end demo: route, choose model, show result
# 
# The next cell will iterate over our 7-route queries, display the predicted route, the chosen model, and the actual output (text or image). If `HF_TOKEN` is not set, remote calls will be skipped and text responses will show a placeholder.

# In[ ]:


# Run end-to-end over the multi-modal queries with provenance capture
system = AdvancedMultiModalSystem()  # Set HF_TOKEN environment variable or pass hf_token="your_token"
results_table = []

for q in multi_modal_queries:
    print("-" * 100)
    print(f"Query: {q}")
    # Provide a sample image URL for image understanding-style prompts
    img_url = None
    q_lower = q.lower()
    if ("photo" in q_lower or "image" in q_lower) and ("generate" not in q_lower and "draw" not in q_lower and "create" not in q_lower):
        img_url = "https://picsum.photos/id/237/800/600"  # robust dog image

    route, result = system.process_message(q, image_url=img_url)
    model_used = result.get("model")
    print(f"Predicted Route: {route}")
    print(f"Model: {model_used}")
    if result.get("type") == "text":
        print("Output:\n", result.get("content")[:500])
        # Summarize text provenance if present
        tmeta = result.get("text_meta")
        if tmeta:
            print("Text provenance:", {k: tmeta[k] for k in tmeta if k != "attempts"})
    elif result.get("type") == "image":
        print("Output: [image below]")
        try:
            ipy_display(result["image"])  # display image in notebook
        except Exception as e:
            print("(Could not display image)", e)
    else:
        print("Output: ", result)

    # Collect provenance for summary
    cap_meta = result.get("caption_meta")
    tmeta = result.get("text_meta")
    results_table.append({
        "query": q,
        "route": route,
        "model": model_used,
        "type": result.get("type"),
        "output_preview": (result.get("content") or "[image]")[:120],
        "caption_success_model": cap_meta.get("success_model") if cap_meta else None,
        "caption_attempts": len(cap_meta.get("attempts", [])) if cap_meta else None,
        "text_success_model": tmeta.get("success_model") if tmeta else None,
        "text_attempts": len(tmeta.get("attempts", [])) if tmeta else None,
    })

print("-" * 100)
print("End-to-end demo complete.")

# In[ ]:


# Summary DataFrame of routing + specialist outputs
import pandas as pd
if results_table:
    df_results = pd.DataFrame(results_table)
    display(df_results)
    # Show basic aggregation of success models
    agg = df_results.groupby(['route'])[['text_attempts','caption_attempts']].mean().reset_index()
    print("\nMean attempts per route:")
    display(agg)
else:
    print("No results captured yet.")

# ### Notes on inference routing
# 
# - The notebook now uses `huggingface_hub.InferenceClient`, which routes requests to the correct backend automatically. This avoids manual URL handling and fixes 404/410 issues.
# - Set `HF_TOKEN` in your environment for API access. Without it, the demo will skip remote calls and show placeholders.
# - You can swap model IDs in the TEXT_MODELS and IMAGE_MODELS maps to your preferred choices.

# ## How the caption fallback works
# 
# To keep the tutorial runnable end-to-end without silently skipping vision, the image-understanding path now:
# - Tries multiple remote caption models in priority order, each with a couple of retries.
# - Sends the image in a few representations (PIL object, PNG bytes, JPEG bytes) to avoid provider quirks.
# - Falls back to a local `transformers` pipeline if the remote providers return empty/StopIteration.
# - Records provenance (which model succeeded, attempt counts) so you can inspect behavior instead of just seeing an error.
# 
# You can tune:
# - `CAPTION_MODELS_PRIORITY`: add/remove models or reorder.
# - `CAPTION_RETRIES`: increase retries per model.
# - `CAPTION_DEBUG` and `CAPTION_VERBOSE`: adjust logging noise.
# - `CAPTION_ALLOW_LOCAL`: disable local fallback if you want purely-remote behavior.

# In[ ]:


# Diagnostics: probe each model capability
PROBE_TEXTS = {
    "general_chat": "Hello there, what's a fun fact about space?",
    "knowledge_qa": "Who discovered penicillin?",
    "code_generation": "Write a Python function that returns Fibonacci numbers up to n.",
    "code_debugging": "The following code throws an error, fix it:\nfor i in range(5) print(i)",
    "summarization": "Summarize: Machine learning enables computers to learn from data without being explicitly programmed, leading to advances in many fields.",
}

print("\n=== Capability Probe ===")
for route, text in PROBE_TEXTS.items():
    model_id = (CHAT_MODELS.get(route) or GENERATION_MODELS.get(route))
    if route in CHAT_MODELS:
        out, meta = run_chat_with_fallback(model_id, text, history=[])
    elif route == "summarization":
        out, meta = run_summarize_with_fallback(model_id, text)
    elif route == "code_generation":
        out, meta = run_text_with_fallback(model_id, text)
    elif route == "code_debugging":
        out, meta = run_chat_with_fallback(model_id, text, history=[])
    else:
        out, meta = run_chat_with_fallback(model_id, text, history=[])
    print(f"Route: {route}\nRequested Model: {model_id}\nSuccess Model: {meta.get('success_model')}\nOutput: {out[:260]}\nAttempts: {len(meta.get('attempts', []))}\n---")

# Vision probes (if token set)
if USE_API:
    try:
        cap, cmeta = hf_image_caption(IMAGE_MODELS["image_understanding"], "https://picsum.photos/id/237/800/600")
        print("Vision caption output:", cap[:180])
        print("Caption success model:", cmeta.get("success_model"))
    except Exception as e:
        print("Vision caption error:", e)

    try:
        gen_img = hf_image_generate(IMAGE_MODELS["image_generation"], "A serene lake at sunrise, painted in impressionist style")
        print("Generated image size:", gen_img.size)
    except Exception as e:
        print("Image generation error:", e)
else:
    print("Token not set; skipping vision probes.")

# In[ ]:




# In[ ]:



