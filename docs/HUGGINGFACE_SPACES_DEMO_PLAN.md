# Hugging Face Spaces Demo Plan

## Goal

Create an interactive, visual demo on Hugging Face Spaces that showcases:
1. **Multi-model routing** - How requests get routed to different LLMs
2. **Manager agent orchestration** - Task classification, interruption, and resumption
3. **Real-time visualization** - See the decision-making process as it happens

---

## Demo Concept

### What Users Will Experience

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🤖 Multi-Agent Orchestration Demo                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │  💬 Chat Input          │    │  📊 Live Routing Visualization     │ │
│  │                         │    │                                     │ │
│  │  [Your message here...] │    │     ┌─────────┐                     │ │
│  │                         │    │     │ Manager │ ←── Analyzing...   │ │
│  │  [Send]                 │    │     └────┬────┘                     │ │
│  │                         │    │          │                          │ │
│  └─────────────────────────┘    │    ┌─────┴─────┐                    │ │
│                                 │    ▼           ▼                    │ │
│  ┌─────────────────────────┐    │ ┌─────┐   ┌─────┐                   │ │
│  │  📝 Task Queue          │    │ │GPT-4│   │Claude│  ← Selected!    │ │
│  │                         │    │ └─────┘   └─────┘                   │ │
│  │  1. ✅ Quick lookup     │    │                                     │ │
│  │  2. 🔄 Report gen...    │    └─────────────────────────────────────┘ │
│  │  3. ⏸️  Paused task      │                                          │
│  │                         │    ┌─────────────────────────────────────┐ │
│  └─────────────────────────┘    │  🔍 Decision Log                    │ │
│                                 │                                     │ │
│                                 │  [12:01] Classified as: QUICK       │ │
│                                 │  [12:01] No interruption needed     │ │
│                                 │  [12:01] Routed to: gpt-4o-mini     │ │
│                                 │  [12:02] Response received (0.8s)   │ │
│                                 └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Architecture

### Option 1: Gradio (Recommended)

**Pros**: Native HF Spaces support, easy Python integration, good for demos
**Cons**: Limited real-time updates without workarounds

```
┌──────────────────────────────────────────────────┐
│              Hugging Face Spaces                  │
│  ┌────────────────────────────────────────────┐  │
│  │           Gradio Application               │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │         Frontend (Gradio UI)         │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  │                    │                       │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │    Manager Agent (Python Backend)    │  │  │
│  │  │    • Request Classification          │  │  │
│  │  │    • Routing Logic                   │  │  │
│  │  │    • Task Queue Management           │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  │                    │                       │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │         LLM Providers                │  │  │
│  │  │    (HuggingFace Inference API)       │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### Option 2: Streamlit

**Pros**: More flexible UI, better real-time updates
**Cons**: Slightly more complex, different paradigm

### Option 3: Custom Docker + FastAPI + React

**Pros**: Full control, best real-time experience
**Cons**: More development effort, complex deployment

---

## Recommended Approach: Gradio

### Why Gradio?

1. **Native HF Spaces support** - Zero config deployment
2. **Python-only** - Matches our existing codebase
3. **Built-in components** - Chatbot, plots, state management
4. **Free hosting** - HF Spaces free tier is sufficient

---

## Implementation Plan

### Phase 1: Core Demo (Week 1)

#### 1.1 Basic Chat Interface
```python
import gradio as gr

def chat(message, history):
    # Classify request
    # Route to appropriate model
    # Return response with metadata
    pass

demo = gr.ChatInterface(chat)
```

#### 1.2 Routing Visualization
- Show which model was selected
- Display classification result (QUICK/COMPLEX/URGENT)
- Animated flow diagram

#### 1.3 Task Queue Display
- List of pending/active/completed tasks
- Real-time status updates

### Phase 2: Interactive Features (Week 2)

#### 2.1 Model Selection Override
- Let users manually select models
- Compare responses from different models

#### 2.2 Interruption Demo
- Simulate long-running task
- Show interruption + checkpoint + resume

#### 2.3 Metrics Dashboard
- Response times per model
- Routing decisions breakdown
- Token usage

### Phase 3: Polish & Deploy (Week 3)

#### 3.1 UI/UX Improvements
- Animations for routing visualization
- Color-coded task states
- Mobile-responsive layout

#### 3.2 Documentation Tab
- In-app explanation of how it works
- Code snippets users can copy
- Link to full repo

#### 3.3 Deployment
- Push to HF Spaces
- Configure secrets (API keys)
- Test and iterate

---

## UI Components Breakdown

### Component 1: Chat Panel

```python
with gr.Column(scale=1):
    chatbot = gr.Chatbot(
        label="Multi-Agent Chat",
        height=400,
        avatar_images=("user.png", "agent.png")
    )
    msg = gr.Textbox(
        placeholder="Ask anything...",
        show_label=False
    )
    send_btn = gr.Button("Send", variant="primary")
```

### Component 2: Routing Visualization

```python
with gr.Column(scale=1):
    gr.Markdown("## 🔀 Routing Decision")
    
    classification = gr.Label(
        label="Request Type",
        value={"QUICK": 0.9, "COMPLEX": 0.1}
    )
    
    selected_model = gr.Textbox(
        label="Selected Model",
        value="gpt-4o-mini"
    )
    
    routing_diagram = gr.HTML(
        value=generate_routing_diagram()  # SVG/HTML
    )
```

### Component 3: Task Queue

```python
with gr.Column(scale=1):
    gr.Markdown("## 📋 Task Queue")
    
    task_table = gr.Dataframe(
        headers=["ID", "Status", "Type", "Model"],
        value=[
            ["#1", "✅ Complete", "QUICK", "gpt-4o-mini"],
            ["#2", "🔄 Running", "COMPLEX", "gpt-4o"],
            ["#3", "⏸️ Paused", "COMPLEX", "claude-3"],
        ]
    )
```

### Component 4: Decision Log

```python
with gr.Column():
    gr.Markdown("## 🔍 Decision Log")
    
    decision_log = gr.Textbox(
        label="",
        lines=10,
        interactive=False,
        value="[12:01:23] New request received\n[12:01:23] Classified as: QUICK\n..."
    )
```

---

## Demo Scenarios

### Scenario 1: Quick Question Routing

**User Input**: "What's 2+2?"

**Visualization Shows**:
1. Manager analyzes → Classification: QUICK (95% confidence)
2. No task running → No interruption needed
3. Route to: `gpt-4o-mini` (fastest for simple tasks)
4. Response time: 0.3s

### Scenario 2: Complex Task

**User Input**: "Analyze the pros and cons of microservices vs monolith architecture"

**Visualization Shows**:
1. Manager analyzes → Classification: COMPLEX (87% confidence)
2. Estimated time: 15-30 seconds
3. Added to queue as Task #5
4. Route to: `gpt-4o` (best for analysis)
5. Progress indicator while generating

### Scenario 3: Interruption Flow

**Setup**: Complex task running (e.g., "Write a detailed report...")

**User Input**: "Actually, what time is it?"

**Visualization Shows**:
1. Manager analyzes → Classification: QUICK
2. ⚠️ Task #5 is running (12s elapsed)
3. Decision: INTERRUPT (quick question, long-running task)
4. Checkpoint saved for Task #5
5. Handle quick question → Response
6. Resume Task #5 from checkpoint

---

## File Structure

```
huggingface-spaces-demo/
├── app.py                    # Main Gradio application
├── manager_agent.py          # Manager logic (from main repo)
├── routing_viz.py            # Visualization helpers
├── mock_llm.py               # Mock LLM for demo (no API needed)
├── requirements.txt          # Dependencies
├── README.md                 # HF Spaces README
├── assets/
│   ├── logo.png
│   ├── user_avatar.png
│   └── agent_avatar.png
└── examples/
    ├── quick_question.json
    ├── complex_task.json
    └── interruption.json
```

---

## API Key Strategy

### Option A: Free Demo Mode (No Keys Required)

Use mock responses or HuggingFace Inference API free tier:

```python
# Use HF Inference API (free, rate-limited)
from huggingface_hub import InferenceClient

client = InferenceClient()
response = client.text_generation(
    prompt,
    model="mistralai/Mistral-7B-Instruct-v0.2"
)
```

### Option B: Bring Your Own Key

Let users input their own API keys (stored in session only):

```python
with gr.Accordion("🔑 API Keys (Optional)", open=False):
    openai_key = gr.Textbox(
        label="OpenAI API Key",
        type="password"
    )
    anthropic_key = gr.Textbox(
        label="Anthropic API Key", 
        type="password"
    )
```

### Option C: HF Spaces Secrets

For the hosted demo, use HF Spaces secrets:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `HF_TOKEN`

---

## Deployment Checklist

### Pre-Deployment

- [ ] Test all demo scenarios locally
- [ ] Ensure no API keys in code
- [ ] Add rate limiting for free tier
- [ ] Create compelling README for HF Spaces
- [ ] Add social preview image

### HF Spaces Configuration

```yaml
# README.md header for HF Spaces
---
title: Multi-Agent Orchestration Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
license: apache-2.0
---
```

### Post-Deployment

- [ ] Test on HF Spaces
- [ ] Share link in repo README
- [ ] Create demo video/GIF
- [ ] Post on social media

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Page loads | 100+ in first week |
| Average session time | > 2 minutes |
| Demo completions | > 50% try all 3 scenarios |
| GitHub clicks | > 10% click through to repo |

---

## Timeline

| Week | Milestone |
|------|-----------|
| Week 1 | Core chat + routing visualization working |
| Week 2 | Task queue, interruption demo, polish |
| Week 3 | Deploy to HF Spaces, documentation, share |

---

## Open Questions

1. **Mock vs Real LLMs?**
   - Mock: Faster, free, predictable for demos
   - Real: More impressive, but costs money

2. **Single page or tabs?**
   - Single: Everything visible, can be crowded
   - Tabs: Cleaner, but less "wow" factor

3. **Mobile support?**
   - HF Spaces has mobile traffic
   - Gradio can be responsive with care

---

## Next Steps

1. **Review this plan** - Any features to add/remove?
2. **Decide on API strategy** - Mock, BYOK, or secrets?
3. **Create mockups** - Quick Figma/sketch of final UI
4. **Start Phase 1** - Basic chat + routing viz

---

## Appendix: Sample Code Snippets

### A. Main App Structure

```python
# app.py
import gradio as gr
from manager_agent import ManagerAgent
from routing_viz import create_routing_diagram

manager = ManagerAgent()

def process_message(message, history, task_queue_state):
    # 1. Classify
    classification = manager.classify(message)
    
    # 2. Check for interruption
    interrupt_decision = manager.check_interrupt(task_queue_state)
    
    # 3. Route and execute
    response, model_used = manager.route_and_execute(message)
    
    # 4. Update visualization
    viz_html = create_routing_diagram(
        classification=classification,
        model=model_used,
        interrupted=interrupt_decision
    )
    
    # 5. Update history
    history.append((message, response))
    
    return history, viz_html, update_task_queue(task_queue_state)

# Build UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Multi-Agent Orchestration Demo")
    
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Type a message...")
            send = gr.Button("Send")
        
        with gr.Column(scale=1):
            routing_viz = gr.HTML()
            task_queue = gr.Dataframe()
            decision_log = gr.Textbox(lines=8)
    
    send.click(
        process_message,
        inputs=[msg, chatbot, task_queue],
        outputs=[chatbot, routing_viz, task_queue]
    )

demo.launch()
```

### B. Routing Diagram Generator

```python
# routing_viz.py
def create_routing_diagram(classification, model, interrupted=False):
    """Generate SVG/HTML for routing visualization."""
    
    models = {
        "gpt-4o": {"color": "#10a37f", "x": 100},
        "gpt-4o-mini": {"color": "#10a37f", "x": 200},
        "claude-3": {"color": "#d97706", "x": 300},
    }
    
    selected = models.get(model, models["gpt-4o"])
    
    html = f"""
    <div style="text-align: center; padding: 20px;">
        <div style="margin-bottom: 20px;">
            <span style="
                background: {'#ef4444' if interrupted else '#22c55e'};
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
            ">
                {classification['type']} ({classification['confidence']:.0%})
            </span>
        </div>
        
        <div style="font-size: 24px; margin: 20px 0;">
            ↓
        </div>
        
        <div style="
            display: inline-block;
            background: {selected['color']};
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: bold;
        ">
            {model}
        </div>
    </div>
    """
    
    return html
```

### C. Mock LLM for Demo

```python
# mock_llm.py
import random
import time

MOCK_RESPONSES = {
    "quick": [
        "The answer is 42!",
        "Here's a quick response for you.",
        "Done! That was easy.",
    ],
    "complex": [
        "Let me analyze this in detail...\n\n**Analysis:**\n1. First point\n2. Second point\n3. Conclusion",
        "This requires careful consideration...\n\nAfter thorough analysis, here's my recommendation...",
    ]
}

def mock_llm_response(message, model, classification):
    """Simulate LLM response with realistic delays."""
    
    if classification["type"] == "QUICK":
        time.sleep(random.uniform(0.3, 0.8))
        return random.choice(MOCK_RESPONSES["quick"])
    else:
        time.sleep(random.uniform(1.5, 3.0))
        return random.choice(MOCK_RESPONSES["complex"])
```

---

*Document Version: 1.0*
*Last Updated: November 2025*
