# Intelligent Manager Agent for Multi-Agent Orchestration

## Overview

This document describes how to build an intelligent "Manager" agent that routes requests through **Arch Gateway** using dynamic model routing. The Manager serves as the central orchestrator for a multi-agent system, analyzing requests and routing them to appropriate workers.

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ARCH GATEWAY (Proxy)                        │
│  • Guardrails & Safety                                           │
│  • Request Routing                                               │
│  • Observability                                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MANAGER AGENT                              │
│  • Analyzes & classifies requests                                │
│  • Routes to appropriate workers                                 │
│  • Manages task interruption/resumption                          │
└─────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        ┌──────────┐      ┌──────────┐      ┌──────────┐
        │ Worker 1 │      │ Worker 2 │      │ Worker 3 │
        │ (Fast)   │      │ (Complex)│      │ (Special)│
        └──────────┘      └──────────┘      └──────────┘
```

---

## What This Solves

| Problem | Solution |
|---------|----------|
| Long tasks block quick questions | Manager interrupts & queues intelligently |
| Routing logic scattered in code | Centralized in Arch config + Manager |
| No visibility into agent decisions | Arch provides tracing & metrics |
| Hard to swap/add LLM providers | Arch abstracts provider details |

---

## How It Works (Simple Version)

### 1. Request Comes In
User asks: *"What's the weather in NYC?"*

### 2. Arch Gateway Processes It
- Applies safety guardrails
- Routes to Manager agent
- Logs everything for observability

### 3. Manager Agent Decides
- **Quick question?** → Handle immediately
- **Complex task?** → Queue it, assign to worker
- **Interruption needed?** → Checkpoint current task, handle new one

### 4. Response Returns
Through Arch Gateway back to user.

---

## Integration with Arch Gateway

### Option A: Manager as a Prompt Target

The Manager agent can be configured as a prompt target in Arch, making it the default handler for all incoming requests:

```yaml
# arch_config.yaml
version: v0.1.0

listeners:
  ingress_traffic:
    address: 0.0.0.0
    port: 10000
    message_format: openai

llm_providers:
  - model: openai/gpt-4o
    access_key: $OPENAI_API_KEY
    routing_preferences:
      - name: orchestration
        description: task routing, coordination, and decision-making

  - model: openai/gpt-4o-mini
    access_key: $OPENAI_API_KEY
    routing_preferences:
      - name: quick_responses
        description: simple questions, status checks, brief answers

prompt_targets:
  - name: manager_agent
    description: Central orchestrator for all requests
    endpoint:
      name: manager_service
      path: /analyze
    system_prompt: |
      You are a Manager agent. Analyze requests and route appropriately.

endpoints:
  manager_service:
    endpoint: localhost:8080
    protocol: http
```

### Option B: Manager Behind Arch (Recommended)

Use Arch for LLM access, with the Manager running as a separate service:

```
User → Arch Gateway → Your App → Manager Agent → Arch Gateway → LLMs
                                       ↓
                                  Worker Agents
```

---

## Manager Agent Code Structure

```python
# manager_agent.py
from openai import OpenAI
from dataclasses import dataclass
from enum import Enum

# Connect to Arch Gateway instead of OpenAI directly
client = OpenAI(
    base_url="http://localhost:12000/v1",  # Arch egress port
    api_key="not-needed"  # Arch handles auth
)

class RequestType(Enum):
    QUICK = "quick"      # < 5 seconds, handle immediately
    COMPLEX = "complex"  # Long-running, queue it
    URGENT = "urgent"    # Interrupt current work

@dataclass
class RoutingDecision:
    request_type: RequestType
    should_interrupt: bool
    target_worker: str
    reason: str

class ManagerAgent:
    def __init__(self):
        self.current_task = None
        self.task_queue = []
    
    def analyze_request(self, user_message: str) -> RoutingDecision:
        """Use LLM to classify the request."""
        
        response = client.chat.completions.create(
            model="orchestration",  # Arch routes to best model
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        
        return self._parse_decision(response)
    
    def route(self, decision: RoutingDecision):
        """Execute the routing decision."""
        
        if decision.should_interrupt and self.current_task:
            self._checkpoint_current_task()
        
        if decision.request_type == RequestType.QUICK:
            return self._handle_immediately(decision)
        else:
            return self._queue_task(decision)
```

---

## Key Components

### 1. Request Classifier

```python
CLASSIFICATION_PROMPT = """
Classify the user's request into one of these categories:

QUICK: Simple questions, status checks, brief lookups
- "What time is it?"
- "How many items in my cart?"

COMPLEX: Multi-step tasks, analysis, generation
- "Write a report on Q3 sales"
- "Analyze this dataset"

URGENT: Time-sensitive, needs immediate attention
- "Stop the current process!"
- "Emergency: server down"

Respond with JSON:
{
    "type": "quick|complex|urgent",
    "estimated_seconds": <number>,
    "reason": "<brief explanation>"
}
"""
```

### 2. Interruption Logic

```python
def should_interrupt(self, new_request: RoutingDecision) -> bool:
    """Decide if we should interrupt the current task."""
    
    if not self.current_task:
        return False
    
    # Always interrupt for urgent requests
    if new_request.request_type == RequestType.URGENT:
        return True
    
    # Interrupt long tasks for quick questions
    if (new_request.request_type == RequestType.QUICK and 
        self.current_task.elapsed_time > 30):
        return True
    
    return False
```

### 3. State Persistence Integration

```python
def _checkpoint_current_task(self):
    """Save task state before interruption."""
    
    checkpoint = {
        "task_id": self.current_task.id,
        "state": self.current_task.get_state(),
        "progress": self.current_task.progress,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to your persistence layer
    self.state_store.save_checkpoint(checkpoint)
    
    # Pause the task
    self.current_task.pause()
```

---

## Using with Omni Router

This repo contains an **Omni Router** that demonstrates multi-model routing. Here's how the Manager integrates:

```python
from omni_router_dev import OmniModelRouter

class ManagerAgent:
    def __init__(self):
        # Use Omni Router for intelligent model selection
        self.router = OmniModelRouter()
    
    def process_request(self, message: str):
        # Router picks best model based on task
        response = self.router.route_and_execute(
            message=message,
            task_type=self._classify_task(message)
        )
        return response
```

---

## Complete Integration Example

### Step 1: Start Arch Gateway

```bash
pip install archgw
archgw up arch_config.yaml
```

### Step 2: Run Manager Service

```python
# manager_service.py
from fastapi import FastAPI
from manager_agent import ManagerAgent

app = FastAPI()
manager = ManagerAgent()

@app.post("/analyze")
async def analyze_request(request: dict):
    message = request["messages"][-1]["content"]
    
    # Classify and route
    decision = manager.analyze_request(message)
    result = manager.route(decision)
    
    return {"response": result}
```

### Step 3: Connect Everything

```python
# client.py
from openai import OpenAI

# All requests go through Arch
client = OpenAI(
    base_url="http://localhost:10000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="gpt-4o",  # Arch handles routing
    messages=[{"role": "user", "content": "Analyze my sales data"}]
)
```

---

## Configuration Reference

### Arch Config for Multi-Agent System

```yaml
version: v0.1.0

listeners:
  # External requests come here
  ingress_traffic:
    address: 0.0.0.0
    port: 10000
    message_format: openai

  # Internal LLM calls go here  
  egress_traffic:
    address: 0.0.0.0
    port: 12000
    message_format: openai

llm_providers:
  - model: openai/gpt-4o
    access_key: $OPENAI_API_KEY
    routing_preferences:
      - name: complex_reasoning
        description: analysis, planning, decision-making

  - model: openai/gpt-4o-mini
    access_key: $OPENAI_API_KEY
    routing_preferences:
      - name: quick_tasks
        description: simple responses, status updates

  - model: anthropic/claude-3-5-sonnet
    access_key: $ANTHROPIC_API_KEY
    routing_preferences:
      - name: long_context
        description: document analysis, lengthy conversations

prompt_guards:
  input_guards:
    jailbreak:
      on_exception:
        message: I can only help with authorized tasks.

prompt_targets:
  - name: manager
    description: Route all requests through the Manager agent
    endpoint:
      name: manager_service
      path: /analyze

endpoints:
  manager_service:
    endpoint: localhost:8080
    protocol: http
```

---

## Benefits of This Architecture

| Feature | Without Arch | With Arch |
|---------|-------------|-----------|
| **Add new LLM** | Change code everywhere | Update config, restart |
| **Guardrails** | Implement in each agent | Centralized, consistent |
| **Observability** | Build custom logging | Built-in tracing |
| **Routing** | Hardcoded if/else | Preference-based, dynamic |
| **Scaling** | Complex load balancing | Arch handles it |

---

## Next Steps

1. **Clone Arch Gateway**: `git clone https://github.com/katanemo/archgw`
2. **Set up config**: Create `arch_config.yaml` with your providers
3. **Build Manager Agent**: Use the code structure above
4. **Connect workers**: Route tasks to specialized agents
5. **Add persistence**: Integrate with your state management system

---

## Related Files in This Repo

| File | Purpose |
|------|---------|
| `omni_router_dev.py` | Multi-model routing logic |
| `route_config.json` | Routing configuration |
| `huggingchat_omni_tutorial.ipynb` | Interactive tutorial |

---

## Resources

- [Arch Gateway Docs](https://docs.archgw.com)
- [Arch GitHub](https://github.com/katanemo/archgw)
- [LangChain Integration Guide](https://python.langchain.com/)
