# ☄️ PRECEPT: Compass-Optimized Memory Evolution for Test-time learning

## Overview

**PRECEPT** is a unified framework that combines:
- **COMPASS** (Genetic-Pareto Optimization) for offline prompt optimization ("Compiler")
- **Evo-Memory/ReMem** for online experience learning ("Runtime")

This creates a **Self-Optimizing Agent** that addresses two fundamental limitations of current AI agents:

1. **Static Intelligence**: Agents usually have fixed prompts that don't improve → Addressed by COMPASS
2. **Catastrophic Forgetting**: Agents fail to retain lessons from past tasks → Addressed by Evo-Memory

## Three-Stream Ingestion Architecture

In PRECEPT, **"Ingestion" is NOT a single event**. It splits into three distinct streams because the agent handles **external data** (documents/facts) differently from **internal wisdom** (experiences).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ☄️ PRECEPT Ingestion Architecture                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐                                                │
│  │  1. HARD INGESTION  │  Pre-Deployment (External Pipeline)            │
│  │    (Knowledge)      │                                                │
│  │                     │  Raw Documents → Chunks → Embeddings → VectorDB│
│  │  PDFs, APIs, Docs   │  The agent is the READER, not the LIBRARIAN   │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    STATIC KNOWLEDGE BASE                         │   │
│  │                       (Vector DB)                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │  2. SOFT INGESTION  │───▶│   DUAL RETRIEVAL    │                    │
│  │    (Wisdom)         │    │                     │                    │
│  │                     │    │  Vector DB + Patches│                    │
│  │  Corrections,       │    └─────────────────────┘                    │
│  │  Warnings,          │              │                                 │
│  │  Experience Patches │              ▼                                 │
│  │                     │    ┌─────────────────────┐                    │
│  │  Runtime (ReMem     │◀───│   THINK-ACT-REFINE  │                    │
│  │  Refine Step)       │    │      (ReMem)        │                    │
│  └─────────────────────┘    └─────────────────────┘                    │
│                                       │                                 │
│                                       ▼                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │ 3. FEEDBACK         │◀───│   EXECUTION TRACES  │                    │
│  │    INGESTION        │    │   (Logs & Metrics)  │                    │
│  │    (Training)       │    └─────────────────────┘                    │
│  │                     │              │                                 │
│  │  COMPASS Phase      │              ▼                                 │
│  │  (Batch/Nightly)    │    ┌─────────────────────┐                    │
│  │                     │───▶│   PROMPT EVOLUTION  │                    │
│  │  Analyze patterns,  │    │     (COMPASS)       │                    │
│  │  Evolve prompts     │    └─────────────────────┘                    │
│  └─────────────────────┘                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. Hard Ingestion (Knowledge Stream)

**When:** Pre-Deployment (Asynchronous/External Pipeline)  
**What:** Raw documents (PDFs, APIs) → Vector DB  
**Why:** Keep heavy processing OUT of the inference loop  
**Who:** External ETL pipeline (NOT the agent)


```python
# Hard Ingestion happens OUTSIDE the agent
from precept import DefaultHardIngestionPipeline

pipeline = DefaultHardIngestionPipeline(
    vector_store=my_vector_store,
    embedding_fn=embed_text,
)

# Run before deployment (async, can take hours)
chunks = await pipeline.ingest_document("shipping_tariffs_2025.pdf")
```

### 2. Soft Ingestion (Wisdom Stream)

**When:** Evo-Memory Phase (Runtime/Real-Time)  
**What:** Meta-data, corrections, experience "patches"  
**Why:** Instantly "patch" flaws in static knowledge WITHOUT re-indexing  
**Who:** Agent (during ReMem Refine step)

```python
from precept import SoftIngestionManager

soft_manager = SoftIngestionManager()

# Agent discovers Vector DB info is outdated
# During the REFINE step, it creates a "patch"
result = soft_manager.ingest_correction(
    target_document_id="hamburg_port_manual",
    correction="Hamburg shows 'operational' but has hidden strike delays",
    source_task="Route pharma shipment Hamburg→Boston",
    source_observation="Booking failed due to undocumented strike",
    confidence=0.95,
    domain="shipping",
)

# Later retrievals automatically include this patch
patches = soft_manager.get_patches_for_retrieval(
    query="Ship via Hamburg",
    document_ids=["hamburg_port_manual"],
    domain="shipping",
)
# Returns: [⚠️ CORRECTION: Hamburg shows 'operational' but has hidden strike delays]
```

### 3. Feedback Ingestion (Training Stream)

**When:** COMPASS Phase (Optimization/Batch)  
**What:** Execution traces, success/failure logs  
**Why:** Teach the system HOW to search better  
**Who:** COMPASS optimizer

```python
from precept import FeedbackIngestionManager, ExecutionTrace

feedback_manager = FeedbackIngestionManager()

# Record execution traces
trace = ExecutionTrace(
    id="trace_001",
    task="Route pharma Hamburg→Boston",
    goal="Speed priority",
    domain="shipping",
    steps=[...],
    total_steps=5,
    success=True,
    final_answer="Routed via Rotterdam (avoided Hamburg strike)",
    confidence=0.9,
    documents_retrieved=["hamburg_manual", "rotterdam_manual"],
    patches_applied=["patch_hamburg_strike"],
    execution_time_ms=1500,
    llm_calls=3,
    tokens_used=2500,
    success_factors=["Used experience patch to avoid Hamburg"],
    failure_factors=[],
)
feedback_manager.ingest_trace(trace)

# COMPASS analyzes patterns (nightly batch)
analysis = feedback_manager.analyze_patterns()
recommendations = feedback_manager.get_consolidation_recommendations()
# Returns: [{"type": "add_instruction", "content": "Always bypass Hamburg for Speed shipments"}]
```

## The Complete PRECEPT Workflow

### Illustrative Example: "Port Strike" Scenario

```
DAY 1 (THE CRISIS)
==================

1. HARD INGESTION (Pre-deployed)
   └── "2025 Tariff Schedule" already in Vector DB
   └── Says: "Hamburg is fully operational"

2. SOFT INGESTION (Runtime - Agent learns)
   └── Agent tries to book via Hamburg → FAILS
   └── Agent creates PATCH: "Hamburg blocked despite manual"
   └── Next query automatically sees the patch
   └── Agent survives using high-latency "memory checks"

NIGHT 1 (THE OPTIMIZATION)
==========================

3. FEEDBACK INGESTION (COMPASS analyzes)
   └── COMPASS ingests 50 traces showing Hamburg failures
   └── Finds pattern: "Avoid Hamburg" appears in 80% of successes
   └── EVOLVES prompt: "CRITICAL: Bypass Hamburg for Speed shipments"

DAY 2 (THE EVOLVED STATE)
=========================

4. CONSOLIDATED WISDOM (Agent wakes up smarter)
   └── New prompt includes: "Auto-bypass Hamburg"
   └── Agent avoids Hamburg INSTANTLY (zero memory lookups)
   └── Old "Hamburg blocked" patches PRUNED from memory
   └── Memory capacity freed for NEXT crisis (e.g., fuel shortage)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ☄️ PRECEPT Unified Framework                    │
│     Compass-Optimized Memory Evolution for Test-time learning    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │   COMPILER LAYER     │    │      RUNTIME LAYER            │  │
│  │      (COMPASS)        │    │      (Evo-Memory)             │  │
│  │                      │    │                               │  │
│  │  • Pareto Selection  │    │  • Episodic Memory Store     │  │
│  │  • Prompt Evolution  │    │  • ReMem Pipeline            │  │
│  │  • Feedback          │    │  • Think-Act-Refine Loop     │  │
│  │    Ingestion         │    │  • Soft Ingestion (Patches)  │  │
│  └──────────┬───────────┘    └──────────────┬────────────────┘  │
│             │                               │                    │
│             └───────────┬───────────────────┘                    │
│                         │                                        │
│             ┌───────────▼───────────┐                           │
│             │  CONSOLIDATION LAYER  │                           │
│             │                       │                           │
│             │  • Frequency Analysis │                           │
│             │  • Rule Extraction    │                           │
│             │  • Patch → Prompt     │                           │
│             │  • Memory Pruning     │                           │
│             └───────────────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Ingestion Summary Table

| Type | Phase | What is Ingested? | Why? |
|------|-------|-------------------|------|
| **Hard Ingestion** | External/Pre-Process | Raw Documents (PDFs, APIs) | Keep heavy processing out of inference loop |
| **Soft Ingestion** | Evo-Memory Phase | Meta-Data & Corrections | Instantly "patch" flaws without re-indexing |
| **Feedback Ingestion** | COMPASS Phase | Execution Traces | Teach system HOW to search better |

## Key Components

### Memory Store (`memory_store.py`)
- **EpisodicMemory**: Stores experiences with domain/skill indexing
- **SemanticMemoryIndex**: Embedding-based retrieval
- **Experience**: Rich data model for learned experiences

### Ingestion Module (`ingestion.py`)
- **HardIngestionPipeline**: Document ingestion (external)
- **SoftIngestionManager**: Experience patches (runtime)
- **FeedbackIngestionManager**: Trace analysis (COMPASS)
- **PRECEPTIngestionCoordinator**: Unified interface

### ReMem Pipeline (`remem_pipeline.py`)
- **ThinkActRefineLoop**: Core execution loop
- **ReMemThought**: Reasoning about tasks using memory
- **ReMemAction**: Action execution with memory guidance
- **ReMemRefinement**: Learning summary after task completion

### Memory Consolidation (`memory_consolidation.py`)
- **FrequencyAnalyzer**: Identifies patterns for consolidation
- **MemoryConsolidator**: Extracts rules and mutates prompts
- **ConsolidatedRule**: Generalizable rules from experience

### Pareto Memory (`pareto_memory.py`)
- **ParetoMemoryManager**: Manages prompt versions from COMPASS
- **TaskTypeRouter**: Routes tasks to optimal prompts
- **PromptVersion**: Tracks prompt performance by task type

### PRECEPT Orchestrator (`precept_orchestrator.py`)
- **PRECEPTOrchestrator**: Main coordinator
- Manages all components
- Handles the optimization cycle

### COMPASS Bridge (`compass_bridge.py`)
- **COMPASSBridge**: Integration with COMPASS retriever
- Imports/exports between systems

## Quick Start

```python
import asyncio
from precept import (
    PRECEPTOrchestrator, 
    PRECEPTConfig,
    PRECEPTIngestionCoordinator,
    SoftIngestionManager,
)

async def main():
    # Configure the agent
    config = PRECEPTConfig(
        max_memories=1000,
        consolidation_interval=50,
        enable_prompt_routing=True,
    )
    
    # Create orchestrator
    agent = PRECEPTOrchestrator(
        llm_client=your_llm_client,
        config=config,
    )
    
    # Set initial prompts
    agent.set_system_prompts({
        "system": "You are a helpful assistant that learns from experience."
    })
    
    # Create soft ingestion manager for runtime patches
    soft_manager = SoftIngestionManager()
    
    # Run tasks (agent learns continuously)
    for task in tasks:
        result = await agent.run_task(
            task=task["description"],
            goal=task["goal"],
            domain=task["domain"],
        )
        
        # If agent discovered something wrong with retrieved docs,
        # it creates a soft patch during the Refine step
        if result.discovered_issue:
            soft_manager.ingest_correction(
                target_document_id=result.problematic_doc_id,
                correction=result.correction_text,
                source_task=task["description"],
                source_observation=result.observation,
            )
    
    # Check improvement
    report = agent.get_improvement_report()
    print(f"Success rate improved by: {report['improvement']:.1%}")

asyncio.run(main())
```

## Benefits Summary

| Feature | COMPASS Alone | Evo-Memory Alone | **PRECEPT** |
|---------|------------|------------------|-----------|
| **Adaptation Speed** | Slow | Fast | **Instant & Permanent** |
| **Token Efficiency** | High | Low | **Optimal** |
| **Generalization** | High | Low | **High** |
| **Knowledge Patching** | ❌ | ✅ (temporary) | **✅ (temporary → permanent)** |

## References

- **COMPASS/GEPA Paper**: "Reflective Prompt Evolution Can Outperform Reinforcement Learning"
- **Evo-Memory Paper**: ["Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory"](https://arxiv.org/html/2511.20857v1)

## File Structure

```
src/precept/
├── __init__.py              # Package exports
├── memory_store.py          # Episodic memory storage
├── ingestion.py             # Three-stream ingestion architecture
├── remem_pipeline.py        # Think-Act-Refine loop
├── memory_consolidation.py  # Memory baking
├── pareto_memory.py         # Prompt routing
├── precept_orchestrator.py    # Main orchestrator
├── compass_bridge.py        # COMPASS integration
└── README.md                # This file

src/configs/
└── precept_config.py          # Configuration presets

examples/
└── precept_example.py         # Usage examples
```

## Name Origin

**PRECEPT** = **C**ompass-**O**ptimized **M**emory **E**volution for **T**est-time learning

Like a precept blazing through the sky, PRECEPT agents continuously improve and adapt, leaving a trail of learned experiences that illuminate future tasks. ☄️
