# PRECEPT: Planning Resilience via Experience, Context Engineering & Probing Trajectories

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/AutoGen-0.7+-green.svg" alt="AutoGen">
  <img src="https://img.shields.io/badge/MCP-1.0+-purple.svg" alt="MCP">
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg" alt="ChromaDB">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-red.svg" alt="OpenAI">
  <img src="https://img.shields.io/badge/uv-Package%20Manager-yellow.svg" alt="uv">
</p>

## Paper

- **arXiv:** https://arxiv.org/abs/2603.09641
- **Code:** https://github.com/arash-shahmansoori/precept-framework
- **Framework overview image:** `paper/arxiv/PRECEPT_framework_linkedin.png`

## Overview

**PRECEPT** (**P**lanning **R**esilience via **E**xperience, **C**ontext **E**ngineering & **P**robing **T**rajectories) is a comprehensive AI agent learning framework that combines:

- **COMPASS (Novel Enhanced GEPA)**: Our improved Genetic-Pareto prompt optimization with ML-based complexity detection, smart rollout allocation, and multi-strategy coordination
- **Memory Evolution**: Episodic memory with semantic retrieval that evolves over time
- **Test-time Learning**: Learn and adapt during inference, not just training
- **AutoGen Integration**: Microsoft's multi-agent framework for scalable agent orchestration
- **MCP (Model Context Protocol)**: Standardized tool integration protocol
- **Context Engineering**: Google Whitepaper patterns for production efficiency

### Key Features

| Feature | Description |
|---------|-------------|
| **Episodic Memory** | Learn from past experiences with semantic retrieval |
| **Rule Learning** | Extract patterns from errors automatically |
| **COMPASS Evolution** | Enhanced GEPA with ML complexity detection, smart rollouts, LLM-powered mutation |
| **Memory Consolidation** | Merge duplicate patterns into rules |
| **Procedural Memory** | Store "how-to" strategies |
| **Dual Retrieval** | Combine static knowledge + dynamic experiences with conflict detection |
| **Conflict Resolution** | Bayesian-informed, evidence-based conflict handling (no hardcoded logic) |
| **Context Engineering** | Google Whitepaper efficiency patterns |
| **Persistence** | ChromaDB vector stores + JSON backups persist across sessions |
| **Strategy Pattern** | Pluggable domain strategies for any use case |
| **Modular Architecture** | Clean separation of concerns |
| **Deterministic Pruning** | Prevents "dumb retries" via constraint tracking |
| **Structured Logging** | Production-ready logging with JSON/colored output |
| **Configuration-Driven** | Modular config system with dependency injection |

### PRECEPT vs Baselines

| Method | Cross-Task Learning | Cross-Episode Memory | Dual Retrieval | Conflict Resolution | Deterministic Pruning | COMPASS Evolution |
|--------|---------------------|---------------------|----------------|---------------------|----------------------|-------------------|
| ExpeL | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Reflexion | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **PRECEPT** | ✅ | ✅ | ✅ | ✅ Bayesian | ✅ | ✅ |

**Baselines**:
- **ExpeL**: Experience-based learning with cross-episode memory but no rule compilation
- **Reflexion**: Verbal reinforcement learning with reflective memory across episodes

### COMPASS vs Basic GEPA

COMPASS is our novel enhancement of the GEPA (Genetic-Pareto) algorithm, extending it with ML-based optimization and intelligent resource allocation:

| Aspect | Basic GEPA | COMPASS (Our Enhancement) |
|--------|-----------|---------------------------|
| Complexity Detection | Fixed 2 hops | ML-based 3+ hops |
| Rollout Allocation | 15 rollouts | 2 rollouts (86% fewer) |
| Strategy Coordination | Single strategy | Multi-strategy coordination |
| Prompt Evolution | Basic mutation | LLM-powered reflection + mutation |
| Accuracy | 50% | 100% |
| Speed | Baseline | 6.7x faster |

---

## Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e3f2fd', 'primaryTextColor': '#000', 'primaryBorderColor': '#1565c0', 'lineColor': '#333', 'secondaryColor': '#f3e5f5', 'tertiaryColor': '#fff3e0'}}}%%
flowchart TB
    subgraph Client["AutoGen PRECEPT Agent"]
        Strategy["DomainStrategy"]
        Agent["PRECEPTAgent"]
        Baseline1["ExpeL Agent"]
        Baseline2["Reflexion Agent"]
    end

    subgraph Server["PRECEPT MCP Server"]
        subgraph Memory["Memory System"]
            ChromaDB[("ChromaDB Vector DB")]
            Episodic["Episodic Memory"]
        end

        subgraph Compass["COMPASS Enhanced GEPA"]
            Evolution["LLM-Powered Evolution"]
            Complexity["ML Complexity Analyzer"]
            Rollout["Smart Rollout Strategy"]
            MultiStrategy["Multi-Strategy Coordinator"]
        end

        subgraph Learning["Learning Engine"]
            Consolidation["Memory Consolidation"]
            RuleParser["Dynamic Rule Parser"]
        end

        subgraph Context["Context Engineering"]
            Reactive["Reactive Retrieval"]
            Background["Background Writes"]
            Procedural["Procedural Memory"]
        end
    end

    subgraph Persistence["Persistence Layer"]
        DB1[("chroma_precept/")]
        DB2[("experiences.json")]
        DB3[("learned_rules.json")]
        DB4[("procedures.json")]
    end

    Client <-->|"STDIO/MCP Protocol"| Server
    ChromaDB --> DB1
    Episodic --> DB2
    Evolution --> DB3
    Procedural --> DB4

    classDef default fill:#fff,stroke:#333,stroke-width:1px,color:#000
    classDef clientClass fill:#bbdefb,stroke:#1565c0,color:#000
    classDef serverClass fill:#e1bee7,stroke:#7b1fa2,color:#000
    classDef persistClass fill:#ffe0b2,stroke:#e65100,color:#000

    style Client fill:#bbdefb,stroke:#1565c0,color:#000
    style Server fill:#e1bee7,stroke:#7b1fa2,color:#000
    style Persistence fill:#ffe0b2,stroke:#e65100,color:#000
    style Compass fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
    style Memory fill:#e3f2fd,stroke:#1976d2,color:#000
    style Learning fill:#f3e5f5,stroke:#7b1fa2,color:#000
    style Context fill:#fff3e0,stroke:#ef6c00,color:#000
```

### System Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'actorTextColor': '#000', 'actorBkg': '#e3f2fd', 'actorBorder': '#1565c0', 'signalColor': '#333', 'signalTextColor': '#000', 'labelTextColor': '#000', 'noteBkgColor': '#fff3e0', 'noteTextColor': '#000'}}}%%
sequenceDiagram
    participant User
    participant AutoGen as AutoGen PRECEPT Agent
    participant Strategy as DomainStrategy
    participant MCP as MCP Client
    participant Server as PRECEPT MCP Server
    participant ChromaDB as ChromaDB
    participant OpenAI as OpenAI API

    User->>AutoGen: Task: Book Rotterdam to Boston
    AutoGen->>Strategy: parse_task(task)
    Strategy-->>AutoGen: ParsedTask(origin=rotterdam)

    AutoGen->>MCP: get_learned_rules()
    MCP->>Server: MCP Call
    Server-->>MCP: Rules: R-482 Use Antwerp

    AutoGen->>Strategy: apply_rule(rule, task)
    Strategy-->>AutoGen: Modified task, use Antwerp

    AutoGen->>MCP: book_shipment(antwerp, boston)
    MCP->>Server: MCP Call
    Server-->>MCP: Success

    AutoGen->>MCP: store_experience(success)
    MCP->>Server: Background Write
    Server->>ChromaDB: Store Embedding

    AutoGen-->>User: Booking Complete (1 step)
```

---

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key
- `uv` package manager (**required** for this project)

### Step 1: Clone the Repository

```bash
git clone https://github.com/arash-shahmansoori/precept-framework.git
cd precept-framework
```

### Step 2: Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### Step 3: Install Dependencies with uv

```bash
# Install project with all dependencies (recommended)
uv pip install -e . --system

# Or for development (includes pytest, black, ruff, etc.)
uv pip install -e ".[dev]" --system

# Or use uv sync for lockfile-based installs
uv sync
```

All PRECEPT dependencies are defined in `pyproject.toml`:

```toml
# PRECEPT Framework Dependencies
"openai>=1.50.0",       # LLM clients (AsyncOpenAI, NOT_GIVEN)
"pydantic>=2.0.0",      # Structured outputs
"numpy>=1.24.0",        # Memory operations
"mcp>=1.0.0",           # MCP server/client
"autogen-agentchat>=0.7.0",  # AutoGen integration
"autogen-ext>=0.7.0",   # AutoGen OpenAI client
"anyio>=4.0.0",         # Async MCP operations
"aiohttp",              # Async HTTP
"tiktoken",             # Token counting
"chromadb",             # Vector database
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
# Required: OpenAI API key (for LLM and embeddings)
OPENAI_API_KEY=sk-your-api-key-here

# Optional: Model overrides (defaults shown)
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional: Alternative LLM providers
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
TOGETHER_API_KEY=your-together-key

# Optional: For real API calls (not simulation)
LOGISTICS_API_URL=https://api.your-logistics-provider.com
LOGISTICS_API_KEY=your-api-key
```

### Step 5: Verify Installation

```bash
# Run the example
uv run examples/precept_autogen_mcp_full.py
```

---

## Quick Start

### Running the Full Demo

```bash
cd precept-framework

# Basic usage (sequential training, 6 train / 4 test, static knowledge enabled)
uv run examples/precept_autogen_mcp_full.py --domain logistics

# Custom train/test split with reproducible seed
uv run examples/precept_autogen_mcp_full.py -d logistics --train 20 --test 15 --seed 42

# With static knowledge (default) - dual retrieval + conflict resolution
uv run examples/precept_autogen_mcp_full.py -d logistics --static-knowledge --train 20 --test 15 --seed 42

# Without static knowledge (pure dynamic learning)
uv run examples/precept_autogen_mcp_full.py -d logistics --no-static-knowledge --train 20 --test 15 --seed 42

# Concurrent training ("Tesla Fleet" mode) + concurrent testing
uv run examples/precept_autogen_mcp_full.py -d logistics -ct -tw 4 -c -w 4 --seed 42

# Full concurrent with static knowledge
uv run examples/precept_autogen_mcp_full.py -d logistics --static-knowledge -ct -tw 4 -c -w 4 --seed 42

# List all available domains
uv run examples/precept_autogen_mcp_full.py --list

# Verbose output with log file
uv run examples/precept_autogen_mcp_full.py -d logistics --verbose --log-file run.log
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--domain` | `-d` | Domain to test | logistics |
| `--train` | | Number of training tasks | 6 |
| `--test` | | Number of test tasks | 4 |
| `--static-knowledge` | `-sk` | Enable static knowledge for all agents | True |
| `--no-static-knowledge` | `-nsk` | Disable static knowledge (pure dynamic) | False |
| `--concurrent` | `-c` | Enable concurrent testing | False |
| `--workers` | `-w` | Concurrent test workers (1-4) | 4 |
| `--concurrent-training` | `-ct` | Enable "Tesla Fleet" training | False |
| `--training-workers` | `-tw` | Concurrent training workers (1-6) | 2 |
| `--agent-workers` | `-aw` | Internal agent concurrency (1-5) | 3 |
| `--seed` | `-s` | Random seed for reproducibility | None |
| `--verbose` | `-v` | Enable DEBUG logging | False |
| `--log-file` | | Log file path (JSON format) | None |
| `--list` | `-l` | List available domains | |

---

## PRECEPT Stages

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e8f5e9', 'primaryTextColor': '#000', 'primaryBorderColor': '#2e7d32', 'lineColor': '#333'}}}%%
flowchart LR
    subgraph Online["Online Learning ReMem"]
        Think["Think"]
        Act["Act"]
        Refine["Refine"]
        Think --> Act --> Refine --> Think
    end

    subgraph Compass["COMPASS Enhanced GEPA"]
        Evolution["LLM Evolution"]
        Complexity["ML Complexity"]
        Rollouts["Smart Rollouts"]
        Strategy["Multi-Strategy"]
        Evolution --> Complexity --> Rollouts --> Strategy
    end

    subgraph Memory["Memory System"]
        Episodic["Episodic Memory"]
        Procedural["Procedural Memory"]
        Rules["Learned Rules"]
    end

    Online -->|"Store Experience"| Memory
    Memory -->|"Retrieve Context"| Online
    Compass -->|"Optimize Decisions"| Online
    Memory -->|"Analyze Patterns"| Compass

    classDef default fill:#fff,stroke:#333,stroke-width:1px,color:#000

    style Online fill:#c8e6c9,stroke:#2e7d32,color:#000
    style Compass fill:#ffe0b2,stroke:#ef6c00,color:#000
    style Memory fill:#bbdefb,stroke:#1565c0,color:#000
```

### 1. COMPASS (Novel Enhanced GEPA)

COMPASS is our novel improvement over the baseline GEPA algorithm, adding several key capabilities:

- **LLM-Powered Evolution**: Reflection on failures with intelligent prompt mutation
- **ML-based Complexity Detection**: Analyzes task complexity (3+ hops vs fixed 2)
- **Smart Rollout Allocation**: Adaptive rollouts (2 vs 15, 86% fewer)
- **Multi-Strategy Coordination**: Coordinates multiple strategies for optimal results
- **Pareto-based Selection**: Optimal candidate selection across multiple objectives

### 2. Evo-Memory (ReMem)
- Episodic memory with semantic retrieval
- Stores task descriptions, outcomes, and strategies
- Vector similarity search using OpenAI embeddings

### 3. Dynamic Rule Parser
- **NO hardcoded knowledge** - learns from rule text dynamically
- Extracts error codes, blocked keywords, and alternatives
- Domain-agnostic pattern extraction

### 4. Vector Database (ChromaDB)
- Persistent embeddings in `data/chroma_precept/`
- Text-embedding-3-small for vectorization
- Semantic similarity search

### 5. Dual Retrieval & Static Knowledge

PRECEPT uniquely combines THREE knowledge sources:

| Source | Description | Example |
|--------|-------------|---------|
| **Static Knowledge** | Pre-ingested facts (port info, APIs, rules) | "Hamburg handles pharma cargo" |
| **Dynamic Experiences** | Learned from task execution | "Hamburg has strikes, use Rotterdam" |
| **Episodic Memory** | Full task trajectories with outcomes | Task strategy that succeeded |

**Cutting-Edge Conflict Resolution**: PRECEPT includes a Bayesian-informed conflict resolution system:

| Capability | Description |
|------------|-------------|
| **Bayesian Uncertainty** | Quantifies confidence using Beta distributions |
| **Evidence-Based Prioritization** | Weighs evidence strength from multiple sources |
| **Anomaly Detection** | Detects when dynamic knowledge may be an outlier |
| **Dynamic Reliability Learning** | Updates source reliability based on outcomes |
| **Thompson Sampling** | Active exploration for re-verification |
| **Multi-Strategy Resolution** | Recency, reliability, specificity, LLM-merge |

```bash
# Static knowledge is loaded when --static-knowledge flag is used
data/static_knowledge/logistics_kb.json   # Port facts, procedures, alerts

# Dynamic knowledge is learned during execution
data/chroma_precept/                      # Learned experiences (vector DB)
data/chroma_static_knowledge/             # Static knowledge (vector DB, lazy-init)
data/precept_experiences.json             # Episodic memory (JSON backup)
```

**Note**: The `chroma_static_knowledge/` directory is only created when `--static-knowledge` is used, preventing unnecessary resource allocation in pure dynamic learning mode.

### 6. Memory Consolidation
- Merge duplicate patterns
- Prune low-value memories
- Bake lessons into rules

### Learning Timeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4CAF50', 'primaryTextColor': '#000', 'primaryBorderColor': '#2e7d32', 'taskTextColor': '#000', 'taskTextOutsideColor': '#000', 'sectionBkgColor': '#e3f2fd', 'altSectionBkgColor': '#f3e5f5', 'gridColor': '#ccc', 'todayLineColor': '#f44336'}}}%%
gantt
    title PRECEPT Learning Over 8 Tasks
    dateFormat X
    axisFormat %s

    section Baselines
    ExpeL/Reflexion retry loop    :crit, b1, 0, 8

    section PRECEPT
    Discovery Phase               :active, c1, 0, 2
    Learning Phase                :c2, 2, 4
    Optimized Phase               :done, c3, 4, 8
```

---

## Strategy Pattern Architecture

PRECEPT uses the **Strategy Pattern** for pluggable, domain-specific behavior:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e3f2fd', 'primaryTextColor': '#000', 'primaryBorderColor': '#1565c0', 'lineColor': '#333', 'classText': '#000'}}}%%
classDiagram
    class DomainStrategy {
        <<abstract>>
        +domain_name: str
        +get_system_prompt() str
        +get_available_actions() List
        +parse_task(task) ParsedTask
        +create_tools() List
        +apply_rule(rule, task) str
    }

    class BaselineDomainStrategy {
        <<abstract>>
        +domain_name: str
        +get_available_options() List
        +get_default_option() str
        +execute_option(option) ActionResult
    }

    class LogisticsDomainStrategy {
        +ORIGIN_PORTS: List
        +DESTINATION_PORTS: List
    }

    class FinanceDomainStrategy {
        +SYMBOLS: List
        +EXCHANGES: List
    }

    class BookingDomainStrategy {
        +AIRLINES: List
        +HOTELS: List
    }

    DomainStrategy <|-- LogisticsDomainStrategy
    DomainStrategy <|-- FinanceDomainStrategy
    DomainStrategy <|-- BookingDomainStrategy
    DomainStrategy <|-- CodingDomainStrategy
    DomainStrategy <|-- DevOpsDomainStrategy
    DomainStrategy <|-- IntegrationDomainStrategy

    BaselineDomainStrategy <|-- LogisticsBaselineStrategy
    BaselineDomainStrategy <|-- FinanceBaselineStrategy
    BaselineDomainStrategy <|-- BookingBaselineStrategy
```

### Available Domain Strategies

| Domain | Learning Strategy | Baseline Strategy | Use Case |
|--------|------------------|-------------------|----------|
| **Logistics** | `LogisticsDomainStrategy` | `LogisticsBaselineStrategy` | Shipping, ports, routes |
| **Coding** | `CodingDomainStrategy` | `CodingBaselineStrategy` | Code generation, debugging |
| **DevOps** | `DevOpsDomainStrategy` | `DevOpsBaselineStrategy` | Cloud, deployment, CI/CD |
| **Finance** | `FinanceDomainStrategy` | `FinanceBaselineStrategy` | Trading, payments, compliance |
| **Booking** | `BookingDomainStrategy` | `BookingBaselineStrategy` | Hotels, flights, reservations |
| **Integration** | `IntegrationDomainStrategy` | `IntegrationBaselineStrategy` | APIs, webhooks, data sync |

### Using Different Domains

```python
from precept import (
    PRECEPTAgent,
    # Baseline agents
    ExpeL,      # Experience-based learning
    Reflexion,  # Verbal reinforcement learning
    # Domain strategies
    get_domain_strategy,
    get_baseline_strategy,
    list_available_domains,
)
from precept.config import PreceptConfig, BaselineConfig

# Option 1: Using the registry (recommended)
strategy = get_domain_strategy("logistics")
baseline_strategy = get_baseline_strategy("logistics")

# Option 2: With custom configuration
config = PreceptConfig()
config.agent.max_attempts = 5
config.agent.enable_llm_reasoning = True

precept_agent = PRECEPTAgent(
    domain_strategy=strategy,
    config=config,
    server_script=SERVER_SCRIPT,
)

# Baseline agents with configuration
baseline_config = BaselineConfig(max_attempts=4, verbose=True)
expel_agent = ExpeL(
    baseline_strategy=baseline_strategy,
    config=baseline_config,
)
reflexion_agent = Reflexion(
    baseline_strategy=baseline_strategy,
    config=baseline_config,
)

# List all available domains
print(list_available_domains())
# ['logistics', 'coding', 'devops', 'finance', 'booking', 'integration']
```

### Configuration System

```python
from precept.config import (
    PreceptConfig,     # Aggregates all config
    AgentConfig,       # PRECEPT agent settings
    BaselineConfig,    # Baseline settings
    LLMConfig,         # LLM model settings
    ConstraintConfig,  # Deterministic pruning
    PromptTemplates,   # Centralized prompts
    setup_logging,     # Logging configuration
    get_logger,        # Get a logger instance
)

# Create full configuration
config = PreceptConfig()

# Customize agent settings
config.agent.max_attempts = 5
config.agent.consolidation_interval = 3
config.agent.compass_evolution_interval = 2
config.agent.enable_llm_reasoning = True
config.agent.max_internal_workers = 4

# Customize LLM settings
config.llm.model = "gpt-4o-mini"
config.llm.temperature = 0.7
config.llm.max_tokens = 500

# Setup logging
setup_logging(level="DEBUG", log_file="experiment.log")
logger = get_logger("precept.my_module")
```

---

## Context Engineering (Google Whitepaper)

The framework implements efficiency patterns from Google's Context Engineering whitepaper:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e3f2fd', 'primaryTextColor': '#000', 'primaryBorderColor': '#1565c0', 'lineColor': '#333'}}}%%
flowchart TB
    subgraph Patterns["Context Engineering Patterns"]
        subgraph Retrieval["Smart Retrieval"]
            MAT["Memory-as-a-Tool"]
            RR["Reactive Retrieval"]
        end

        subgraph Writing["Efficient Writing"]
            BW["Background Writes"]
            SC["Session Compaction"]
        end

        subgraph Learning["Knowledge Management"]
            PM["Procedural Memory"]
            SMC["Smart Consolidation"]
            MS["Memory Scoping"]
        end
    end

    Task["New Task"] --> MAT
    MAT -->|"Need context?"| RR
    RR -->|"Yes"| DB[("ChromaDB")]
    RR -->|"No"| Agent["Agent"]
    DB --> Agent
    Agent -->|"Complete"| BW
    BW -->|"Async"| DB
    Agent -->|"Long trajectory"| SC
    SC --> PM
    PM --> SMC
    SMC -->|"Conflicts?"| MS

    classDef default fill:#fff,stroke:#333,stroke-width:1px,color:#000

    style Patterns fill:#f5f5f5,stroke:#616161,color:#000
    style Retrieval fill:#bbdefb,stroke:#1565c0,color:#000
    style Writing fill:#c8e6c9,stroke:#2e7d32,color:#000
    style Learning fill:#ffe0b2,stroke:#ef6c00,color:#000
```

| Pattern | Description | Benefit |
|---------|-------------|---------|
| **Memory-as-a-Tool** | Agent decides WHEN to retrieve | Reduced latency |
| **Reactive Retrieval** | Skip unnecessary retrievals | 15-30% faster |
| **Background Writes** | Async memory storage | Non-blocking |
| **Session Compaction** | Compress long trajectories | Token savings |
| **Procedural Memory** | Store strategies, not just facts | Better reuse |
| **Smart Consolidation** | Detect duplicates/conflicts | Cleaner memory |

---

## MCP Server Tools

The PRECEPT MCP Server exposes 15+ tools:

### Memory Tools
| Tool | Description |
|------|-------------|
| `retrieve_memories` | Dual retrieval (static + dynamic + episodic) with conflict detection |
| `retrieve_with_dual_mode` | Explicit dual retrieval with source attribution |
| `store_experience` | Background-optimized memory storage with ChromaDB persistence |
| `get_learned_rules` | Get compiled rules from past errors |
| `update_memory_usefulness` | Update memory usefulness feedback |

### Static Knowledge Tools
| Tool | Description |
|------|-------------|
| `ingest_static_knowledge` | Ingest JSON/text knowledge items into ChromaDB (lazy-initialized) |
| `ingest_document_to_knowledge_base` | Ingest documents (PDF, Markdown, Web, 80+ formats) |
| `get_document_processor_info` | Show available document processors and formats |
| `get_static_knowledge_stats` | Get statistics about static knowledge base |

### Conflict Resolution Tools
| Tool | Description |
|------|-------------|
| `get_conflict_resolution_stats` | Get Bayesian conflict resolution statistics |

### Learning Tools
| Tool | Description |
|------|-------------|
| `record_error` | Record error for pattern learning |
| `record_solution` | Record successful solution for rule compilation |
| `trigger_consolidation` | LLM-powered memory consolidation |
| `trigger_compass_evolution` | COMPASS prompt evolution (enhanced GEPA) |
| `get_evolved_prompt` | Get evolved system prompt with rules |

### Domain Tools
| Tool | Description |
|------|-------------|
| `book_shipment` | Book logistics shipment (with API gateway) |
| `check_port` | Check port availability |
| `clear_customs` | Handle customs clearance |

### Procedural Tools
| Tool | Description |
|------|-------------|
| `store_procedure` | Store how-to strategy |
| `get_procedure` | Retrieve matching procedures |
| `get_server_stats` | Full server statistics

### File Locking (Production)

All JSON file operations use `fcntl.flock` for atomic reads/writes, ensuring thread-safety in concurrent training scenarios.

---

## File Structure

```
precept-framework/
├── src/
│   ├── precept/                         # Main PRECEPT package
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   CONFIGURATION (Modular Config System)
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── config/
│   │   │   ├── __init__.py              # Re-exports all config classes
│   │   │   ├── agent.py                 # AgentConfig (PRECEPT agent settings)
│   │   │   ├── baseline.py              # BaselineConfig (baseline settings)
│   │   │   ├── constraints.py           # ConstraintConfig (pruning settings)
│   │   │   ├── llm.py                   # LLMConfig (model, temperature, tokens)
│   │   │   ├── logging.py               # LogConfig, setup_logging, get_logger
│   │   │   ├── paths.py                 # DataPaths, get_project_root
│   │   │   ├── prompts.py               # PromptTemplates (centralized prompts)
│   │   │   ├── unified.py               # PreceptConfig (aggregates all configs)
│   │   │   ├── multi_condition.py       # Multi-condition experiment config
│   │   │   ├── dynamic_static_knowledge.py  # Dynamic static knowledge config
│   │   │   ├── logistics.py             # Logistics domain config
│   │   │   ├── finance.py               # Finance domain config
│   │   │   ├── booking.py               # Booking domain config
│   │   │   ├── coding.py                # Coding domain config
│   │   │   ├── devops.py                # DevOps domain config
│   │   │   ├── integration.py           # Integration domain config
│   │   │   └── factual_extraction.py    # Factual extraction config
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   DOMAIN STRATEGIES (Strategy Pattern)
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── domain_strategies/
│   │   │   ├── __init__.py              # Package exports
│   │   │   ├── base.py                  # DomainStrategy, BaselineDomainStrategy
│   │   │   ├── logistics.py             # LogisticsDomainStrategy
│   │   │   ├── coding.py                # CodingDomainStrategy
│   │   │   ├── devops.py                # DevOpsDomainStrategy
│   │   │   ├── finance.py               # FinanceDomainStrategy
│   │   │   ├── booking.py               # BookingDomainStrategy
│   │   │   └── integration.py           # IntegrationDomainStrategy
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   AUTOGEN AGENTS
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── precept_agent.py             # PRECEPTAgent (learning agent)
│   │   ├── autogen_precept_agent.py     # AutoGen-integrated PRECEPT agent
│   │   ├── autogen_integration.py       # AutoGen framework integration
│   │   ├── agent_functions.py           # Pure functions for PRECEPT agent
│   │   ├── baseline_agents.py           # ExpeL, Reflexion baselines
│   │   ├── baseline_functions.py        # Pure functions for baselines
│   │   ├── constraints.py               # RefineInterceptor (deterministic pruning)
│   │   ├── structured_outputs.py        # Pydantic models for LLM responses
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   MCP INTEGRATION (Model Context Protocol)
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── precept_mcp_server.py        # Standalone MCP server with 15+ tools
│   │   ├── precept_mcp_client.py        # MCP client wrapper
│   │   ├── compass_mcp_client.py        # PRECEPTMCPClientWithCOMPASS
│   │   ├── mcp_remem_integration.py     # MCP-wrapped ReMem Think-Act-Refine
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   DOCUMENT PROCESSING
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── document_processors/
│   │   │   ├── __init__.py              # Package exports
│   │   │   ├── base.py                  # Base document processor
│   │   │   ├── chunkers.py              # Text chunking strategies
│   │   │   ├── factory.py               # Processor factory
│   │   │   ├── langchain_adapter.py     # LangChain compatibility
│   │   │   ├── registry.py              # Processor registry
│   │   │   └── processors/              # Format-specific processors
│   │   │       ├── __init__.py
│   │   │       ├── markdown.py          # Markdown processor
│   │   │       ├── pdf.py               # PDF processor
│   │   │       ├── text.py              # Plain text processor
│   │   │       └── web.py               # Web content processor
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   RULE LEARNING
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── rule_parser.py               # DynamicRuleParser (NO hardcoded knowledge)
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   SCENARIO GENERATION (All 6 Domains)
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── scenario_generators/
│   │   │   ├── __init__.py              # Package exports
│   │   │   ├── logistics.py             # LogisticsScenarioGenerator
│   │   │   ├── coding.py                # CodingScenarioGenerator
│   │   │   ├── devops.py                # DevOpsScenarioGenerator
│   │   │   ├── finance.py               # FinanceScenarioGenerator
│   │   │   ├── booking.py               # BookingScenarioGenerator
│   │   │   └── integration.py           # IntegrationScenarioGenerator
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   STRATEGY REGISTRY
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── strategy_registry.py         # get_domain_strategy, list_available_domains
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   CORE MEMORY SYSTEM
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── memory_store.py              # Episodic memory with semantic indexing
│   │   ├── memory_consolidation.py      # LLM-powered memory consolidation
│   │   ├── pareto_memory.py             # Pareto-optimal memory management
│   │   ├── ingestion.py                 # Soft ingestion + dual retrieval
│   │   ├── conflict_resolution.py       # Bayesian conflict resolution
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   COMPASS (Enhanced GEPA with ML Optimization)
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── gepa.py                      # COMPASS evolution engine (LLM reflection)
│   │   ├── complexity_analyzer.py       # ML-based complexity detection
│   │   ├── compass_integration.py       # Smart rollouts & multi-strategy coordination
│   │   ├── compass_controller.py        # COMPASS controller
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   REMEM PIPELINE (Think-Act-Refine)
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── remem_pipeline.py            # ReMem loop with online learning
│   │   ├── precept_orchestrator.py      # Full PRECEPT orchestrator
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   CONTEXT ENGINEERING (Google Whitepaper)
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── context_engineering.py       # Efficiency patterns
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   CODE EXECUTION
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── code_executor.py             # Code execution engine
│   │   ├── single_turn_docker_agent.py  # Single-turn Docker execution
│   │   ├── multiturn_docker_agent.py    # Multi-turn Docker execution
│   │   ├── execution_tracer.py          # Execution tracing
│   │   ├── execution_feedback_processor.py  # Feedback processing
│   │   ├── dynamic_coding_config.py     # Dynamic coding configuration
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   UTILITIES & INFRASTRUCTURE
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── llm_clients.py               # OpenAI LLM + embedding clients (lazy-init)
│   │   ├── black_swan_gen.py            # Black Swan scenario generation
│   │   ├── scoring.py                   # Experience scoring functions
│   │   ├── simulation.py                # Black Swan world simulation
│   │   ├── baselines.py                 # RAG+Tools baseline for comparison
│   │   ├── csp_constraint_manager.py    # CSP constraint management
│   │   │
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   │   PACKAGE
│   │   │   ═══════════════════════════════════════════════════════════════
│   │   ├── __init__.py                  # Package exports (all public APIs)
│   │   └── README_PRECEPT.md            # Internal module documentation
│   │
│   └── models/                          # LLM API Wrappers (Multiple Providers)
│       ├── __init__.py                  # Unified model exports
│       ├── openai_api.py                # OpenAI chat + structured outputs
│       ├── anthropic_api.py             # Anthropic Claude integration
│       ├── gemini_api.py                # Google Gemini integration
│       ├── groq_api.py                  # Groq API integration
│       ├── together_ai_api_models.py    # Together AI integration
│       └── embedding_models.py          # OpenAI embeddings (lazy-init)
│
├── scripts/                             # Publication experiment scripts
│   ├── README_EXPERIMENTS.md            # Experiment documentation
│   ├── run_exp1_main_comparison.py      # Exp1: Main comparison (PRECEPT vs baselines)
│   ├── run_exp2_static_knowledge_ablation.py  # Exp2: Static knowledge ablation
│   ├── run_exp3_training_size_ablation.py     # Exp3: Training size ablation
│   ├── run_exp4_continuous_learning.py  # Exp4: Continuous learning
│   ├── run_exp5_model_ablation.py       # Exp5: Model ablation
│   ├── run_exp6_compositional_generalization.py  # Exp6: Compositional generalization
│   ├── run_exp7_rule_drift.py           # Exp7: Rule drift adaptation
│   ├── generate_exp6_main_figure.py     # Generate Exp6 publication figures
│   ├── generate_exp6_figures.py         # Additional Exp6 figures
│   ├── generate_exp7_main_figure.py     # Generate Exp7 publication figures
│   ├── test_hybrid_parsing.py           # Standalone hybrid parsing test (requires API)
│   ├── test_precept_workflow.py         # Standalone workflow test (requires API)
│   └── create_results/                  # Result generation scripts
│       ├── README.md
│       ├── generate_consolidated_results.py
│       ├── generate_exp1_main_comparison_results.py
│       ├── generate_exp2_static_knowledge_results.py
│       ├── generate_exp3_training_size_results.py
│       ├── generate_exp4_continuous_learning_results.py
│       └── generate_exp5_model_ablation_results.py
│
├── examples/
│   ├── precept_autogen_mcp_full.py      # PRIMARY: Full comparison (PRECEPT vs ExpeL vs Reflexion)
│   └── config/
│       ├── __init__.py                  # Experiment config exports
│       ├── experiment.py                # ExperimentConfig, DATA_DIR, SERVER_SCRIPT
│       └── display.py                   # ExperimentDisplay, ProgressTracker
│
├── data/                                # Auto-created persistence directory
│   ├── chroma_precept/                  # Vector DB for PRECEPT experiences
│   ├── chroma_expel/                    # Vector DB for ExpeL experiences
│   ├── chroma_full_reflexion/           # Vector DB for Reflexion experiences
│   ├── static_knowledge/                # Source static knowledge files
│   │   └── *.json                       # Domain knowledge (ports, procedures, alerts)
│   ├── publication_results/             # Publication experiment results
│   │   ├── exp1_*/                      # Main comparison results
│   │   ├── exp2_*/                      # Static knowledge ablation results
│   │   ├── exp3_*/                      # Training size ablation results
│   │   ├── exp4_*/                      # Continuous learning results
│   │   ├── exp5_*/                      # Model ablation results
│   │   ├── exp6_*/                      # Compositional generalization results
│   │   └── exp7_*/                      # Rule drift results
│   └── experiment_results_*.json        # Saved experiment results
│
├── tests/                               # Comprehensive test suite (512 tests)
│   ├── conftest.py                      # Shared fixtures, path setup
│   ├── fixtures/
│   │   ├── __init__.py
│   │   └── mock_data.py                 # Sample tasks, errors, rules
│   ├── unit/                            # Unit tests (pure functions)
│   │   ├── config/                      # Configuration tests
│   │   │   ├── test_agent.py
│   │   │   ├── test_baseline.py
│   │   │   ├── test_constraints.py
│   │   │   ├── test_llm.py
│   │   │   └── test_unified.py
│   │   ├── domain_strategies/
│   │   │   └── test_logistics.py
│   │   ├── test_agent_functions.py      # Agent logic tests
│   │   ├── test_baseline_functions.py   # Baseline logic tests
│   │   ├── test_black_swan_gen.py       # Black Swan CSP generator tests
│   │   ├── test_compass_integration.py  # COMPASS integration tests
│   │   ├── test_conflict_resolution.py  # Conflict resolution tests
│   │   ├── test_constraints.py          # Pruning tests
│   │   ├── test_csp_constraint_manager.py # CSP constraint manager tests
│   │   ├── test_gepa_detailed.py        # GEPA/COMPASS tests
│   │   ├── test_ingestion_detailed.py   # Ingestion tests
│   │   ├── test_memory_store_detailed.py # Memory store tests
│   │   ├── test_precept_modules.py      # Module import tests
│   │   ├── test_rule_parser_detailed.py # Rule parser tests
│   │   ├── test_scenario_generators.py  # Scenario generation tests
│   │   └── test_structured_outputs.py   # LLM output tests
│   └── integration/                     # Integration tests
│       ├── test_baseline_agents.py      # Baseline agent tests
│       ├── test_deterministic_keys.py   # Deterministic key verification
│       ├── test_learning_integration.py # Learning component tests
│       ├── test_mcp_mock_integration.py # MCP mock tests
│       ├── test_precept_agent.py        # Full agent tests
│       └── test_workflow_integration.py # End-to-end workflow tests
│
├── docs/                                # Documentation
│   ├── EXPERIMENT_ANALYSIS.md           # Experiment results and analysis
│   ├── PUBLICATION_EXPERIMENT_STRATEGY.md   # Publication experiment strategy
│   ├── PUBLICATION_READY_GUIDELINES.md  # Publication guidelines
│   ├── ABLATION_STUDIES_SUMMARY.md      # Ablation study summary
│   ├── THEORETICAL_BOUNDS.md            # Theoretical analysis
│   ├── CONTINUOUS_LEARNING.md           # Continuous learning documentation
│   ├── RECOMMENDED_HYPERPARAMETERS.md   # Hyperparameter recommendations
│   └── *.md                             # Additional documentation files
│
├── .env                                 # API keys (not in git)
├── .python-version                      # Python version pin (3.12 for uv)
├── pyproject.toml                       # Project config (uv dependencies)
└── README.md                            # This file
```

### Module Dependency Graph

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e3f2fd', 'primaryTextColor': '#000', 'primaryBorderColor': '#1565c0', 'lineColor': '#333'}}}%%
flowchart TD
    Demo["precept_autogen_mcp_full.py"]

    subgraph Config["Configuration"]
        ConfigMod["config/"]
        Logging["logging.py"]
    end

    subgraph Strategies["Domain Strategies"]
        DomainStrat["domain_strategies/"]
        Registry["strategy_registry.py"]
        RuleParser["rule_parser.py"]
    end

    subgraph Agents["Agents"]
        PRECEPTAgent["precept_agent.py"]
        AgentFuncs["agent_functions.py"]
        BaselineAgents["baseline_agents.py"]
        Constraints["constraints.py"]
    end

    subgraph MCP["MCP Integration"]
        MCPClient["compass_mcp_client.py"]
        MCPServer["precept_mcp_server.py"]
    end

    subgraph Core["Core Memory"]
        MemStore["memory_store.py"]
        Ingestion["ingestion.py"]
    end

    subgraph Compass["COMPASS Enhanced GEPA"]
        GEPA["gepa.py"]
        Complexity["complexity_analyzer.py"]
        CompassInt["compass_integration.py"]
    end

    subgraph Learning["Learning and Context"]
        Consolidation["memory_consolidation.py"]
        ContextEng["context_engineering.py"]
    end

    Demo --> PRECEPTAgent
    Demo --> BaselineAgents
    Demo --> Registry
    Demo --> Config
    PRECEPTAgent --> DomainStrat
    PRECEPTAgent --> MCPClient
    PRECEPTAgent --> AgentFuncs
    PRECEPTAgent --> Constraints
    PRECEPTAgent --> Config
    AgentFuncs --> RuleParser
    BaselineAgents --> DomainStrat
    BaselineAgents --> Config
    MCPClient --> MCPServer
    MCPClient --> Compass
    MCPServer --> MemStore
    MCPServer --> GEPA
    MCPServer --> ContextEng
    MemStore --> Ingestion

    classDef default fill:#fff,stroke:#333,stroke-width:1px,color:#000

    style Demo fill:#c8e6c9,stroke:#2e7d32,color:#000
    style Config fill:#e1bee7,stroke:#7b1fa2,color:#000
    style Strategies fill:#bbdefb,stroke:#1565c0,color:#000
    style Agents fill:#ffe0b2,stroke:#ef6c00,color:#000
    style MCP fill:#e1bee7,stroke:#7b1fa2,color:#000
    style Core fill:#f8bbd9,stroke:#c2185b,color:#000
    style Learning fill:#c8e6c9,stroke:#388e3c,color:#000
    style Compass fill:#fff9c4,stroke:#ffa000,color:#000
```

### Baseline Agents Comparison

| Agent | Learning Type | Cross-Task | Cross-Episode | Pruning |
|-------|---------------|------------|---------------|---------|
| **ExpeL** | Experience-based learning | ❌ | ✅ | ❌ |
| **Reflexion** | Verbal reinforcement | ❌ | ✅ | ❌ |
| **PRECEPTAgent** | Full learning stack | ✅ | ✅ | ✅ |

```python
# All methods use the SAME LLM call budget
# The difference is WHAT they learn:

# 1. ExpeL (Experience Learning)
#    - Stores experiences and insights across episodes
#    - Retrieves relevant past experiences
#    - NO rule compilation, NO deterministic pruning

# 2. Reflexion (Verbal Reinforcement)
#    - Cross-episode memory for task types
#    - Verbal reflections persist across episodes
#    - NO rule compilation, NO deterministic pruning

# 3. PRECEPTAgent (Full Stack)
#    - Cross-task learning (rules apply everywhere)
#    - Memory consolidation (patterns → rules)
#    - COMPASS prompt evolution
#    - Deterministic pruning (prevents dumb retries)
```

---

## Testing

PRECEPT includes a comprehensive test suite following software engineering best practices.

### Test Suite Structure

The test suite includes **512 tests** covering all PRECEPT components.

```
tests/
├── conftest.py                    # Shared fixtures and pytest configuration
├── fixtures/
│   ├── __init__.py
│   └── mock_data.py               # Sample data for tests
├── unit/                          # Unit tests (pure functions)
│   ├── config/                    # Configuration module tests
│   │   ├── test_agent.py
│   │   ├── test_baseline.py
│   │   ├── test_constraints.py
│   │   ├── test_llm.py
│   │   └── test_unified.py
│   ├── domain_strategies/
│   │   └── test_logistics.py
│   ├── test_agent_functions.py    # Core agent logic tests
│   ├── test_baseline_functions.py # Baseline agent logic tests
│   ├── test_black_swan_gen.py     # Black Swan CSP generator tests
│   ├── test_compass_integration.py # COMPASS integration tests
│   ├── test_conflict_resolution.py # Conflict resolution tests
│   ├── test_constraints.py        # Pruning/constraint tests
│   ├── test_csp_constraint_manager.py # CSP constraint manager tests
│   ├── test_gepa_detailed.py      # GEPA/COMPASS tests
│   ├── test_ingestion_detailed.py # Ingestion tests
│   ├── test_memory_store_detailed.py # Memory store tests
│   ├── test_precept_modules.py    # Core module import tests
│   ├── test_rule_parser_detailed.py # Rule parser tests
│   ├── test_scenario_generators.py # Scenario generation tests
│   └── test_structured_outputs.py # LLM output parsing tests
└── integration/                   # Integration tests
    ├── test_baseline_agents.py    # Baseline agent tests
    ├── test_deterministic_keys.py # Deterministic key verification
    ├── test_learning_integration.py    # Learning component tests
    ├── test_mcp_mock_integration.py    # MCP mock tests
    ├── test_precept_agent.py      # PRECEPTAgent tests
    └── test_workflow_integration.py    # End-to-end workflow tests
```

### Run Unit & Integration Tests

```bash
# Run all PRECEPT tests
uv run pytest tests/unit tests/integration -v

# Run with coverage report
uv run pytest tests/unit tests/integration --cov=precept --cov-report=html

# Run specific test modules
uv run pytest tests/unit/test_agent_functions.py -v

# Run only fast unit tests
uv run pytest tests/unit -v --ignore=tests/integration

# Run tests matching a pattern
uv run pytest tests/ -k "test_parse" -v
```

### Run Full Integration Test (Experiment)

```bash
# Default domain (logistics), sequential training
uv run examples/precept_autogen_mcp_full.py

# With reproducible seed
uv run examples/precept_autogen_mcp_full.py --domain logistics --seed 42
```

### Run with Different Domains

```bash
# Logistics (default) - port closures, customs, black swan shipping
uv run examples/precept_autogen_mcp_full.py --domain logistics --train 8 --test 4

# Coding - runtime errors, dependency issues
uv run examples/precept_autogen_mcp_full.py --domain coding --train 6 --test 4

# DevOps - CloudFormation, IAM, K8s failures
uv run examples/precept_autogen_mcp_full.py --domain devops --train 6 --test 4

# Finance - trading, compliance, market volatility
uv run examples/precept_autogen_mcp_full.py --domain finance --train 6 --test 4

# Booking - reservations, overbooking, payments
uv run examples/precept_autogen_mcp_full.py --domain booking --train 6 --test 4

# Integration - OAuth, APIs, webhooks
uv run examples/precept_autogen_mcp_full.py --domain integration --train 6 --test 4

# List all available domains
uv run examples/precept_autogen_mcp_full.py --list
```

### Concurrent Training ("Tesla Fleet" Mode)

```bash
# Enable concurrent training for ~4x speedup
uv run examples/precept_autogen_mcp_full.py -d logistics -ct -tw 4 --seed 42

# Full concurrent mode (both training and testing)
uv run examples/precept_autogen_mcp_full.py -d logistics -ct -tw 4 -c -w 4 --seed 42
```

### Verbose Mode with Logging

```bash
# Enable DEBUG logging
uv run examples/precept_autogen_mcp_full.py --verbose

# Log to file (JSON format)
uv run examples/precept_autogen_mcp_full.py --verbose --log-file experiment.log
```

### Clean Data and Rerun

```bash
# Remove all learned data (both dynamic and static knowledge stores)
rm -rf data/chroma_precept data/chroma_static_knowledge data/precept_*.json data/experiment_results_*.json

# Run fresh with seed for reproducibility (with static knowledge)
uv run examples/precept_autogen_mcp_full.py --static-knowledge --seed 42

# Run fresh without static knowledge (pure dynamic learning)
uv run examples/precept_autogen_mcp_full.py --no-static-knowledge --seed 42
```

---

## Publication Experiments

PRECEPT includes comprehensive experiment scripts for generating publication-quality results. See `scripts/README_EXPERIMENTS.md` for detailed documentation.

### Available Experiments

| Exp | Name | Purpose | Script |
|-----|------|---------|--------|
| **1** | Main Comparison | PRECEPT vs ExpeL vs Reflexion across 6 domains | `run_exp1_main_comparison.py` |
| **3** | Training Size Ablation | Sample efficiency analysis | `run_exp3_training_size_ablation.py` |
| **4** | Continuous Learning | Cross-episode memory & online adaptation | `run_exp4_continuous_learning.py` |
| **6** | Compositional Generalization | Atomic to composite generalization | `run_exp6_compositional_generalization.py` |
| **7** | Rule Drift | Non-stationary environment adaptation | `run_exp7_rule_drift.py` |

### Running Publication Experiments

```bash
# Experiment 1: Main comparison (all domains, 10 seeds)
uv run scripts/run_exp1_main_comparison.py --publication

# Experiment 6: Compositional generalization
uv run scripts/run_exp6_compositional_generalization.py --publication

# Experiment 7: Rule drift (logistics domain)
uv run scripts/run_exp7_rule_drift.py --publication --domains logistics

# Quick validation run (fewer seeds)
uv run scripts/run_exp1_main_comparison.py --very-quick
```

### Statistical Standards

All publication experiments meet top-tier journal requirements:

| Requirement | Value |
|-------------|-------|
| Independent Runs (N) | 10 |
| Confidence Intervals | 95% (t-distribution) |
| Significance Tests | Paired t-test |
| Effect Size | Cohen's d |

---

## Configuration

### Configuration Files

Configuration is now modular in `src/precept/config/`:

| File | Description |
|------|-------------|
| `agent.py` | AgentConfig - PRECEPT agent settings |
| `baseline.py` | BaselineConfig - Baseline agent settings |
| `llm.py` | LLMConfig - Model, temperature, tokens |
| `constraints.py` | ConstraintConfig - Deterministic pruning |
| `prompts.py` | PromptTemplates - Centralized LLM prompts |
| `logging.py` | LogConfig - Logging setup and formatters |
| `paths.py` | DataPaths - File and directory paths |
| `unified.py` | PreceptConfig - Aggregates all configs |

### Agent Configuration

```python
from precept.config import AgentConfig

config = AgentConfig(
    max_attempts=4,               # Max retry attempts per task
    consolidation_interval=3,     # Trigger consolidation every N tasks
    compass_evolution_interval=2, # Trigger COMPASS evolution every N tasks
    failure_threshold=2,          # Emergency evolution after N consecutive failures
    enable_llm_reasoning=True,    # Enable LLM reasoning (Tier 2)
    force_llm_reasoning=False,    # Force LLM on every attempt
    verbose_llm=False,            # Verbose LLM logging
    max_internal_workers=3,       # Internal concurrency per agent
    max_pivots=3,                 # Max pivots after initial failure
    max_memories=100,             # Max memories to store
    enable_compass_optimization=True,
)
```

### Baseline Configuration

```python
from precept.config import BaselineConfig

config = BaselineConfig(
    model="gpt-4o-mini",          # LLM model
    max_attempts=4,               # Max retry attempts
    temperature=0.7,              # LLM temperature
    max_tokens=200,               # Max tokens for response
    reflection_max_tokens=300,    # Tokens for reflection
    full_reflexion_max_tokens=400,# Tokens for full reflexion
    max_reflections_per_type=20,  # Max cross-episode memories
    verbose=False,                # Verbose logging
    max_internal_workers=3,       # Internal concurrency
)
```

### Production Settings

```python
from precept.config import PreceptConfig

config = PreceptConfig()
config.agent.consolidation_interval = 50    # Every 50 tasks
config.agent.compass_evolution_interval = 100  # Every 100 tasks
config.agent.failure_threshold = 3          # After 3 consecutive failures
config.agent.max_memories = 1000            # Store more memories
```

---

## Troubleshooting

### "Connection closed" Error
```
mcp.shared.exceptions.McpError: Connection closed
```
**Solution**: The MCP server process terminated unexpectedly. Check for Python errors in the server script.

### "Module not found" Error
```
ModuleNotFoundError: No module named 'precept'
```
**Solution**: Install the project with uv:
```bash
uv pip install -e . --system
```

### "OpenAI API key not found"
```
❌ OpenAI API not available. Set OPENAI_API_KEY in .env
```
**Solution**: Create `.env` file with your API key.

### "NOT_GIVEN import error"
```
ImportError: cannot import name 'NOT_GIVEN' from 'openai'
```
**Solution**: Upgrade openai package:
```bash
uv pip install 'openai>=1.50.0' --system --upgrade
```

### Missing dependencies
```
ModuleNotFoundError: No module named 'tiktoken'
```
**Solution**: Install all dependencies:
```bash
uv pip install -e . --system
```

### MCP Disconnect Errors (asyncio shutdown)
```
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```
**Solution**: These are harmless cleanup errors from the MCP library during asyncio shutdown. They are automatically suppressed in the experiment runner. If you see them, they can be safely ignored.

### JSONRPC Parsing Errors (MCP Server)
```
pydantic_core._pydantic_core.ValidationError: Invalid JSON: trailing characters
```
**Solution**: This occurs when the MCP server writes to stdout instead of stderr. Ensure all logging in `precept_mcp_server.py` goes to `sys.stderr`. The framework now handles this automatically with lazy logger initialization.

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- **AutoGen**: Microsoft Research - Multi-agent orchestration framework
- **MCP**: Anthropic - Model Context Protocol for tool integration
- **GEPA**: "Reflective Prompt Evolution Can Outperform Reinforcement Learning" paper (COMPASS builds upon and extends this work)
- **Reflexion**: Shinn et al., 2023 - "Reflexion: Language Agents with Verbal Reinforcement Learning"
- **ExpeL**: Zhao et al., 2024 - "ExpeL: LLM Agents Are Experiential Learners"
- **Context Engineering**: Google Whitepaper 2024 - Efficiency patterns
- **ChromaDB**: Vector database for semantic embeddings
- **uv**: Astral - Fast Python package manager
- **Rich**: Terminal formatting and progress bars
- **Pydantic**: Data validation and structured outputs

---

## Support

For issues and feature requests, please open a GitHub issue.

---

<p align="center">
  <b>PRECEPT: Planning Resilience via Experience, Context Engineering & Probing Trajectories</b><br>
  <i>Learn once, benefit forever. Baselines repeat the same mistakes.</i>
</p>
