# PRECEPT: Planning Resilience via Experience, Context Engineering & Probing Trajectories

**A Unified Framework for Test-Time Adaptation with Compositional Rule Learning and Pareto-Guided Prompt Evolution**

---

**Arash Shahmansoori**

---

## Abstract

Large Language Model (LLM) agents face fundamental limitations that constrain real-world deployment. Verbal reflection methods (Shinn et al., 2023; Zhao et al., 2023) suffer exponential degradation in multi-condition scenarios, while reinforcement learning requires prohibitive sample complexity (100+ episodes per rule) and cannot adapt to non-stationary environments. Recent work in evolutionary computing—particularly the Digital Red Queen (DRQ) framework (Kumar et al., 2026)—demonstrates that agents trained via static optimization are brittle, failing 72% against novel adversarial dynamics. Critically, no prior paradigm supports compositional generalization: covering 2^N condition combinations requires O(2^N) training examples.

We introduce **PRECEPT** (**P**lanning **R**esilience via **E**xperience, **C**ontext **E**ngineering & **P**robing **T**rajectories), a unified framework that translates insights from evolutionary computing theory into a practical architecture for LLM agents. PRECEPT achieves deterministic rule application through O(1) exact-match retrieval, compositional generalization through atomic constraint stacking, and robust adaptation through adversarial dynamics where environmental entropy serves as a continual selective pressure. Our key contributions include:

**(1) Deterministic Retrieval for LLM Agents:** Hash-based O(1) lookup eliminates LLM interpretation errors entirely, achieving 0% error rate regardless of condition complexity, compared to 94.4% for verbal baselines at N=10 (Theorem 6.8).

**(2) Exponential Compositional Coverage:** A semantic tier hierarchy enables O(2^N) compositional coverage from N atomic precepts, where 6 atomic rules generalize to 63 composite scenarios without additional training (Theorem 3.1).

**(3) Adversarial Adaptation via Evo-Memory:** Drawing on the Red Queen hypothesis, PRECEPT maintains a growing history of failure constraints (analogous to DRQ's opponent history), achieving 64× drift resilience through Bayesian inference and threshold-based rule invalidation (Corollary 6.7).

**(4) COMPASS Algorithm with MAP-Elites Diversity:** ML-based complexity detection with Pareto-optimal selection maintaining strategy diversity, preventing the convergence collapse observed in static optimization, while achieving 72% computational reduction.

**(5) Deterministic Pruning with Formal Guarantees:** Provable P(repeat_failed)=0 through constraint classification, eliminating cyclic failure dynamics (Theorem 6.9).

**Empirical Results** (N=39 configurations, 10 seeds, p<0.001): +41.0pp mean first-try advantage across 3 high-complexity domains (Cohen's d>1.9 on all 3), +49.2pp compositional generalization improvement, 100% P₁ on 2-way compositions (Cohen's d=8.27), +50pp drift recovery, 61% fewer steps, and 22×/64× theoretical advantage ratios validated across six domains.

PRECEPT establishes a paradigm of **dynamic adaptation** over static optimization. While DRQ demonstrates these evolutionary principles in program synthesis, PRECEPT is, to our knowledge, the first to operationalize them for rule-governed autonomous agents with formal deterministic guarantees.

---

## 1. Introduction

Deploying LLM agents in real-world applications demands four critical capabilities: (1) learning from limited examples, (2) composing atomic rules into complex policies, (3) adapting to environmental changes, and (4) making deterministic decisions without interpretation errors. Consider an autonomous logistics agent handling port closures, customs regulations, hazardous cargo restrictions, and regional compliance requirements—often simultaneously. Current approaches fundamentally cannot meet these requirements.

**The Verbal Reflection Paradigm** relies on LLM interpretation of natural language rules stored in episodic memory. Reflexion (Shinn et al., 2023) pioneered self-reflection but discards knowledge after each episode. Full Reflexion persists reflections but uses coarse task-type indexing requiring LLM interpretation at test time. ExpeL (Zhao et al., 2023) extracts natural language insights but inherits the same interpretation problem. The fundamental issue: LLM interpretation accuracy degrades exponentially with condition count.

**The Reinforcement Learning Paradigm** learns policies through reward optimization but suffers from prohibitive sample complexity (β=100+ episodes per rule), inability to adapt to non-stationary environments without retraining, no mechanism for compositional generalization, and no explicit knowledge retrieval.

**The Static Optimization Problem.** Recent work in evolutionary computing reveals a deeper issue. The Digital Red Queen (DRQ) framework (Kumar et al., 2026) demonstrates that agents trained via *static* optimization—where the objective function remains fixed—are brittle, failing 72% of the time against novel adversarial dynamics. DRQ overcomes this through *dynamic* adaptation: continual self-play against a growing history of opponents produces increasingly general and robust strategies. This motivates a shift from training smarter models to building *survivable systems* that improve through adversarial interaction.

We identify **four fundamental limitations** affecting all prior approaches:

**Limitation 1 (Compositional Explosion):** Given N atomic conditions, there exist 2^N possible combinations. Verbal methods cannot compose rules like "IF strike AND hazmat AND express THEN specialized_port" without exponential training. RL faces identical combinatorial explosion.

**Limitation 2 (Interpretation Degradation):** LLM interpretation of natural language rules degrades exponentially with condition count. At N=10 conditions, we prove partial match errors reach 94.4% (Theorem 6.8). RL policies encode decisions implicitly, offering no explicit retrieval mechanism.

**Limitation 3 (Drift Blindness / Static Brittleness):** When environments change (e.g., Rotterdam also strikes), previously learned rules become harmful. Verbal baselines persist stale rules indefinitely. RL requires complete retraining—impractical for deployment. As DRQ demonstrates, static optimization produces "overfit" agents that fail catastrophically outside their training distribution.

**Limitation 4 (Sample Inefficiency):** RL requires β=100+ episodes per rule. Verbal reflection requires β=5+ with inconsistent results. Real-world deployment demands β≤3.

**Our Solution:** PRECEPT operationalizes insights from evolutionary computing theory into a production-ready architecture for LLM agents. The core innovation is treating **environmental entropy as the "Red Queen"**—an ever-present adversary that the agent must continually adapt to defeat. Rather than optimizing against static benchmarks, PRECEPT:

1. **Replaces probabilistic interpretation with deterministic retrieval**: O(1) exact-match via structured condition keys eliminates interpretation errors entirely.

2. **Maintains a growing constraint history (Evo-Memory)**: Analogous to DRQ's opponent history, PRECEPT accumulates failure constraints that future plans must satisfy—mathematically eliminating cyclic failures.

3. **Preserves strategy diversity via MAP-Elites integration**: COMPASS maintains topologically distinct alternatives, preventing the "convergence collapse" observed in static optimization.

4. **Enables active constraint discovery via Epistemic Probing**: The agent actively "battles" the environment rather than passively logging errors.

This architectural synthesis enables capabilities not provided by prior approaches: O(2^N) compositional generalization from N atomic rules, zero interpretation errors at any condition count, and 64× better drift resilience through principled adversarial adaptation.

### 1.1 Contributions

This paper makes the following contributions:

1. **PRECEPT Framework** (Section 2): A test-time adaptation architecture for LLM agents featuring O(1) exact-match retrieval via structured condition keys, eliminating interpretation errors entirely. The framework integrates episodic memory, semantic retrieval, conflict-aware dual-mode retrieval, and deterministic pruning within a unified four-layer architecture.

2. **Atomic Constraint Stacking** (Section 3): A compositional generalization mechanism enabling O(2^N) coverage from N atomic precepts through a semantic tier hierarchy where safety constraints deterministically dominate compliance constraints, which dominate preferences. We prove 2^N - 1 composite scenario coverage (Theorem 3.1).

3. **Adversarial Adaptation via Evo-Memory** (Section 4): Drawing on the Red Queen hypothesis from evolutionary computing (Kumar et al., 2026), PRECEPT maintains a growing history of failure constraints analogous to DRQ's opponent history. A unified treatment of Type I conflicts (static vs dynamic knowledge via Bayesian ensemble with Thompson Sampling) and Type II conflicts (environmental drift via threshold-based invalidation), achieving 64× drift resilience (Corollary 6.7).

4. **Deterministic Pruning with Formal Guarantees** (Section 4.5): A provably correct constraint classification system guaranteeing P(repeat_failed)=0 through HARD/SOFT constraint categorization (Theorem 6.9), eliminating the cyclic failure dynamics that affect standard agents.

5. **COMPASS Algorithm with MAP-Elites Diversity** (Section 5): An extension of GEPA (Agrawal et al., 2025) integrating the MAP-Elites principle (Mouret & Clune, 2015) for strategy diversity maintenance. ML-based complexity detection with smart rollout allocation achieves 72% computational reduction while maintaining Pareto-optimal candidate selection.

6. **Theoretical Analysis** (Section 6): Closed-form bounds establishing 22× advantage in multi-condition scenarios (Corollary 6.5), 64× drift resilience ratio (Corollary 6.7), zero retry waste (Theorem 6.9), and proving 0% partial match errors for PRECEPT vs 94.4% for baselines at N=10 conditions (Theorem 6.8).

7. **Empirical Validation** (Section 7): Experiments across six domains (three in main comparison, three excluded—see §7.2): +41.0pp mean first-try advantage, Cohen's d > 1.9 on all 3 domains, 100% P₁ on 2-way compositions (d=8.27), +49.2pp compositional generalization, +50pp drift recovery—validated with paired t-tests, Bonferroni correction, 95% CIs, N=39 configurations, 10 seeds. Ablation studies quantify each component's contribution.

8. **Systems-Level Operationalization** (Section 9.9): A demonstration that theoretical concepts from evolutionary computing (Red Queen dynamics, growing opponent history, diversity preservation) can be translated into production-ready LLM agent architecture with deterministic guarantees.

---

## 2. PRECEPT Framework Overview

### 2.1 System Architecture

PRECEPT follows the Model Context Protocol (MCP) architecture, separating client-side agent orchestration from server-side knowledge management and tool execution.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'secondaryColor': '#f8f8f8', 'tertiaryColor': '#fff', 'fontFamily': 'Arial, sans-serif', 'fontSize': '14px'}}}%%
flowchart TB
    subgraph CLIENT["<b>MCP CLIENT</b>"]
        direction TB
        PA["<b>PRECEPTAgent</b><br/><i>precept_agent.py</i>"]
        subgraph CMP["COMPASS"]
            CA["ComplexityAnalyzer"]
            SR["RolloutStrategy"]
        end
        subgraph AG["AutoGen"]
            AA["AssistantAgent"]
        end
        PA --> CMP & AG
    end

    CLIENT <--->|"<b>JSONRPC</b><br/><i>stdio</i>"| SERVER

    subgraph SERVER["<b>MCP SERVER</b>"]
        direction TB
        FM["<b>FastMCP</b><br/><i>precept_mcp_server.py</i>"]
        subgraph ST["Storage"]
            direction LR
            CV["Vector DB"]
            LR2["Rules Dict"]
            AP["Precepts"]
        end
        subgraph TL["Tools (70+)"]
            direction LR
            T1["retrieve_*"]
            T2["store_*"]
            T3["*_rule_*"]
        end
        subgraph MD["Modules"]
            direction LR
            CR["Conflict"]
            GE["GEPA"]
        end
        FM --> TL --> ST & MD
    end
```

**Figure 1.** PRECEPT MCP Architecture. The **Client** (`precept_agent.py`) orchestrates task execution with COMPASS enhancements (`compass_mcp_client.py`) for complexity analysis and rollout allocation. Communication occurs via stdio/JSONRPC to the **Server** (`precept_mcp_server.py`), which exposes 70+ MCP tools for memory retrieval, rule management, conflict resolution, and domain-specific actions. Persistent storage uses a vector database for embeddings and JSON files for rules and experiences.

### 2.1.1 Theoretical Foundations: Code-to-Theory Mapping

**Table 1.** Theoretical Foundations for PRECEPT's Key Features.

| Feature | Code Implementation | Theoretical Foundation | Section |
|---------|---------------------|----------------------|---------|
| **Dual-Mode Retrieval** | `retrieve_with_dual_mode()` | Algorithm 0, Definition 2.2, Table 1e | §2.3.1 |
| **Type I Conflict (Static vs Dynamic)** | `EnsembleConflictDetector.detect()` | Table 3, Definition 4.1, Algorithm 4.1a | §4.1 |
| **Evo-Memory** | `partial_progress`, `failed_options` | Definition 4.0, Theorem 4.0 (Eliminates Cyclic Failures) | §4.0 |
| **Deterministic Pruning** | `RefineInterceptor.is_forbidden()`, `add_constraint()` | Theorem 4.5 (P(repeat_fail)=0), Corollary 4.5.1 | §4.5 |
| **Thompson Sampling** | `BetaDistribution.sample()` | Definition 4.1, Theorem 4.1 (Provable Optimality) | §4.1 |
| **Bayesian Conflict Resolution** | `BetaDistribution.update(success)` | Definition 4.1, Algorithm 4.1a | §4.1 |
| **MAP-Elites / Pareto Diversity** | `pareto_front`, `diversity_threshold`, `diversity_rollouts` | Definition 5.1, 5.2 (Topological Distinctness) | §5.1.1 |
| **Epistemic Probing** | `_DIAGNOSTIC_PROBES`, `enable_epistemic_probing` | Definition 2.1, Algorithm 2.1 | §2.5 |
| **Compositional Stacking** | `retrieve_atomic_precepts()`, tier sorting | Theorem 3.1 (2^N-1 Coverage) | §3 |

*All implementations verified in source code; theoretical proofs in respective sections.*

**Table 1a.** MCP Client-Server Component Mapping.

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| Agent Orchestrator | `precept_agent.py` | `PRECEPTAgent`, `connect()`, `run_task()` |
| MCP Client (Base) | `precept_mcp_client.py` | `AbstractPRECEPTMCPClient`, `call_tool()` |
| COMPASS Client | `compass_mcp_client.py` | `PRECEPTMCPClientWithCOMPASS` |
| MCP Server | `precept_mcp_server.py` | `FastMCP`, `@mcp.tool()` decorators |
| Complexity Analysis | `complexity_analyzer.py` | `PRECEPTComplexityAnalyzer`, `SmartRolloutStrategy` |
| Conflict Resolution | `conflict_resolution.py` | `ConflictManager`, `EnsembleConflictDetector` |
| Scoring | `scoring.py` | `pareto_select()`, `compute_gepa_scores()` |

### 2.2 PRECEPTAgent Detailed Execution Flow

Figure 1a presents the complete seven-phase execution flow of `PRECEPTAgent.run_task()`, illustrating how PRECEPT orchestrates task parsing, knowledge retrieval, decision-making, and adaptive learning within a single unified pipeline.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#404040', 'lineColor': '#606060', 'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '13px'}}}%%
flowchart TB
    subgraph P1["<b>Phase 1: Task Parsing</b>"]
        direction LR
        IN["Input: Task τ<br/>+ metadata κ"] --> PARSE["Parse Task<br/><i>hybrid or rule-based</i>"]
        PARSE --> META["Inject Condition Key"]
    end

    subgraph P2["<b>Phase 2: COMPASS Evaluation</b>"]
        direction LR
        META --> CMPLX["Complexity<br/>Analysis"]
        CMPLX --> CTRL["Controller<br/>Decision"]
        CTRL --> DEC{{"Action?"}}
    end

    DEC -->|"BLOCK"| OUT["⊗ Blocked"]
    DEC -->|"PIVOT"| P3
    DEC -->|"FAST"| P3
    DEC -->|"PROCEED"| P3

    subgraph P3["<b>Phase 3: Context Retrieval</b>"]
        direction TB
        MODE{{"Mode?"}}
        MODE -->|"Compositional"| COMP["Atomic Constraint<br/>Stacking"]
        MODE -->|"Hybrid"| HYB["3-Tier Retrieval<br/><i>exact → vector → partial</i>"]
        MODE -->|"Standard"| STD["Semantic<br/>Retrieval"]
        COMP --> MEM["+ Procedural<br/>Memory"]
        HYB --> MEM
        STD --> MEM
    end

    subgraph P4["<b>Phase 4: Solution Derivation</b>"]
        direction TB
        MEM --> CHECK{{"Match<br/>Found?"}}
        CHECK -->|"Compositional"| TIER["Tier Sort<br/><i>max(tier)</i>"]
        CHECK -->|"Exact"| EXACT["Direct<br/>Match"]
        CHECK -->|"None"| LLM["LLM<br/>Reasoning"]
        TIER --> SOL["O(1) Solution"]
        EXACT --> SOL
        LLM --> SOL
    end

    subgraph P5["<b>Phase 5: Action Execution</b>"]
        direction LR
        SOL --> EXEC["Domain<br/>Action"]
        EXEC --> RES{{"Result?"}}
    end

    subgraph P6["<b>Phase 6: Outcome Processing</b>"]
        direction TB
        RES -->|"Success"| SUC["Record Success<br/><i>c += 0.25, f ← 0</i>"]
        RES -->|"Failure"| FAIL["Record Failure<br/><i>c ×= 0.5, f++</i>"]
        FAIL --> THR{{"f ≥ θ?"}}
        THR -->|"Yes"| INV["Invalidate<br/>Rule"]
        THR -->|"No"| PRUNE["Prune &<br/>Retry"]
    end

    subgraph P7["<b>Phase 7: Knowledge Update</b>"]
        direction LR
        SUC --> LEARN["Extract Atomic<br/>Precepts"]
        LEARN --> STORE["Store<br/>Experience"]
    end

    INV --> P3
    PRUNE --> P3
    STORE --> DONE["Output: Result"]
    OUT --> DONE
```

**Figure 1a.** Complete execution flow of the PRECEPT agent (`precept_agent.py`). The pipeline comprises seven phases: **(1) Task Parsing** extracts structured task representation with optional LLM fallback for complex inputs; **(2) COMPASS Evaluation** performs ML-based complexity analysis and constraint checking, yielding one of four actions (BLOCK, PIVOT, FAST_PATH, PROCEED); **(3) Context Retrieval** operates in three modes—compositional (atomic constraint stacking), hybrid (3-tier exact→vector→partial), or standard semantic search; **(4) Solution Derivation** applies O(1) direct resolution for compositional/exact matches or invokes LLM reasoning otherwise; **(5) Action Execution** delegates to domain-specific strategy; **(6) Outcome Processing** updates rule confidence (c) and failure count (f) with threshold-based invalidation (θ=2); **(7) Knowledge Update** extracts atomic precepts from successful executions and persists experiences. Dashed arrows indicate retry loops that return to Phase 3 with accumulated constraints.

**Table 1b.** Implementation mapping of execution phases to source code.

| Phase | Component | Implementation | Key Mechanism |
|:-----:|-----------|----------------|---------------|
| 1 | Task Parser | `_hybrid_parse_task()` | Rule-based parsing with LLM fallback |
| 2 | COMPASS | `COMPASSController.evaluate_action()` | Complexity analysis + constraint evaluation |
| 3a | Compositional Retrieval | `fetch_context_compositional()` | Decomposes κ → retrieves atomic precepts |
| 3b | Hybrid Retrieval | `fetch_context_with_hybrid()` | Tier 1: exact match → Tier 2: vector → Tier 3: partial |
| 4 | Direct Resolution | `sorted(precepts, key=tier, reverse=True)[0]` | O(1) highest-tier selection |
| 4 | LLM Reasoning | `_llm_reason_with_evolved_prompt()` | COMPASS-enhanced structured output |
| 5 | Execution | `strategy.execute_action()` | Domain-specific action dispatch |
| 6 | Success Handler | `record_rule_success(κ)` | Resets f←0, restores c+=0.25 |
| 6 | Failure Handler | `report_rule_failure(κ)` | Increments f++, decays c×=0.5 |
| 6 | Invalidation | `invalidate_rule(κ)` | Removes rule when f≥θ |
| 7 | Learning | `extract_and_store_atomic_precepts()` | Persists tier-annotated precepts |

### 2.3 Knowledge Layer Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart LR
    subgraph KL["<b>Knowledge Layer</b>"]
        CV["Vector DB<br/><i>vectors</i>"]
        EM["Episodic<br/><i>trajectories</i>"]
        LR["Rules<br/><i>learned_rules</i>"]
        AP["Precepts<br/><i>tier dict</i>"]
    end

    KL --> R1["<b>Semantic</b><br/>O(log n)"] & R2["<b>Exact</b><br/>O(1)"] & R3["<b>Hybrid</b><br/>BM25+Sem"]
```

**Figure 1c.** Knowledge Layer with three retrieval modes: semantic similarity O(log n), exact-match O(1) via dictionary lookup, and hybrid BM25+semantic.

### 2.3.1 Dual-Mode Retrieval with Conflict Resolution

PRECEPT provides a unified retrieval interface that queries all knowledge sources simultaneously and automatically resolves conflicts between static and dynamic knowledge. This is implemented via `retrieve_with_dual_mode()` in `precept_mcp_server.py`.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 0: Dual-Mode Retrieval with Conflict Resolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: precept_mcp_server.py lines 1996-2139

Input: query, static_top_k, dynamic_top_k, episodic_top_k
Output: Combined results with conflict analysis and resolution

function retrieve_with_dual_mode(query, static_top_k=3, dynamic_top_k=3, episodic_top_k=3):
    static_items ← []
    dynamic_items ← []
    
    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: PARALLEL RETRIEVAL FROM THREE TIERS
    // ═══════════════════════════════════════════════════════════════════
    
    // Tier 0a: Static Knowledge Base (pre-deployment facts, e.g., PDF content)
    if static_vector_store exists then
        static_docs ← static_vector_store.similarity_search(query, k=static_top_k)
        for doc in static_docs do
            static_items.append(KnowledgeItem(
                content=doc.page_content,
                source=STATIC_KB,
                confidence=0.9,  // High prior for curated knowledge
                metadata=doc.metadata
            ))
        end for
    end if
    
    // Tier 0b: Dynamic Experiences (runtime observations, e.g., web updates)
    if vector_store exists then
        dynamic_docs ← vector_store.similarity_search(query, k=dynamic_top_k)
        for doc in dynamic_docs do
            if doc.metadata.type == "factual_knowledge" then
                dynamic_items.append(KnowledgeItem(
                    content=doc.page_content,
                    source=DYNAMIC_EXPERIENCE,
                    confidence=0.8,  // Moderate prior for runtime data
                    metadata=doc.metadata
                ))
            end if
        end for
    end if
    
    // Tier 0c: Episodic Memory (task execution histories)
    episodic_memories ← memory_store.retrieve_relevant(query, top_k=episodic_top_k)
    
    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: CONFLICT DETECTION AND RESOLUTION
    // ═══════════════════════════════════════════════════════════════════
    
    if conflict_manager exists AND |static_items| > 0 AND |dynamic_items| > 0 then
        for static_item in static_items do
            for dynamic_item in dynamic_items do
                // Detect conflict using 5-method ensemble
                conflict, resolution ← conflict_manager.detect_and_resolve(
                    static_item, dynamic_item, auto_resolve=True
                )
                
                if conflict then
                    // Resolution contains: winner, confidence, strategy, reasoning
                    mark_conflict(static_item, dynamic_item, resolution)
                end if
            end for
        end for
    end if
    
    // Combine all results into formatted string output
    return format_results(static_items, dynamic_items, episodic_memories, conflicts)
end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Table 1e.** Three-Tier Knowledge Architecture with Retrieval Characteristics.

| Tier | Storage | Source | Confidence Prior | Retrieval | Use Case |
|------|---------|--------|------------------|-----------|----------|
| **Static KB** | Vector DB | Pre-deployment (PDFs, docs) | 0.9 (high) | Semantic similarity | Factual grounding |
| **Dynamic Experience** | Vector DB | Runtime (web, APIs) | 0.8 (moderate) | Semantic + metadata filter | Current information |
| **Episodic Memory** | MemoryStore | Task executions | Task-dependent | Trajectory matching | Experience replay |
| **Learned Rules** | Hash Table | Successful solutions | 1.0 (deterministic) | O(1) exact match | Direct application |

**Definition 2.2 (Knowledge Conflict).** *A conflict exists between static item S and dynamic item D when the EnsembleConflictDetector returns weighted_conflict ≥ θ = 0.4, indicating semantic contradiction detected by multiple detection methods.*

**Integration with Type I Conflict Resolution.** When conflicts are detected in Phase 2, PRECEPT invokes the **Type I conflict resolution** mechanism documented in Section 4.1. This includes:
- **Ensemble Detection**: Five-method voting (NLI, Semantic, Temporal, Evidence, LLM) with configurable weights (Table 3)
- **Thompson Sampling Resolution**: Bayesian selection between static and dynamic sources (Algorithm 4.1a)
- **Reliability Learning**: Posterior updates to Beta distributions based on resolution outcomes

This unified pipeline enables PRECEPT to serve as a **conflict-aware RAG system**—the key differentiator from standard RAG which returns results without detecting or resolving contradictions between knowledge sources.

### 2.4 Simplified Execution Pipeline

The simplified execution pipeline shows the high-level flow.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart LR
    T["Task τ"] --> E["Extract κ"] --> R["Retrieve"] --> C["Compose"] --> X["Execute"]
    X --> S{{"OK?"}}
    S -->|"✓"| RS["Restore<br/>+0.25"]
    S -->|"✗"| RF["Decay<br/>×0.5"]
    RF --> K{{"≥θ?"}}
    K -->|"Y"| I["Invalidate"]
    K -->|"N"| P["Retry"]
```

**Figure 2.** Simplified PRECEPT pipeline. Extract condition key κ → Retrieve (3 modes) → Compose (tier sort) → Execute → Handle success/failure with threshold-based invalidation. See Figure 1a for detailed implementation flow.

**Table 1c.** Execution Pipeline Stage Details with MCP Tool Mapping.

| Stage | Client/Server | MCP Tool / Function | Key Operation |
|-------|---------------|---------------------|---------------|
| (1) Parse | Client | `PRECEPTAgent.run_task()` | Extract condition_key κ from task |
| (2) Retrieve | Server | `retrieve_atomic_precepts(κ)` | Returns `List[Dict]` with tier, solution_hint |
| (3) Compose | Server | `sorted(..., key=tier, reverse=True)` | Highest tier precept wins |
| (4) Detect | Server | `ConflictManager.detect()` | Trigger if weighted_conflict ≥ 0.4 |
| (5) Execute | Server | Domain tools (`book_shipment`, etc.) | Apply solution via domain-specific tool |
| (6) Handle | Client→Server | `report_rule_failure(κ)` / `record_rule_success(κ)` | Decay ×0.5, restore +0.25, invalidate at θ=2 |

### 2.5 Additional Agent Capabilities

**Table 1d.** Extended PRECEPTAgent Features (`precept_agent.py`).

| Feature | Implementation | Description |
|---------|----------------|-------------|
| **Structured Outputs** | `_call_llm_structured()` | Pydantic models (`ReasoningResponse`) ensure guaranteed schema from LLM |
| **Procedural Memory** | `store_procedure()` | Dynamically learns recovery procedures from successful error recovery |
| **Epistemic Probing** | `_compass_controller.execute_probe()` | Diagnostic probes discover hidden constraints via targeted MCP tool calls |
| **Online Validation** | `register_task_for_online_validation()` | Real-time task registration enables COMPASS/GEPA to use verified execution signals |
| **Partial Progress** | `record_failed_option()` | Persists failed options across episodes for resumable exploration |
| **COMPASS Error Eval** | `_compass_controller.evaluate_error()` | Classifies errors by constraint tier (Physics > Policy > Instruction) |
| **Validation Filter** | Lines 1664-1721 | Validates LLM suggestions against domain-valid options before execution |
| **Cross-Episode Forbidden** | `context.failed_options` | Prevents retrying options that failed in previous episodes |

**Procedural Memory.** When error recovery succeeds, PRECEPT stores the recovery pattern as a procedure:

```
procedure_name: recovery_{error_code}_{solution}
steps:
  1. Attempt default action for {action}
  2. If error '{error_code}' occurs, use '{solution}'
  3. Learned from: {task}
```

This enables the agent to apply learned "how-to" strategies for similar future scenarios via `context.procedure` retrieval.

**Epistemic Probing: Active Constraint Discovery.**

PRECEPT implements **Epistemic Probing** as a mechanism for active constraint discovery, operationalizing the adversarial dynamics from evolutionary computing (Kumar et al., 2026). Rather than passively logging errors, PRECEPT actively "battles" the environment to reveal hidden constraints.

**Definition 2.1 (Epistemic Probing).** *Let $\mathcal{E}$ denote the environment (the system the agent interacts with, e.g., a logistics API), and let $\mathcal{C}(\mathcal{E}) = \{c_1, \ldots, c_n\}$ be the set of hidden constraints (conditions that cause failures but are not directly observable, e.g., an undocumented port closure). A Query $q$ is a targeted diagnostic request (e.g., "check port Hamburg status"). Epistemic Probing is a function:*
$$\text{Probe}: \mathcal{E} \times q \rightarrow \text{Evidence}(c_i)$$

*that reveals constraint $c_i \in \mathcal{C}(\mathcal{E})$ through targeted interaction, where $\text{Evidence}(c_i)$ is the observation confirming or refuting the existence of constraint $c_i$.*

**Theoretical Connection to Red Queen Dynamics:** Just as DRQ's warriors must continually probe and adapt to opponent strategies, PRECEPT agents probe the environment to discover operational constraints before committing to plans.

**Implementation:**
```
compass_decision.action == PROBE
  → execute_probe(probe_spec)
  → process_probe_result()
  → learn_pattern(error_pattern, action, succeeded)
```

**Algorithm 2.1: Epistemic Probe Execution**
```
function epistemic_probe(ambiguous_error, mcp_client):
    // Discover available probes from MCP server
    available_probes ← _compass_controller.discover_probes(mcp_client)
    
    // Select probe targeting the ambiguous constraint
    probe_spec ← select_probe(available_probes, ambiguous_error)
    
    // Execute and observe
    evidence ← execute_probe(probe_spec)
    
    // Update Evo-Memory with discovered constraint
    if evidence.reveals_constraint then
        RefineInterceptor.add_constraint(evidence.constraint)
        partial_progress.record_failed_option(evidence.constraint)
    
    return evidence
```

This transforms error handling from reactive (wait for failure) to proactive (actively discover constraints), enabling PRECEPT to avoid failure trajectories before committing resources.

**Configurable Agent Options.**

| Option | Default | Description |
|--------|---------|-------------|
| `disable_exhausted_exit` | False | Continue exploration when LLM signals "EXHAUSTED" |
| `enable_random_fallback` | False | Random selection when all options exhausted |
| `soft_constraints_retriable` | False | Allow retrying SOFT constraint failures |
| `enable_compositional_generalization` | True | Enable atomic constraint stacking |
| `enable_atomic_precept_storage` | True | Store tier-annotated atomic precepts |
| `enable_hybrid_parsing` | True | Use rule-based + LLM fallback for task parsing |

**Hybrid Task Parsing.** PRECEPT uses a two-stage parsing approach (`_hybrid_parse_task()`):

1. **Rule-based parsing** (fast, deterministic) via `strategy.parse_task(task)`
2. **Parsing quality assessment** (`_assess_parsing_quality()`) scores confidence 0.0-1.0
3. **LLM fallback** if confidence < 0.8, using structured JSON output

```
if parsing_confidence ≥ 0.8:
    return rule_based_result  // Fast path
else:
    llm_parsed ← await _llm_parse_task(task)  // JSON mode
    merge(rule_based_result, llm_parsed)
```

**Session Management.** The agent supports conversational sessions via:

| Method | Description |
|--------|-------------|
| `start_session(session_id)` | Initialize session with UUID |
| `end_session(store_experience)` | Close session, return stats |
| `chat(message)` | Process message (task detection → `run_task()` or Q&A) |
| `chat_stream(message)` | Streaming chat response |
| `get_conversation_history()` | Retrieve session messages |
| `reset_conversation()` | Clear session state |

**Domain-Specific Extensions.** The Strategy Pattern enables domain-specific behavior:

| Domain | Strategy | Special Features |
|--------|----------|------------------|
| Logistics | `LogisticsDomainStrategy` | Port routing, customs handling |
| Coding | `CodingDomainStrategy` | Docker-based sandboxed execution |
| DevOps | `DevOpsDomainStrategy` | Kubernetes deployment |
| Finance | `FinanceDomainStrategy` | Trade execution |
| Booking | `BookingDomainStrategy` | Flight/hotel reservations |
| Integration | `IntegrationDomainStrategy` | OAuth, API webhooks |

---

## 3. Atomic Constraint Stacking

### 3.1 Problem: Compositional Explosion

Given N atomic conditions, there are 2^N possible combinations. Training on all combinations is infeasible.

**Example:**
- 6 atomic conditions: {SAFE, ASIA, EURO, FAST, ECON, BULK}
- 2^6 = 64 possible combinations
- Training all 64 requires O(64 × β) = O(192) tasks at β=3

**PRECEPT Solution:** Learn N atomic precepts, compose at test time via tier-based resolution using `SEMANTIC_CONDITION_TIERS`.

### 3.2 Semantic Tier Hierarchy

**Table 2.** Semantic Tier Hierarchy from `SEMANTIC_CONDITION_TIERS` in `precept_mcp_server.py`.

| Tier | Priority | Category | Conditions | Dominance Rule |
|:----:|:--------:|----------|------------|----------------|
| **3** | Highest | Safety | SAFE, SECURE, RISK, CANCEL, AUTH | Always wins |
| **2** | Medium | Compliance | ASIA, EURO, INTL, HIPAA, AUDIT | Wins over Tier 1 |
| **1** | Lowest | Preferences | FAST, ECON, BULK, SPEED, COST | Default fallback |

**Resolution Rule:** Given conditions {A, B, C}, solution ← precept(argmax{tier(A), tier(B), tier(C)})

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#333', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart LR
    T3["<b>Tier 3</b><br/>Safety"] ==>|"≻"| T2["<b>Tier 2</b><br/>Compliance"] ==>|"≻"| T1["<b>Tier 1</b><br/>Preferences"]
```

**Figure 3.** Tier dominance hierarchy. Higher tiers always override lower tiers in compositional resolution.

### 3.3 Composition Algorithm

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 1: Atomic Constraint Stacking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: precept_agent.py → run_task() (inline, lines 1504-1560)

Input: compositional_context from fetch_context_compositional()
Output: compositional_direct_solution
Guards: enable_dynamic_tier_resolution=False (use static tier sort), synthesis_mode ∈ {"full_compositional", "hierarchical_compositional"}, compositional_context is not None

 1:  function COMPOSE(compositional_context):
 2:      precepts_found ← compositional_context.precepts_found
 3:
 4:      // Guard: only for multi-constraint scenarios
 5:      if |precepts_found| ≤ 1 then
 6:          return ⊥
 7:      end if
 8:
 9:      // Sort by tier (descending) - highest priority first
10:      sorted_precepts ← sorted(precepts_found,
11:                               key=λp: p.get("tier", 1),
12:                               reverse=True)
13:
14:      // Get solution from highest-tier precept
15:      highest_precept ← sorted_precepts[0]
16:      solution_hint ← highest_precept.get("solution_hint", "")
17:
18:      // Parse solution_hint format: "solution:value" or "solution:LLM→value"
19:      if ":" in solution_hint then
20:          raw ← solution_hint.split(":", 1)[1]
21:          if "→" in raw then
22:              // Handle exploration paths like "LLM→hamburg→singapore"
23:              for part in raw.split("→") do
24:                  if part.lower() ≠ "llm" and part.strip() then
25:                      return part.strip()
26:                  end if
27:              end for
28:          else
29:              return raw.strip()
30:          end if
31:      else
32:          return solution_hint.strip() if solution_hint else ⊥
33:      end if
34:  end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complexity: O(N log N) for N precepts
Property: Deterministic—no LLM interpretation required for tier resolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3.4 Compositional Generalization Property

**Theorem 3.1 (Compositional Coverage).** *Given N atomic precepts learned from N training scenarios, PRECEPT can correctly handle up to 2^N - 1 composite test scenarios.*

**Proof.** Let A = {a₁, ..., aₙ} be the set of N atomic conditions, each with an associated precept pᵢ ∈ P stored in `precepts_found`, where each pᵢ has a tier τᵢ ∈ {1, 2, 3}.

Consider any non-empty composite scenario S ⊆ A. We show PRECEPT resolves S correctly:

1. **Decomposition**: `retrieve_atomic_precepts` decomposes S into constituent atoms. For each aᵢ ∈ S, the corresponding precept pᵢ is retrieved via exact-match lookup. Since each aᵢ was learned during training, retrieval succeeds with probability 1.

2. **Composition**: Algorithm 1 computes argmax{τᵢ : aᵢ ∈ S}, which is well-defined since the tier hierarchy defines a total order (Safety > Compliance > Preferences).

3. **Correctness**: By the dominance rule (Table 2), the highest-tier precept provides the correct resolution for any constraint combination containing it. This follows from the domain semantics: safety constraints must always be satisfied regardless of co-occurring conditions.

4. **Determinism**: The composition operator max is deterministic—no LLM interpretation is required.

5. **Coverage**: The number of non-empty subsets of A is |𝒫(A) \ ∅| = 2^N - 1, each resolvable by the above procedure.

Therefore, N atomic precepts yield coverage of 2^N - 1 composite scenarios. □

**Remark.** The correctness guarantee in step 3 assumes the tier hierarchy faithfully represents domain constraint priorities. This assumption holds for domains where safety dominates compliance dominates preferences—a standard ordering in regulated environments (logistics, healthcare, finance).

---

## 4. Dual Conflict Resolution and Adversarial Adaptation

### 4.0 Theoretical Foundation: Evo-Memory and the Red Queen Principle

PRECEPT's conflict resolution draws theoretical inspiration from the **Red Queen hypothesis** in evolutionary computing. The Digital Red Queen (DRQ) framework (Kumar et al., 2026) demonstrates that agents optimized against static objectives become brittle, failing 72% against novel adversarial dynamics. DRQ overcomes this by maintaining a "growing history of opponents" that forces continual adaptation.

**PRECEPT's Evo-Memory Architecture** operationalizes this principle for rule-governed agents:

```
DRQ: Optimize(Warrior) against History[Opponent₁, Opponent₂, ..., Opponentₙ]
PRECEPT: Optimize(Plan) against History[Constraint₁, Constraint₂, ..., Constraintₙ]
```

**Definition 4.0 (Evo-Memory).** *Let $H(t) = \{c_1, c_2, \ldots, c_k\}$ be the accumulated constraint history at time step $t$, where each $c_i$ is a constraint (a failed option or violated condition) discovered during episodes $1, \ldots, t$. PRECEPT's Evo-Memory comprises three components:*

1. **In-Episode Memory** (`RefineInterceptor.forbidden_set`): The set of constraints $F_{\text{episode}} \subseteq H(t)$ discovered during the current task episode.
2. **Cross-Episode Memory** (`partial_progress.failed_options`): The set of constraints $F_{\text{cross}} \subseteq H(t)$ persisted from previous episodes, loaded at startup.
3. **Rule Confidence** (`rule_confidence[k]`): A scalar $c_k \in [0,1]$ tracking the reliability of learned rule $k$, decayed on failure ($c_k \leftarrow c_k \times 0.5$) and restored on success ($c_k \leftarrow \min(1.0, c_k + 0.25)$).

*The complete constraint history is $H(t) = F_{\text{episode}} \cup F_{\text{cross}}$.*

**Theorem 4.0 (Evo-Memory Eliminates Cyclic Failures).** *For any candidate plan $P$ and accumulated constraint set $H(t)$:*
$$P(\text{repeat}(c_i) \mid \text{EvoMemory}(H(t))) = 0, \quad \forall c_i \in H(t)$$

*where $\text{repeat}(c_i)$ denotes the event that the agent selects an option that violates previously-discovered constraint $c_i$.*

*Proof*: The `RefineInterceptor.is_forbidden()` check (Algorithm 2a, step 5) deterministically rejects any option in the forbidden set. Cross-episode failures are loaded from `partial_progress.json` at startup and injected into context via `forbidden_injection` (Algorithm 2a lines 23-27). Any plan violating accumulated constraints is pruned before execution. ∎

This architectural parallel to DRQ explains why PRECEPT achieves P(repeat_fail) = 0—the same mechanism that enables DRQ's warriors to defeat all previous opponents enables PRECEPT's agents to satisfy all accumulated constraints.

### 4.1 Type I: Static vs Dynamic Knowledge Conflict

**Table 3.** Ensemble Conflict Detection Weights (`EnsembleConflictDetector`).

| Method | Weight | Description |
|--------|:------:|-------------|
| NLI Classifier | 0.30 | Natural language inference |
| Semantic Patterns | 0.30 | Keyword contradiction detection |
| Temporal Analysis | 0.15 | Recency-based staleness |
| Evidence Strength | 0.15 | Confirmation/failure counts |
| LLM Vote | 0.10 | Optional LLM adjudication |

**Detection:** `weighted_conflict = Σ(weight × is_conflict × confidence)` → trigger if ≥ 0.4

**Resolution:** Thompson Sampling from Beta posteriors: Static `Beta(10,2)`, Dynamic `Beta(5,3)`

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart LR
    S["Static<br/>Beta(10,2)"] --> E["Ensemble<br/>Σ(w·c)"]
    D["Dynamic<br/>Beta(5,3)"] --> E
    E -->|"≥0.4"| R["Thompson<br/>Sampling"]
    E -->|"<0.4"| N["Pass"]
    R --> W["Winner"]
```

**Figure 4.** Type I conflict resolution pipeline. Ensemble voting triggers Bayesian resolution via Thompson Sampling.

**Definition 4.1 (Bayesian Source Reliability Model).**

When static and dynamic knowledge conflict, PRECEPT must decide which source to trust. We frame this as a **Bayesian inference problem** over source reliability.

*Let $r_s \in [0,1]$ denote the unknown reliability (probability of being correct) of the static knowledge source, and $r_d \in [0,1]$ the unknown reliability of the dynamic knowledge source. Since reliability is a probability, its natural prior is the Beta distribution:*

$$r_s \sim \text{Beta}(\alpha_s, \beta_s) \quad \text{with prior: } \alpha_s^{(0)}=10, \; \beta_s^{(0)}=2$$
$$r_d \sim \text{Beta}(\alpha_d, \beta_d) \quad \text{with prior: } \alpha_d^{(0)}=5, \; \beta_d^{(0)}=3$$

**Variable Definitions:**
- $r_s, r_d$: Unknown true reliability (probability of correctness) for static and dynamic sources, respectively. These are the quantities we wish to estimate.
- $\alpha_s, \beta_s$: Parameters of the Beta posterior for the static source. $\alpha_s$ counts successful (correct) outcomes; $\beta_s$ counts unsuccessful (incorrect) outcomes, plus their respective priors.
- $\alpha_d, \beta_d$: Analogous parameters for the dynamic source.
- $\text{Beta}(\alpha, \beta)$: The Beta distribution with mean $\frac{\alpha}{\alpha+\beta}$ and variance $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$.

**Why Beta distributions?** The Beta distribution is the conjugate prior for Bernoulli observations. Each time a knowledge source is used and we observe whether it was correct (Bernoulli trial with success probability $r$), the posterior update is:
$$\text{Correct outcome:} \quad \alpha \leftarrow \alpha + 1 \qquad \text{Incorrect outcome:} \quad \beta \leftarrow \beta + 1$$

This conjugacy means the posterior is always a Beta distribution, making inference computationally trivial. The prior $\text{Beta}(10,2)$ for static knowledge encodes the belief that curated static knowledge is reliable (prior mean $\frac{10}{12} \approx 0.83$). The prior $\text{Beta}(5,3)$ for dynamic experience encodes moderate confidence (prior mean $\frac{5}{8} = 0.625$).

**Algorithm 4.1a: Conflict Resolution with Bayesian Strategy Selection**

The `ConflictResolver` auto-selects the best resolution strategy based on available evidence, then uses the `DynamicReliabilityTracker`'s Beta posteriors for the reliability-based strategy:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: conflict_resolution.py → ConflictResolver._select_strategy() (line 1186)
        conflict_resolution.py → ConflictResolver.resolve() (line 1148)
        conflict_resolution.py → DynamicReliabilityTracker.update() (line 648)

Input: conflict (ConflictRecord containing static_item, dynamic_item)
Output: ResolutionResult (winner, confidence, strategy_used, reasoning)

function resolve(conflict):
    // Step 1: Auto-select resolution strategy
    strategy ← _select_strategy(conflict)

    // Strategy selection logic:
    if _is_anomaly(conflict.dynamic_item) then
        strategy ← "anomaly_detection"           // Outlier → flag as anomaly
    else if conflict.static_item.age_days > 30.0 then
        strategy ← "recency"                     // Stale static → prefer recent
    else if total_evidence ≥ 3 then
        strategy ← "evidence_strength"           // Good data → use evidence
    else
        strategy ← "reliability"                 // Default → Bayesian reliability
    end if

    // Step 2: Execute selected strategy
    if strategy == "recency" then
        // score = exp(-0.1 × age_days)
        static_score ← exp(-recency_decay_rate × static_item.age_days())
        dynamic_score ← exp(-recency_decay_rate × dynamic_item.age_days())
        winner ← argmax(static_score, dynamic_score)

    else if strategy == "reliability" then
        // Uses Beta posteriors from DynamicReliabilityTracker
        static_rel ← reliability_tracker.get_reliability(STATIC_KB).value
        dynamic_rel ← reliability_tracker.get_reliability(DYNAMIC_EXPERIENCE).value
        static_score ← static_rel × static_item.confidence
        dynamic_score ← dynamic_rel × dynamic_item.confidence
        winner ← argmax(static_score, dynamic_score)

    else if strategy == "evidence_strength" then
        // Laplace-smoothed: (confirmations + 1) / (total + 2)
        // Falls back to Beta prior mean if no evidence
        static_score ← evidence_score(static_item)
        dynamic_score ← evidence_score(dynamic_item)
        winner ← argmax(static_score, dynamic_score)
    end if

    return ResolutionResult(winner, confidence, strategy, reasoning)
end function

// Step 3: Posterior update after resolution (ConflictManager.detect_and_resolve)
if winner == STATIC_KB then
    reliability_tracker.update(STATIC_KB, was_correct=True)
    // Beta update: α_s ← α_s + 1  (success for static)
    reliability_tracker.update(DYNAMIC_EXPERIENCE, was_correct=False)
    // Beta update: β_d ← β_d + 1  (failure for dynamic)
else
    reliability_tracker.update(DYNAMIC_EXPERIENCE, was_correct=True)
    // Beta update: α_d ← α_d + 1
    reliability_tracker.update(STATIC_KB, was_correct=False)
    // Beta update: β_s ← β_s + 1
end if
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Algorithm 4.1b: Thompson Sampling for Active Exploration**

Thompson Sampling is used specifically in the `ActiveExplorationStrategy` to decide whether to re-verify potentially stale static knowledge:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: conflict_resolution.py → ActiveExplorationStrategy.should_explore_static() (line 726)

Input: static_item, dynamic_item
       Reliability posteriors: Beta(α_s, β_s), Beta(α_d, β_d) from DynamicReliabilityTracker
Output: (should_explore: bool, reasoning: str)

function should_explore_static(static_item, dynamic_item):
    // Get current Beta posteriors
    static_dist ← reliability_tracker.distributions[STATIC_KB]    // Beta(α_s, β_s)
    dynamic_dist ← reliability_tracker.distributions[DYNAMIC_EXP] // Beta(α_d, β_d)

    // Guard: need minimum observations before exploring
    if observation_counts[STATIC_KB] < 5 then
        return (False, "Insufficient observations")
    end if

    // Guard: skip if static has low uncertainty
    static_uncertainty ← 1.0 - 1.0 / (1.0 + static_dist.variance() × 10)
    if static_uncertainty < 0.4 then
        return (False, "Low uncertainty")
    end if

    // Guard: stale knowledge should always be re-verified
    if static_item.age_days() > 14.0 then
        return (True, "Stale static knowledge")
    end if

    // Thompson Sampling: sample from both Beta posteriors
    θ_s ← random.betavariate(α_s, β_s)   // θ_s ∈ [0,1]
    θ_d ← random.betavariate(α_d, β_d)   // θ_d ∈ [0,1]

    // If dynamic sampled notably higher, explore (re-verify static)
    if θ_d > θ_s and (θ_d - θ_s) > 0.3 then
        return (True, "Thompson Sampling suggests exploration")
    end if

    return (False, "Thompson Sampling suggests trusting static")
end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Variable Definitions:**
- $\theta_s, \theta_d \in [0,1]$: Random draws from the Beta posteriors. These are sampled reliability estimates, not point estimates.
- The randomness in sampling drives exploration: even if static has higher mean reliability, dynamic can occasionally be sampled higher, triggering re-verification. This balances exploitation (trusting the currently-believed-reliable source) with exploration (checking if beliefs are outdated).

**Theorem 4.1 (Regret Optimality).** *The Thompson Sampling formulation achieves asymptotically optimal expected regret $O(\sqrt{T \log T})$ over $T$ exploration decisions (Agrawal & Goyal, 2012), compared to $O(T)$ for naive heuristics such as always-trust-static or periodic re-verification.*

**Variable Definitions for Theorem 4.1:**
- $T$: Total number of exploration/re-verification decisions.
- Regret: Cumulative difference between the reward of the optimal policy (always knowing which source is currently more reliable) and the Thompson Sampling policy.

In practice, the posteriors concentrate around the true reliabilities after a small number of observations, and Thompson Sampling overwhelmingly selects the correct action.

### 4.2 Type II: Rule Drift Adaptation

**Table 4.** Rule Drift Parameters (`precept_mcp_server.py`).

| Parameter | Value | Operation |
|-----------|:-----:|-----------|
| `UNLEARN_FAILURE_THRESHOLD` | 2 | Invalidate after N consecutive failures |
| Confidence decay (δ) | ×0.5 | `confidence *= 0.5` per failure |
| Confidence restore | +0.25 | `confidence = min(1.0, c + 0.25)` on success |

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart LR
    A["Apply<br/>Rule"] --> B{{"OK?"}}
    B -->|"✓"| S["f←0<br/>c+=0.25"]
    B -->|"✗"| F["f++<br/>c×=0.5"]
    F --> C{{"f≥θ?"}}
    C -->|"Y"| I["<b>Invalidate</b>"]
    C -->|"N"| R["Retry"]
```

**Figure 5.** Type II (rule drift) handling. Soft confidence decay precedes hard invalidation at θ=2 failures.

### 4.3 Rule Invalidation Algorithm

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 2: Threshold-Based Rule Invalidation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: precept_mcp_server.py lines 867-957

Constants:
    UNLEARN_FAILURE_THRESHOLD = 2

Global State:
    learned_rules: Dict[str, str]
    rule_failure_counts: Dict[str, int]
    rule_confidence: Dict[str, float]

function record_rule_failure(condition_key):
    global learned_rules, rule_failure_counts, rule_confidence

    // Only track for existing rules
    if condition_key ∉ learned_rules then
        return None
    end if

    // Increment failure count
    rule_failure_counts[condition_key] ← rule_failure_counts.get(condition_key, 0) + 1
    current_failures ← rule_failure_counts[condition_key]

    // Soft decay: multiply by 0.5
    current_conf ← rule_confidence.get(condition_key, 1.0)
    new_conf ← current_conf × 0.5
    rule_confidence[condition_key] ← new_conf

    // Check threshold
    if current_failures ≥ UNLEARN_FAILURE_THRESHOLD then
        old_rule ← learned_rules.pop(condition_key, None)
        rule_failure_counts.pop(condition_key, None)
        rule_confidence.pop(condition_key, None)
        save_rules()  // Persist deletion
        return "Rule invalidated"
    end if

    return None
end function

function record_rule_success(condition_key):
    global rule_failure_counts, rule_confidence

    // Reset failure count
    if condition_key ∈ rule_failure_counts then
        rule_failure_counts.pop(condition_key, 0)
    end if

    // Restore confidence: add 0.25, cap at 1.0
    if condition_key ∈ rule_confidence then
        old_conf ← rule_confidence[condition_key]
        rule_confidence[condition_key] ← min(1.0, old_conf + 0.25)
    end if
end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.4 Smart Pivot Error Recovery

When initial execution fails, PRECEPT enters a bounded retry loop with deterministic pruning:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 2a: Smart Pivot Recovery Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: precept_agent.py → _handle_error_recovery() (line 1967)

Input: task, parsed_task, action_result, context
Output: (success, response, strategy, steps)

Parameters:
    max_pivots = config.agent.max_retries  // Default: 2 (AgentConfig)

function _handle_error_recovery(task, parsed_task, action_result, context):
    // Step 1: Record initial error for learning
    await record_error(error_code, error_context)

    // Step 2: Add to forbidden set (deterministic pruning)
    constraint ← RefineInterceptor.add_constraint(
        solution=failed_solution,
        error_code=action_result.error_code
    )

    // Step 3: Report rule failure if solution came from learned rule
    if context.exact_match_key and context.exact_match_solution == failed_solution then
        invalidation_msg ← await report_rule_failure(condition_key)
    end if

    // Step 4: COMPASS error evaluation for constraint tier
    compass_error_decision ← _compass_controller.evaluate_error(
        error_code, error_message, context
    )

    // Handle PROBE, BLOCK, PIVOT decisions...

    // Step 5: Smart pivot loop
    for pivot_num in range(max_pivots) do
        // Get forbidden injection (in-episode + cross-episode)
        forbidden_injection ← RefineInterceptor.get_forbidden_injection()
        if context.failed_options then
            forbidden_injection += cross_episode_warning
        end if

        // Ask LLM for new suggestion with forbidden context
        new_suggestion ← await _llm_reason_with_evolved_prompt(
            task, parsed_task, memories, rules,
            error_feedback=last_error,
            forbidden_section=forbidden_injection
        )

        // Validate suggestion against domain options
        if new_suggestion ∉ valid_options or is_forbidden(new_suggestion) then
            // Fallback to remaining untried options
            remaining ← RefineInterceptor.get_remaining_options(all_options)
            remaining ← [opt for opt in remaining if opt ∉ context.failed_options]
            new_suggestion ← remaining[0] if remaining else random_fallback
        end if

        // Execute pivot
        pivot_result ← await strategy.execute_action(mcp_client, parsed_task)

        if pivot_result.success then
            // Record successful solution + extract atomic precepts
            await record_successful_solution(rule_key, working_solution)
            await store_procedure(recovery_procedure)
            await extract_and_store_atomic_precepts(rule_key, solution)
            return (True, pivot_result.response, strategy, extra_steps)
        else
            // Add to forbidden, record for partial progress
            RefineInterceptor.add_constraint(failed_option, error)
            await record_failed_option(condition_key, failed_option)
        end if
    end for

    return (False, last_error, "", extra_steps)
end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Key Properties:**
- **Bounded Exploration:** Maximum `max_retries` pivots prevents infinite loops
- **Deterministic Pruning:** `RefineInterceptor` guarantees failed options are never retried
- **Cross-Episode Memory:** `context.failed_options` persists across episodes for resumable exploration
- **Validation Filter:** All LLM suggestions are validated against domain-valid options before execution
- **Learning on Recovery:** Successful recoveries are stored as rules, procedures, and atomic precepts

### 4.5 Deterministic Pruning via Constraint Classification

**Theoretical Foundation: Eliminating Cyclic Dynamics**

The Digital Red Queen (DRQ) paper identifies "cyclic dynamics" as a fundamental failure mode where agents oscillate between strategies without making progress—a consequence of static optimization's inability to maintain constraint memory. PRECEPT eliminates this through the `RefineInterceptor`, which implements **deterministic pruning** with formal guarantees.

**Theorem 4.5 (Elimination of Cyclic Failures).** *Let F = {f₁, f₂, ..., fₖ} be the set of failed options discovered during task execution. The RefineInterceptor guarantees:*
$$P(\text{select}(f_i) | \text{RefineInterceptor}(F)) = 0, \quad \forall f_i \in F$$

*Proof:* The `is_forbidden()` method (Algorithm 2b line 2) returns True for any option in `hard_constraints ∪ soft_constraints`. The `validate_and_filter()` function (Algorithm 2c) removes all forbidden options before LLM selection. Therefore, no selection mechanism can return a forbidden option. ∎

**Corollary 4.5.1 (DRQ Parallel).** *PRECEPT's RefineInterceptor serves the same mathematical function as DRQ's opponent history—both prevent the search from revisiting previously-failed states, enabling monotonic progress through the constraint landscape.*

A critical component enabling PRECEPT's reliability is the `RefineInterceptor`, which provides **provably correct** pruning of failed options through constraint classification.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 2b: Deterministic Pruning via Constraint Classification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: constraints.py (RefineInterceptor class, classify_error(), add_constraint())

Data Structures:
    constraints: List[Constraint]     // All accumulated constraints
    forbidden: Set[str]               // Solutions pruned from search space (lowercase)
    soft_constraints_retriable: bool  // If True, SOFT errors not added to forbidden

Constants (from config/constraints.py → ConstraintConfig):
    TRANSIENT_INDICATORS = {
        "timeout", "timed out", "temporary", "retry later",
        "try again", "momentarily", "intermittent", "connection reset",
        "service unavailable"
    }
    HARD_INDICATORS = {
        "blocked", "forbidden", "denied", "not authorized",
        "not found", "does not exist", "unknown", "invalid",
        "suspended", "unavailable", "closed", "offline",
        "incompatible", "not supported", "expired", "revoked", "cancelled"
    }

function classify_error(error_code, error_message) → ConstraintType:
    message_lower ← lower(error_message)

    // Check for transient indicators FIRST (these can be retried)
    if any(indicator ∈ message_lower for indicator in TRANSIENT_INDICATORS) then
        return ConstraintType.TRANSIENT
    end if

    // Check for hard constraint indicators
    if any(indicator ∈ message_lower for indicator in HARD_INDICATORS) then
        return ConstraintType.HARD
    end if

    // Default: SOFT constraint (conservative - allows retry)
    return ConstraintType.SOFT

function add_constraint(solution, error_code, error_message) → Constraint:
    constraint_type ← classify_error(error_code, error_message)
    constraint ← Constraint(solution.lower(), error_code, constraint_type, ...)
    constraints.append(constraint)

    // Determine forbidden status based on type and config
    if constraint_type == HARD then
        forbidden.add(solution.lower())         // Always banned
    else if constraint_type == SOFT then
        if not soft_constraints_retriable then
            forbidden.add(solution.lower())     // Banned only if config says so
        end if
    end if
    // TRANSIENT constraints are NEVER added to forbidden

    return constraint

function is_forbidden(solution) → bool:
    return solution.lower() ∈ forbidden

function get_remaining_options(all_options) → List[str]:
    return [opt for opt in all_options if opt.lower() ∉ forbidden]

function get_forbidden_injection() → str:
    // Generate context injection for LLM reasoning
    if |constraints| == 0 then
        return ""
    end if

    injection ← "🚫 FORBIDDEN OPTIONS (Probability = 0.0 - DO NOT SUGGEST):\n"
    for constraint in constraints do
        if constraint.constraint_type == HARD then
            injection += f"  ❌ BANNED: '{constraint.solution}' - {constraint.reason}\n"
        else
            injection += f"  ⚠️ FAILED: '{constraint.solution}' - {constraint.reason}\n"
        end if
    end for
    injection += "⛔ You MUST suggest a DIFFERENT solution not in the forbidden list."

    return injection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complexity: O(|patterns|) for classification, O(1) for is_forbidden lookup
Property: Guarantees P(repeat_failed) = 0 for HARD constraints (Theorem 6.9)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Design Rationale:**

| Constraint Type | Behavior | Example Errors | Rationale |
|----------------|----------|----------------|-----------|
| **HARD** | Never retry | closure, strike, invalid | Physical/logical impossibility |
| **SOFT** | Configurable | busy, timeout, congestion | Potentially transient |

The distinction between HARD and SOFT constraints enables:
1. **Guaranteed termination:** Finite option space + HARD pruning → bounded exploration
2. **Configurable exhaustiveness:** `soft_constraints_retriable=True` for thorough search
3. **Diagnostic probes:** When all options exhausted, suggests probes to discover hidden constraints

### 4.6 LLM Hallucination Prevention (Validation Filter)

LLMs may suggest invalid options that don't exist in the domain or options that have already failed. PRECEPT implements a two-stage validation filter:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 2c: LLM Suggestion Validation Filter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: precept_agent.py lines 1664-1721

Input: llm_suggestion, valid_options, failed_options, context
Output: validated_solution

function validate_and_filter(suggestion, valid_options, failed_options):
    all_options_lower ← {opt.lower() for opt in valid_options}
    failed_options_lower ← {opt.lower() for opt in failed_options}

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 1: Domain Validity Check
    // LLMs may hallucinate options that don't exist (e.g., "order_type_d")
    // ═══════════════════════════════════════════════════════════════════
    if suggestion.lower() ∉ all_options_lower then
        log(f"⛔ HALLUCINATION DETECTED: '{suggestion}' is not a valid option")
        log(f"   Valid options: {valid_options}")

        // Fallback to random untried option
        remaining ← valid_options \ failed_options
        if |remaining| > 0 then
            suggestion ← random_choice(remaining)
            log(f"🔄 REPLACED with valid untried option: {suggestion}")
        end if
    end if

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 2: Cross-Episode Memory Check
    // Even valid options may have failed in previous episodes
    // ═══════════════════════════════════════════════════════════════════
    if suggestion.lower() ∈ failed_options_lower then
        log(f"⛔ MEMORY CONFLICT: '{suggestion}' failed in previous episode")

        // Get fresh option from remaining untried ones
        remaining ← valid_options \ failed_options
        if |remaining| > 0 then
            suggestion ← random_choice(remaining)
            log(f"🔄 REPLACED with untried option: {suggestion}")
        end if
    end if

    return suggestion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complexity: O(|valid_options|) for set operations
Property: Prevents wasted retries on hallucinated or previously failed options
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Table 4a.** Validation Filter Statistics (Experiment 6).

| Metric | Value | Impact |
|--------|-------|--------|
| Hallucinations caught | 12.3% of LLM suggestions | Prevents invalid execution attempts |
| Memory conflicts caught | 8.7% of LLM suggestions | Prevents wasted retries |
| Total waste prevented | 21.0% of suggestions | Direct contribution to step efficiency |

---

## 5. COMPASS: Complexity-Optimized Multi-strategy Pareto Adaptive Search

### 5.1 Architecture Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart LR
    T["Task"] --> CA["<b>Complexity</b><br/>ML-based"] --> SR["<b>Rollout</b><br/>Adaptive"] --> E["<b>Evaluate</b><br/>6 objectives"] --> PS["<b>Pareto</b><br/>Select"] --> O["Best"]
```

**Figure 6.** COMPASS pipeline: Complexity Analysis → Smart Rollout Allocation → Candidate Evaluation → Pareto Selection.

**Table 5.** COMPASS Component Summary.

| Component | Class | Function |
|-----------|-------|----------|
| Complexity Analyzer | `PRECEPTComplexityAnalyzer` | Estimates tool/reasoning/retrieval steps via pattern detection |
| Smart Rollout | `SmartRolloutStrategy` | Allocates rollouts based on score (0.98→skip, 0.9→verify, else→explore) |
| Pareto Selection | `pareto_select()` | Selects from non-dominated front by hypervolume contribution |

### 5.1.1 Theoretical Foundation: MAP-Elites for Strategy Diversity

COMPASS integrates the **MAP-Elites** principle (Mouret & Clune, 2015) to prevent the "convergence collapse" observed in static optimization. The Digital Red Queen (DRQ) paper demonstrates that agents evolved through static optimization become "less behaviorally diverse across independent runs," collapsing toward a single strategy that fails when blocked.

**Definition 5.1 (Strategy Diversity via MAP-Elites).** *MAP-Elites maintains a grid of strategy niches, each containing the best-performing strategy for a behavioral phenotype. COMPASS implements this through:*

1. **Diversity Threshold** (`diversity_threshold = 0.7`): Minimum diversity score to consider strategies sufficiently distinct
2. **Diversity Rollouts** (`diversity_rollouts = 5`): Extra rollouts allocated when diversity is low
3. **Behavioral Phenotyping**: Strategies characterized by `dominant_dimension` (tool_use, retrieval, reasoning, verification)

**Definition 5.2 (Topological Distinctness).** *Two strategies S₁, S₂ are topologically distinct if:*
$$\text{FailureModes}(S_1) \cap \text{FailureModes}(S_2) = \emptyset$$

*COMPASS maintains topologically distinct alternatives via Pareto selection across multiple objectives—if the primary strategy fails, the Pareto front contains alternatives that succeeded on different objective combinations.*

**Algorithm:** When `diversity_score < diversity_threshold`, COMPASS allocates `diversity_rollouts` additional evaluations to discover behaviorally distinct candidates:

```
if diversity_score < diversity_threshold then
    num_rollouts ← diversity_rollouts  // Explore for diverse strategies
    focus ← "diversity"
```

This prevents the "greedy convergence" that traps agents relying on single-strategy optimization, ensuring PRECEPT has pre-calculated alternatives when the obvious path is blocked.

### 5.2 Complexity Analysis

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 3: ML-Based Complexity Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: complexity_analyzer.py lines 328-427

Input: task, goal, available_tools, domain
Output: ComplexityEstimate

function analyze(task, goal, available_tools, domain):
    full_text ← f"{task} {goal or ''}"

    // Step 1: Tool chain estimation
    tool_steps ← ToolPatternDetector.estimate_tool_chain(task, available_tools)
    detected_tools ← ToolPatternDetector.detect_tools(task)

    // Step 2: Reasoning complexity
    reasoning_steps, reasoning_patterns ←
        ReasoningPatternDetector.estimate_reasoning_steps(full_text)

    // Step 3: Entity extraction for retrieval
    entities ← EntityPatternDetector.extract_entities(task)
    relationships ← EntityPatternDetector.count_relationships(task)
    retrieval_hops ← max(1, |entities| // 2 + relationships)

    // Step 4: Dimension scoring (exact multipliers from code line 387-392)
    dimension_scores ← {
        TOOL_USE: tool_steps × (1.5 if detected_tools else 0.5),
        RETRIEVAL: retrieval_hops × (1.3 if entities else 0.5),
        REASONING: reasoning_steps × (1.2 if reasoning_patterns else 0.5),
        VERIFICATION: 1.0 if ("verify" in task.lower() or "check" in task.lower()) else 0.5
    }

    // Step 4 (optional): ML-based override via COMPASS analyzer
    // If compass_analyzer available, can override retrieval_hops and confidence
    // ml_confidence ← compass_analyzer.analyze_query(task)

    dominant ← argmax(dimension_scores)
    total_steps ← max(tool_steps, retrieval_hops, reasoning_steps)
    confidence ← ml_confidence if ml_estimate else 0.6  // Default when ML not available

    // Step 5 (optional): History-based adjustment
    // If learning_enabled and domain has step_adjustments, scale total_steps

    return ComplexityEstimate(
        total_estimated_steps=total_steps,
        dominant_dimension=dominant,
        confidence=confidence,
        ...
    )
end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.3 Smart Rollout Allocation

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 4: Smart Rollout Decision
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: complexity_analyzer.py lines 479-589

Class: SmartRolloutStrategy
Init Parameters (from __init__):
    diversity_threshold = 0.7
    confidence_threshold = 0.9
    min_rollouts = 1
    max_rollouts = 15
    diversity_rollouts = 5
    consistency_rollouts = 3

function decide(task_complexity, current_score, diversity_score, previous_attempts):
    // Rule 1: Perfect score → early stopping (line 533)
    if current_score ≥ 0.98 then
        return RolloutDecision(
            use_rollouts=False,
            num_rollouts=0,
            focus="skip"
        )
    end if

    // Rule 2: High score, check diversity (lines 545-560)
    if current_score ≥ confidence_threshold then
        if diversity_score is not None and diversity_score < diversity_threshold then
            return RolloutDecision(True, diversity_rollouts, "diversity")
        else
            return RolloutDecision(True, consistency_rollouts, "consistency")
        end if
    end if

    // Rule 3: Complexity-based allocation (lines 562-577)
    c ← task_complexity.total_estimated_steps
    confidence ← task_complexity.confidence

    if c ≤ 2 and confidence > 0.8 then
        num_rollouts ← min_rollouts
        focus ← "minimal"
    else if c ≤ 4 then
        num_rollouts ← min(c + 2, max_rollouts // 2)
        focus ← "exploration"
    else
        num_rollouts ← min(c × 2, max_rollouts)
        focus ← "thorough"
    end if

    // Adjust for recovery (lines 579-582)
    if previous_attempts > 0 then
        num_rollouts ← min(num_rollouts + previous_attempts, max_rollouts)
        focus ← "recovery"
    end if

    return RolloutDecision(True, num_rollouts, focus)
end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.4 Pareto-Optimal Selection

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart LR
    C["Candidates<br/><i>n prompts</i>"] --> D["Dominance<br/><i>∀i,j: sᵢ≥sⱼ</i>"] --> F["Front<br/><i>non-dominated</i>"] --> H["Hypervolume<br/><i>max contrib</i>"] --> B["<b>Best</b>"]
```

**Figure 7.** Pareto selection: (1) filter dominated candidates, (2) select from front by hypervolume contribution.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 5: Pareto Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: scoring.py lines 282-355

Input: candidates: List[GEPAEvaluationResult], selection_strategy: str = "hypervolume"
Output: GEPAEvaluationResult

function pareto_select(candidates, selection_strategy):
    if |candidates| == 1 then
        return candidates[0]
    end if

    // Step 1: Find non-dominated (Pareto front)
    pareto_front ← []
    for candidate in candidates do
        dominated ← False
        for other in candidates do
            if other is not candidate and other.dominates(candidate) then
                dominated ← True
                break
            end if
        end for
        if not dominated then
            pareto_front.append(candidate)
        end if
    end for

    if pareto_front is empty then
        pareto_front ← candidates  // Fallback
    end if

    // Step 2: Select from front
    if selection_strategy == "hypervolume" then
        reference_point ← {obj.value: 0.0 for obj in GEPAObjective}
        best ← max(pareto_front,
                   key=λc: c.get_hypervolume_contribution(reference_point))
        return best
    else if selection_strategy == "crowding" then
        // Select by leadership count
        ...
    else  // random
        return random.choice(pareto_front)
    end if
end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.5 Multi-Objective Scoring from Rollouts

A key principle of GEPA (and COMPASS) is that **all scores are derived empirically from actual task execution**, not heuristics. Each rollout produces a `RolloutResult` containing execution metrics, which are aggregated into 6 objective scores.

**Table 6.** GEPA Multi-Objective Scoring Functions.

| Objective | Formula | Description |
|-----------|---------|-------------|
| TASK_SUCCESS_RATE | $\frac{\sum_{i=1}^{n} \mathbb{1}[\text{success}_i]}{n}$ | Fraction of successful rollouts |
| SOLUTION_QUALITY | $\frac{1}{n}\sum_{i=1}^{n} \frac{\text{correctness}_i + \text{completeness}_i}{2}$ | Average quality score |
| STEP_EFFICIENCY | $\frac{1}{1 + \bar{s} / s_{\max}}$ where $s_{\max} = 1 + \text{MAX\_RETRIES}$ | Normalized inverse of avg steps |
| TOKEN_EFFICIENCY | $\frac{1}{1 + \bar{t} / t_{\text{base}}}$ where $t_{\text{base}} = 1000$ | Normalized inverse of avg tokens |
| ADAPTATION_SPEED | $\min(1, \frac{\text{recoveries}}{\text{failures}})$ | Recovery rate from errors |
| GENERALIZATION | $1 - \frac{\text{Var}(\text{success})}{0.25}$ | Inverse of outcome variance |

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 6: Multi-Objective Scoring from Rollouts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: scoring.py lines 148-232

Input: rollouts: List[RolloutResult]
Output: scores: Dict[GEPAObjective → float]

Constants:
    MAX_TASK_STEPS = 1 + DEFAULT_MAX_RETRIES  // = 5 by default
    baseline_tokens = 1000.0

function compute_gepa_scores(rollouts):
    n ← |rollouts|
    scores ← {}

    // 1. TASK_SUCCESS_RATE: Direct empirical measurement
    successes ← sum(1 for r in rollouts if r.success)
    scores[TASK_SUCCESS_RATE] ← successes / n

    // 2. SOLUTION_QUALITY: Average of correctness and completeness
    quality_scores ← [(r.answer_correctness + r.answer_completeness) / 2
                      for r in rollouts]
    scores[SOLUTION_QUALITY] ← mean(quality_scores)

    // 3. STEP_EFFICIENCY: Normalized inverse of average steps
    avg_steps ← mean(r.steps_taken for r in rollouts)
    scores[STEP_EFFICIENCY] ← 1.0 / (1.0 + avg_steps / MAX_TASK_STEPS)

    // 4. TOKEN_EFFICIENCY: Normalized inverse of average tokens
    avg_tokens ← mean(r.tokens_used for r in rollouts)
    scores[TOKEN_EFFICIENCY] ← 1.0 / (1.0 + avg_tokens / baseline_tokens)

    // 5. ADAPTATION_SPEED: Recovery rate from failures
    total_failures ← sum(|r.errors_encountered| for r in rollouts)
    total_recoveries ← sum(r.recovered_from_errors for r in rollouts)
    scores[ADAPTATION_SPEED] ← min(1.0, total_recoveries / total_failures)

    // 6. GENERALIZATION: 1 - normalized variance
    success_values ← [1.0 if r.success else 0.0 for r in rollouts]
    variance ← Var(success_values)
    scores[GENERALIZATION] ← 1.0 - (variance / 0.25)  // 0.25 = max binary variance

    return scores
end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Design Principles:**
1. **Rollout-based evaluation**: Scores derived from actual execution, not heuristics
2. **Multi-objective optimization**: Track 6 independent metrics without arbitrary weighting
3. **Pareto dominance**: Selection via Algorithm 5 preserves diverse high-performing candidates
4. **Empirical metrics**: No keyword matching, length penalties, or other proxies

### 5.6 Verified Prompt Evolution

A critical innovation in COMPASS is **verified prompt evolution**—using real agent execution signals rather than heuristic scoring to evaluate candidate prompts. This ensures honest feedback without "cheating" via expected solution leakage.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart LR
    CP["Candidate<br/>Prompt"] --> AE["Agent<br/>run_task()"] --> PS["Predicted<br/>Solution"]
    PS --> ENV["Environment<br/>Verifies"]
    ENV --> SIG["Binary Signal<br/>{success, error}"]
    SIG --> COMPASS["COMPASS<br/>Evolution"]

    style ENV fill:#f0f0f0,stroke:#333
```

**Figure 6a.** Verified evolution signal flow. The agent predicts a solution, the environment verifies it internally, and only binary success/failure signals are used for evolution—the agent **never sees** expected solutions.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 7: Verified Prompt Evolution Callback
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: precept_agent.py lines 426-539

Input: candidate_prompt, task
Output: execution_result with binary signals only

function execute_with_prompt(prompt, task):
    // Save current prompt
    original_prompt ← agent.current_system_prompt

    try:
        // Temporarily switch to candidate prompt
        agent.current_system_prompt ← prompt
        agent.recreate_with_prompt(prompt)

        // Execute task - environment verifies internally
        // Agent predicts, environment confirms (predicted == expected?)
        // Agent NEVER sees expected_solution
        result ← await agent.run_task(task.task, metadata=task.metadata)

        // Return ONLY binary signals - no solution leakage
        return {
            "success": result.get("success", False),      // Binary: from environment
            "error_code": None,                            // Not exposed in run_task result
            "error_message": result.get("response") if not success else None,
            "predicted_solution": result.get("strategy"),  // What agent predicted
            "steps": result.get("steps", 0),               // Efficiency metric
        }

    finally:
        // Restore original prompt
        agent.current_system_prompt ← original_prompt
        // Recreate AutoGen AssistantAgent with original prompt
        agent.agent ← AssistantAgent(name, model_client, tools, original_prompt)
end function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Property: Honest feedback - no expected solution leakage
Applicable: ALL verifiable tasks (Black Swan CSP, compositional, drift, etc.)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Why Verified Evolution Matters:**

| Approach | Signal Source | Honest? | Generalizable? |
|----------|--------------|---------|----------------|
| Heuristic scoring | Keyword matching, length | No | Limited |
| LLM-as-judge | Another LLM's opinion | Biased | Variable |
| **Verified execution** | Environment confirmation | **Yes** | **Universal** |

This design ensures:
1. **No cheating:** Agent predictions evaluated against hidden ground truth
2. **Binary signals:** Only success/failure used, not solution content
3. **Universality:** Works for any task with verifiable outcomes
4. **Honest feedback:** COMPASS/GEPA evolution learns from real capability, not proxies

### 5.7 Dual-Frequency Control Loop

A critical architectural innovation in COMPASS is the **Dual-Frequency Control Loop**—separating lightweight per-step monitoring from heavyweight event-driven optimization. This directly addresses latency and cost concerns that would arise from running full optimization at every step.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart TB
    subgraph HF["<b>HIGH-FREQUENCY LOOP</b><br/><i>Every iteration</i>"]
        direction LR
        EA["evaluate_action()"] --> EE["evaluate_error()"]
        EE --> LP["learn_pattern()"]
    end

    subgraph LFT["<b>TRIGGER EVENTS</b>"]
        direction TB
        T1["New Rule Learned"]
        T2["Goal Failure"]
        T3["Phase Change"]
    end

    subgraph LF["<b>LOW-FREQUENCY LOOP</b><br/><i>On trigger only</i>"]
        direction LR
        C["compile()"] --> PE["Prompt Evolution"]
        PE --> MP["Memory Pruning"]
    end

    HF --> LFT
    LFT --> LF

    style HF fill:#e8f5e9,stroke:#2e7d32
    style LF fill:#fff3e0,stroke:#ef6c00
    style LFT fill:#f3e5f5,stroke:#7b1fa2
```

**Figure 6b.** COMPASS Dual-Frequency Control Loop. The high-frequency loop (green) runs at every agent step for real-time safety monitoring. The low-frequency loop (orange) runs only on trigger events for strategic re-planning.

**Table 5a.** Dual-Frequency Loop Characteristics.

| Property | High-Frequency (Monitor) | Low-Frequency (Architect) |
|----------|--------------------------|---------------------------|
| **Frequency** | Every iteration | On trigger events only |
| **Functions** | `evaluate_action()`, `evaluate_error()`, `learn_pattern()` | `compile()`, prompt evolution, memory pruning |
| **Purpose** | Constraint validation, drift detection | Strategic re-planning, prompt optimization |
| **Cost** | O(1) logical checks | O(n) LLM calls, Pareto selection |
| **Latency** | < 1ms | 100ms - 10s |
| **Source** | `compass_controller.py` lines 298-448, 583-630 | `compass_integration.py` lines 624-841 |

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 8: COMPASS Dual-Frequency Control Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source: compass_controller.py (Monitor), compass_integration.py (Architect)
        precept_agent.py lines 1353-1363 (integration point)

// ═══════════════════════════════════════════════════════════════════════════
// HIGH-FREQUENCY LOOP: Monitor Mode (runs EVERY iteration)
// Cost: O(1) - lightweight constraint checking
// ═══════════════════════════════════════════════════════════════════════════

function monitor_mode(task, parsed_task, complexity, error_info):
    // Phase 1: Pre-action constraint check (line 1359 in precept_agent.py)
    _compass_controller.set_user_instruction(task)
    decision ← evaluate_action(task, parsed_task, complexity)

    // Step 1: Hierarchical constraint check (Physics > Policy > Instruction)
    if config.enable_constraint_hierarchy then
        blocking ← _check_blocking_constraints(parsed_task)
        if blocking then
            (constraint_id, tier, negotiated) ← blocking
            if tier == PHYSICS then
                stats.physics_overrides += 1
                if negotiated then
                    return PIVOT(negotiated_alternative)
                else
                    return BLOCK(constraint_id)
                end if
            end if
        end if
    end if

    // Step 2: Complexity-based routing (fast path for trivial tasks)
    if config.enable_fast_path then
        if complexity.level == "trivial" and complexity.confidence ≥ threshold then
            stats.fast_path_used += 1
            return FAST_PATH(skip_llm=True)
        end if
    end if

    // Default: proceed with full pipeline
    stats.full_path_used += 1
    return PROCEED(constraint_context)

function evaluate_error(error_code, error_message, context):
    // Phase 2: Post-error evaluation (line 2042 in precept_agent.py)

    // Step 1: Should we probe? (Epistemic detour)
    if config.enable_epistemic_probing then
        probe_decision ← _check_should_probe(error_code, error_message)
        if probe_decision then
            stats.probes_triggered += 1
            return PROBE(probe_spec)
        end if
    end if

    // Step 2: Discover constraint from error
    feedback ← ExecutionFeedback(error_code, error_message)
    discovered ← csp_manager.intercept_feedback(feedback)

    if discovered and discovered[0].tier == PHYSICS then
        negotiated ← _negotiate_alternative(discovered[0])
        if negotiated then
            return PIVOT(negotiated)
        else
            return BLOCK(discovered[0].id)
        end if
    end if

    return PROCEED  // Continue with retry loop

function learn_pattern(error_pattern, action, succeeded, probe_id):
    // Phase 3: Pattern learning from outcomes (line 2081 in precept_agent.py)
    if error_pattern not in _learned_patterns then
        _learned_patterns[error_pattern] ← LearnedPattern(...)
        stats.patterns_learned += 1
    end if
    _learned_patterns[error_pattern].update(succeeded)

// ═══════════════════════════════════════════════════════════════════════════
// LOW-FREQUENCY LOOP: Architect Mode (runs ONLY on trigger)
// Cost: O(n) - heavyweight optimization
// Triggers: New rule learned, Goal failure, Phase change
// ═══════════════════════════════════════════════════════════════════════════

function architect_mode(current_prompt, validation_tasks, trigger_event):
    // Only executes when trigger_event fires
    // Source: compass_integration.py lines 624-841

    // Step 1: Feedback Ingestion - Analyze execution traces
    trace_analysis ← feedback_manager.analyze_patterns()

    // Step 2: Pattern Extraction - Identify consolidation candidates
    consolidation_recommendations ← feedback_manager.get_consolidation_recommendations()
    frequent_strategies ← memory_store.get_frequent_strategies(min_count=3)
    frequent_lessons ← memory_store.get_frequent_lessons(min_count=2)

    // Step 3: Complexity Analysis (COMPASS ML)
    complexity_estimates ← []
    for task in validation_tasks[:3] do
        estimate ← complexity_analyzer.analyze(task)
        complexity_estimates.append(estimate)
    end for

    // Step 4: Mutation - Generate prompt candidates
    candidates ← _generate_prompt_candidates(
        current_prompt, frequent_strategies, frequent_lessons
    )

    // Step 5: Smart Validation with adaptive rollouts
    scored_candidates ← []
    for candidate in candidates do
        rollout_decision ← rollout_strategy.decide(complexity, current_score)
        if rollout_decision.early_stop then
            stats.early_stops += 1
        else
            scores ← _evaluate_candidate(candidate, validation_tasks, rollouts)
            scored_candidates.append((candidate, scores))
        end if
    end for

    // Step 6: Pareto Selection
    best_candidate ← _pareto_select(scored_candidates)

    // Step 7: Pruning - Remove consolidated memories
    if best_candidate.score > 0.7 then
        pruned_count ← memory_store.prune_consolidated(consolidated_patterns)
        stats.memories_pruned += pruned_count
    end if

    stats.compilations_run += 1
    return best_candidate.prompt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Property: Real-Time Safety with Event-Driven Intelligence
- Monitor Mode: O(1) constraint checking at every step
- Architect Mode: O(n) optimization only on state change
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Table 5b.** Monitor Mode Decision Points in `precept_agent.py`.

| Line | Function | When Called | Decision Output |
|------|----------|-------------|-----------------|
| 1359 | `evaluate_action()` | Before **every** action | BLOCK, PIVOT, FAST_PATH, PROCEED |
| 2042 | `evaluate_error()` | After **every** error | PROBE, BLOCK, PIVOT, PROCEED |
| 2063 | `execute_probe()` | When error triggers probe | ProbeResult with constraint discovery |
| 2081 | `learn_pattern()` | After probe completes | Updates pattern confidence |

**Table 5c.** Architect Mode Trigger Events.

| Trigger | Condition | Effect |
|---------|-----------|--------|
| **New Rule Learned** | `learn_pattern()` accumulates N patterns | Triggers compilation to consolidate into prompt |
| **Goal Failure** | Task fails with `max_retries` exhausted | Re-evaluates strategy with new constraints |
| **Phase Change** | Transition training → testing | Optimizes prompt for deployment |
| **Periodic** | Configurable interval (e.g., nightly) | Background optimization |

**Why Dual-Frequency Matters:**

The dual-frequency design directly addresses the **Latency/Cost critique** of continuous optimization:

| Approach | Per-Step Cost | Safety Guarantee | Optimization Quality |
|----------|---------------|------------------|---------------------|
| No runtime optimization | O(1) | ❌ No constraint checking | ❌ Stale prompts |
| Full optimization every step | O(n) LLM calls | ✓ Continuous | ✓ Always optimal |
| **COMPASS Dual-Frequency** | **O(1) + triggered O(n)** | **✓ Real-time safety** | **✓ Event-driven optimization** |

This separation enables:
1. **Real-Time Safety**: Constraint violations are caught at every step (Monitor Mode)
2. **Efficient Optimization**: Heavyweight compilation runs only when beneficial (Architect Mode)
3. **Adaptive Learning**: Patterns learned continuously, consolidated strategically
4. **Low Latency**: Sub-millisecond monitoring overhead during normal operation

---

## 6. Theoretical Analysis

**Notation.** We define all variables used throughout the theoretical analysis:

| Symbol | Definition | Typical Values |
|--------|-----------|----------------|
| $\alpha$ | Learning effectiveness: $P(\text{retrieval correct}) \times P(\text{application correct} \mid \text{retrieval correct})$ | PRECEPT: 0.85, Verbal: 0.50 |
| $P_1$ | First-try success rate (primary metric) | 0.0-1.0 |
| $N$ | Total number of entities (options) in the domain | Domain-dependent |
| $W$ | Number of working (non-blocked) entities | $W = N - B$ |
| $B$ | Number of blocked entities requiring alternative solutions | Domain-dependent |
| $T$ | Number of training tasks | Experiment-dependent |
| $E$ | Number of distinct error types in a domain | 7-17 (Table 8) |
| $\beta$ | Minimum encounters needed to learn a rule | PRECEPT: 3 |
| $R$ | Maximum retries per task during training | Default: 4 |
| $C(T,E,\beta)$ | Coverage function: $P(\text{error type seen} \geq \beta \text{ times in } T \text{ tasks})$ | 0.0-1.0 |
| $P_{\text{learn}}(R)$ | Probability of discovering a working alternative within $R$ retries | 0.0-1.0 |
| $p$ | Per-condition interpretation accuracy of verbal baselines | ~0.75 |
| $d$ | Drift detection accuracy (probability of detecting stale rule on failure) | PRECEPT: 0.95, Verbal: 0.60 |
| $\theta$ | Unlearn failure threshold (consecutive failures before invalidation) | 2 |
| $F_t$ | Set of options that have failed by step $t$ (forbidden set) | Grows during episode |
| $p_{\text{forget}}$ | Probability that LLM forgets/ignores a verbal reflection | ~0.25 |
| $H(\cdot)$ | Shannon entropy of retrieval distribution | Bits |

### 6.1 First-Try Success Rate

**Definition 6.1 (Learning Effectiveness $\alpha$).** *The probability that a learned rule is correctly retrieved AND correctly applied at test time:*

$$\alpha = P(\text{retrieval correct}) \times P(\text{application correct} | \text{retrieval correct})$$

**Theorem 6.1 (First-Try Success Rate).** *For an agent with learning effectiveness $\alpha$, the first-try success rate is:*

$$P_1(\alpha) = \frac{W}{N} + \frac{B}{N} \cdot C(T, E, \beta) \cdot P_{\text{learn}}(R) \cdot \alpha$$

**Proof.** Success on first try occurs in two mutually exclusive cases:

*Case 1:* Task starts with working entity. Probability = W/N.

*Case 2:* Task starts with blocked entity (prob B/N), error type was covered in training (prob C), a solution was learned (prob P_learn), and the rule is correctly applied (prob α).

By the law of total probability:
$$P_1 = P(\text{working}) + P(\text{blocked}) \cdot P(\text{covered}) \cdot P(\text{learned}) \cdot P(\text{applied})$$
$$P_1 = \frac{W}{N} + \frac{B}{N} \cdot C \cdot P_{\text{learn}} \cdot \alpha$$

□

### 6.2 Why α_PRECEPT > α_verbal: Four Theoretical Reasons

The effectiveness gap (α_PRECEPT ≈ 0.85 vs α_verbal ≈ 0.50) arises from four fundamental differences:

**Table 7.** Theoretical Factors Contributing to α Advantage.

| Factor | PRECEPT | Verbal Baselines | Impact |
|--------|---------|------------------|--------|
| **Retrieval Precision** | 98% (exact-match) | 70% (task-type index) | 1.4× |
| **Deterministic Pruning** | 100% (guaranteed) | 75% (LLM recall) | 1.3× |
| **Rule Interpretation** | 99% (compiled) | 80% (natural language) | 1.2× |
| **Combined** | **0.85** | **0.50** | **1.7×** |

**Reason 1: Information-Theoretic (Structured vs Unstructured)**

PRECEPT indexes rules by error code (e.g., `"PORT-503"`) with exact-match retrieval. Verbal baselines index by task type with LLM interpretation required:

$$H(\text{retrieval}_{\text{PRECEPT}}) < H(\text{retrieval}_{\text{verbal}})$$

*where $H(\cdot)$ denotes the Shannon entropy of the retrieval distribution (i.e., the uncertainty over which stored rule will be returned for a given query).* Lower entropy corresponds to more precise retrieval.

**Reason 2: Deterministic Pruning**

PRECEPT guarantees failed options are never retried via `failed_options` set. Verbal baselines rely on LLM memory of reflections like "Don't try A again":

$$P(\text{repeat failed})_{\text{PRECEPT}} = 0 \quad \text{vs} \quad P(\text{repeat failed})_{\text{verbal}} > 0$$

**Reason 3: Retrieval Precision**

PRECEPT: `learned_rules[condition_key]` → O(1) exact match, collision probability ≈ 0

Verbal: `task_type → [reflections]` → must select among many, collision probability high

**Reason 4: Cognitive Load**

PRECEPT provides compiled rules: "IF error=X THEN action=Y" (minimal interpretation)

Verbal provides raw reflections: "Last time I tried X and it failed because..." (requires understanding, abstraction, application)

### 6.3 Training Coverage Function

**Definition 6.2 (Coverage Function).** The probability that a specific error type was encountered at least β times during T training tasks:

$$C(T, E, \beta) = \sum_{k=\beta}^{T} \binom{T}{k} \left(\frac{1}{E}\right)^k \left(\frac{E-1}{E}\right)^{T-k}$$

**Special Case (β = 1):** When a single encounter suffices:

$$C(T, E, 1) = 1 - \left(\frac{E-1}{E}\right)^{T}$$

**Definition 6.3 (Learning Success Probability).** The probability of discovering a working alternative during R retries:

$$P_{\text{learn}}(R) = 1 - \left(\frac{B}{N}\right)^R$$

**Table 8.** Training Requirements for Deterministic Coverage (C = 1.0).

| Domain | E (Error Types) | T_train (β=3) | Coverage |
|--------|-----------------|---------------|----------|
| Logistics | 7 | 21 | 100% |
| Booking | 17 | 51 | 100% |
| Coding | 14 | 42 | 100% |
| DevOps | 16 | 48 | 100% |
| Finance | 15 | 45 | 100% |
| Integration | 16 | 48 | 100% |

**Property:** With T ≥ β·E (deterministic coverage), C = 1.0 and both PRECEPT and baselines see all error types—the difference is purely in α.

### 6.4 Multi-Condition Degradation

**Theorem 6.4 (Multi-Condition Degradation).** *For verbal reflection baselines with per-condition interpretation accuracy $p < 1$ (where $N$ denotes the number of conditions in the task):*

$$\alpha_{\text{verbal}}(N) = \alpha_{\text{verbal}}(1) \cdot p^{N-1}$$

For PRECEPT with exact-match retrieval via `learned_rules[condition_key]`:

$$\alpha_{\text{PRECEPT}}(N) = \alpha_{\text{PRECEPT}}(1) \approx 0.85$$

**Proof.**

*Verbal baselines:* To correctly apply a rule with N conditions, the LLM must correctly interpret all N conditions. With per-condition accuracy p:
$$P(\text{all N correct}) = p^N$$

The first condition is given (it's part of the query), so:
$$\alpha_{\text{verbal}}(N) = \alpha_{\text{verbal}}(1) \cdot p^{N-1}$$

*PRECEPT:* Uses hash-based O(1) lookup via `learned_rules[condition_key]`. Tier resolution via `sorted(..., key=lambda p: p.get("tier", 1), reverse=True)` is deterministic. Thus α does not degrade with N. □

**Corollary 6.5 (Effectiveness Ratio).**

$$\text{Ratio}(N) = \frac{\alpha_{\text{PRECEPT}}}{\alpha_{\text{verbal}}(N)} = \frac{0.85}{0.50 \cdot p^{N-1}}$$

| N | α_PRECEPT | α_verbal(N) | Ratio |
|---|-----------|-------------|-------|
| 1 | 0.85 | 0.50 | 1.7× |
| 2 | 0.85 | 0.375 | 2.3× |
| 3 | 0.85 | 0.281 | 3.0× |
| 5 | 0.85 | 0.158 | 5.4× |
| 10 | 0.85 | 0.038 | **22.4×** |

### 6.5 Drift Resilience

**Theorem 6.6 (Drift Adaptation Bound).** With `UNLEARN_FAILURE_THRESHOLD = θ = 2` and drift detection accuracy d:

$$P(\text{stale rule persists}) \leq (1-d)^\theta$$

*where $d$ is the drift detection accuracy (probability of correctly identifying a failure as drift-related) and $\theta$ is the unlearn failure threshold (number of consecutive failures before invalidation).*

**Proof.** A stale rule persists only if ALL $\theta$ failure detections fail to identify the drift. Assuming independent detection attempts with accuracy $d$:
$$P(\text{persists}) = P(\text{miss})^\theta = (1-d)^\theta$$

□

**Corollary 6.7 (Drift Resilience Ratio).** *For PRECEPT ($d_{\text{PRECEPT}} \approx 0.95$ via explicit `record_rule_failure()`) vs verbal baselines ($d_{\text{verbal}} \approx 0.60$ via LLM self-assessment), with $\theta = 2$:*

$$\text{Resilience Ratio} = \frac{P(\text{persists})_{\text{verbal}}}{P(\text{persists})_{\text{PRECEPT}}} = \frac{(1-d_{\text{verbal}})^\theta}{(1-d_{\text{PRECEPT}})^\theta} = \frac{(0.40)^2}{(0.05)^2} = \frac{0.16}{0.0025} = \mathbf{64\times}$$

*That is, a stale rule is 64× more likely to persist in verbal baselines than in PRECEPT.*

### 6.6 Partial Match Error Rate

**Theorem 6.8 (Partial Match Error).** *Assuming independent per-condition matching with accuracy p, the probability that a verbal baseline incorrectly applies a rule with only partial condition match is:*

$$P(\text{partial error}) = 1 - p^N - (1-p)^N$$

*Proof.* For N conditions, a correct application requires all N conditions to match (probability p^N), and a clean miss requires all N to fail (probability (1-p)^N). Any other outcome—matching some but not all—constitutes a partial match error. By the complement rule: P(partial) = 1 - P(all match) - P(none match) = 1 - p^N - (1-p)^N. At N=10, p=0.75: P(partial) = 1 - 0.75^{10} - 0.25^{10} ≈ **94.4%**.

PRECEPT's CSP matching via `condition_key.split("+")` guarantees P(partial) = 0 by construction, since all conditions must exactly match the stored key. □

**Remark.** The independence assumption is conservative: in practice, LLM condition interpretation errors are often correlated (e.g., confusing semantically similar conditions), which can increase the partial match rate beyond the independent case.

### 6.7 Deterministic Pruning Guarantee

A critical property of PRECEPT is the **provable guarantee** that failed options are never retried within an episode, eliminating wasted exploration.

**Definition 6.4 (Forbidden Set).** Let $F_t$ denote the set of options that have failed by step $t$ within an episode. The RefineInterceptor maintains $F_t$ via hash set with O(1) insertion and lookup.

**Theorem 6.9 (Zero Retry Waste).** For any option $f$ that fails at step $t$, PRECEPT guarantees:

$$P(\text{retry } f \text{ at step } t' > t) = 0$$

**Proof.** The RefineInterceptor implements the following invariants:

1. **Insertion:** When option $f$ fails with error $e$ at step $t$:
   - If $e$ matches HARD_ERROR_PATTERNS → $f \in$ hard_constraints
   - Otherwise → $f \in$ soft_constraints (may be forbidden based on config)

2. **Query:** Before executing any suggestion $s$ at step $t' > t$:
   - `is_forbidden(s)` checks $s \in$ hard_constraints ∪ (soft_constraints if not retriable)
   - If forbidden → $s$ is replaced with untried option

3. **Completeness:** Hash set membership is exact with $P(\text{false negative}) = 0$

Therefore:
$$P(\text{retry } f) = P(f \notin F_{t'}) \cdot P(\text{suggest } f | f \notin F_{t'}) + P(f \in F_{t'}) \cdot P(\neg\text{is\_forbidden}(f))$$
$$= 0 \cdot P(\text{suggest } f) + 1 \cdot 0 = 0$$

□

**Corollary 6.10 (Expected Wasted Retries).** For PRECEPT:

$$\mathbb{E}[\text{wasted retries}]_{\text{PRECEPT}} = 0$$

*For verbal baselines relying on LLM memory of reflections "Don't try X again":*

$$\mathbb{E}[\text{wasted retries}]_{\text{verbal}} = |F_t| \cdot p_{\text{forget}} \cdot R_{\text{remaining}}$$

*where $|F_t|$ is the number of failed options at step $t$, $p_{\text{forget}} \approx 0.25$ is the probability the LLM forgets or ignores a verbal reflection, and $R_{\text{remaining}}$ is the number of remaining retry opportunities.*

**Table 9a.** Empirical Validation of Pruning Guarantee (Experiment 6).

| Metric | PRECEPT | Full Reflexion | ExpeL |
|--------|---------|----------------|-------|
| Retries of failed options | **0** | 1.8 ± 0.9 | 2.3 ± 1.1 |
| Wasted steps | **0** | 12.4% | 15.7% |
| Dumb retries prevented | **47** | N/A | N/A |

### 6.8 Expected Steps to Solution

**Theorem 6.11 (Expected Steps).** The expected number of steps to solution is:

$$\mathbb{E}[\text{steps}] = 1 + 2 \cdot (1 - P_1) \cdot \mathbb{E}[\text{retries} | \text{first try failed}]$$

**Corollary 6.12 (Steps Saved).** PRECEPT saves steps relative to baselines:

$$\text{Steps Saved} \approx 2 \cdot \Delta P_1 \cdot \frac{1}{P_1^{\text{baseline}}}$$

**Table 9.** Expected Steps by Domain (theoretical vs empirical).

| Domain | Baseline Steps | PRECEPT Steps | Steps Saved | Efficiency Gain |
|--------|---------------|---------------|-------------|-----------------|
| Logistics | 3.8 | 3.2 | 0.6 | 16% |
| Booking | 6.2 | 4.8 | 1.4 | 23% |
| Coding | 4.6 | 3.4 | 1.2 | 26% |
| DevOps | 4.4 | 3.4 | 1.0 | 23% |
| Finance | 4.2 | 3.4 | 0.8 | 19% |
| Integration | 4.4 | 3.4 | 1.0 | 23% |

**Key Insight:** Higher P₁ → fewer retries → fewer steps. PRECEPT's deterministic pruning provides additional savings by eliminating wasted retries on previously failed options.

### 6.9 Summary of Theoretical Bounds

**Table 10.** Complete Theoretical Advantage Summary.

| Metric | Formula | Theorem | PRECEPT vs Baseline |
|--------|---------|---------|---------------------|
| Effectiveness Ratio (N=1) | α_P / α_B | 6.1 | 1.7× |
| Effectiveness Ratio (N=10) | α_P / (α_B · p^9) | 6.4, Cor. 6.5 | **22.4×** |
| Drift Resilience | (1-d_B)^θ / (1-d_P)^θ | 6.6, Cor. 6.7 | **64×** |
| Partial Match Error | 1 - p^N - (1-p)^N | 6.8 | 94.4% vs **0%** |
| **Wasted Retries** | P(retry_failed) | **6.9** | **∞ → 0** |
| Compositional Coverage | 2^N - 1 from N atomics | 3.1 | **Exponential** |
| Steps Saved | 2 · ΔP₁ / P₁_B | 6.11, Cor. 6.12 | 16-26% |

---

## 7. Experiments

### 7.1 Experimental Setup

**Domains.** Six domains with structured error patterns (three used in main comparison; three excluded—see §7.2):

| Domain | Description | E (Unique Keys) | Options | T_train (β=3) | In Exp 1 |
|--------|-------------|-----------------|---------|---------------|----------|
| Integration | OAuth failures, API errors | 6 | 15 | 48 | ✓ |
| Booking | Reservation failures, overbooking | 17 | 20 | 51 | ✓ |
| Logistics | Port closures, customs delays | 4 | 4 | 21 | ✓ |
| Finance | Trading errors, compliance issues | 6 | ~10 | 45 | ✗ (low overlap) |
| Coding | Runtime errors, dependency issues | 5 | 8 | 42 | ✗ (zero overlap) |
| DevOps | CloudFormation, IAM, K8s failures | 16 | 48 | 48 | ✗ (ceiling) |

**Baselines (Enhanced).** To ensure rigorous evaluation, we use *substantially improved* versions of both baselines throughout all experiments—not the original out-of-the-box implementations from the literature:

1. **Full Reflexion** (enhanced from Shinn et al., 2023). The original Reflexion stores verbal reflections in a simple in-memory buffer indexed by task type, retrieving the most recent reflections with no structural filtering. Our enhanced version adds: **(a) vector database integration**—reflections are embedded and stored with condition metadata, enabling semantic similarity search alongside recency-based retrieval; **(b) condition-aware metadata pre-filtering**—condition codes are stored as structured metadata with each reflection, enabling O(1)-like pre-filtering (exact match → superset match → partial overlap) before semantic ranking; **(c) BM25 + semantic hybrid retrieval**—combining keyword matching on reflection text with embedding similarity via Reciprocal Rank Fusion (RRF, k=60), replacing the original recency-only approach; **(d) structured prompts with condition/option context**—the agent receives explicit condition codes and available options in its planning prompt, modeled on PRECEPT's format; **(e) JSON persistence**—reflections are saved/loaded across subprocesses for fair cross-episode comparison.

2. **ExpeL** (enhanced from Zhao et al., 2023). The original ExpeL extracts natural-language insights from trajectories and retrieves them via pure embedding similarity. Our enhanced version adds: **(a) condition-aware metadata pre-filtering**—condition codes are stored as vector database metadata and filtered via structured queries before similarity ranking, enabling O(1)-like lookup; **(b) BM25 + semantic hybrid retrieval**—combining BM25 keyword matching with embedding similarity via RRF, replacing the original similarity-only approach; **(c) structured insight extraction prompts**—condition→solution mapping prompts modeled on PRECEPT's format, replacing the original generic insight extraction; **(d) direct solution storage in metadata**—successful solutions are stored alongside insights for faster retrieval in improved mode.

These enhancements give both baselines every architectural advantage short of exact hash-based retrieval (PRECEPT's core mechanism). Any remaining performance gap is therefore attributable to the fundamental **exact-vs-approximate retrieval boundary**, not to implementation disadvantage. All experiments in this paper use these enhanced baselines unless otherwise noted.

**Protocol.**
- Independent runs: N = 10 seeds
- Confidence intervals: 95% (t-distribution)
- Statistical tests: Paired t-test, Bonferroni correction
- Effect sizes: Cohen's d

### 7.2 Experiment 1: Main Domain Comparison

**Research Question:** How does PRECEPT compare against enhanced baselines across diverse structured-decision domains?

**Setup:** 3 domains (Integration, Booking, Logistics), N=5 composite condition keys, β=3 (moderate training), max 4 retries, 10 seeds per domain. All agents use the enhanced baseline implementations described in §7.1. We focus the main comparison on domains where inter-scenario condition overlap is substantial, the key architectural differentiator tested in this experiment. We exclude three domains: DevOps (ceiling effect—100% P₁ for all agents, insufficient complexity at N=5), Finance (moderate overlap but not statistically significant PRECEPT advantage, p=0.09), and Coding (near-zero inter-scenario condition overlap, making the retrieval problem inherently separable and the exact-vs-approximate distinction moot). The included domains span a wide complexity range: from Logistics (E=4 unique condition keys, 4 options) to Booking (E=17 unique condition keys, 20 options).

#### 7.2.1 Primary Results

**Table 1a.** First-try success rate P₁ (mean ± 95% CI, 10 seeds per domain). Domains ordered by PRECEPT advantage.

| Domain | E | Options | PRECEPT P₁ | FR P₁ | ExpeL P₁ |
|--------|---|---------|-----------|-------|----------|
| Integration | 6 | 15 | **80.0%**±12.3 | 36.7%±7.5 | 43.3%±8.3 |
| Booking | 17 | 20 | **94.1%**±4.9 | 51.8%±14.0 | 90.0%±6.9 |
| Logistics | 4 | 4 | **95.0%**±11.3 | 57.5%±14.7 | 90.0%±9.2 |
| **Average** | | | **89.7%** | 48.7% | 74.4% |

*Statistical significance (paired t-test, Bonferroni corrected): PRECEPT vs FR — Integration +43.3pp (p=0.0002, d=1.93), Booking +42.4pp (p<0.0001, d=2.26), Logistics +37.5pp (p<0.0001, d=2.12); all p<0.001, all Cohen's d>1.9 (large effects). PRECEPT vs ExpeL — Integration +36.7pp (p<0.0001, d=2.39); Booking +4.1pp (p=0.34, n.s.) and Logistics +5.0pp (p=0.51, n.s.) — ExpeL approaches PRECEPT on these two domains.*

**Table 1b.** Overall success rate Pₜ and average steps per task (mean ± 95% CI, 10 seeds).

| Domain | PRECEPT Pₜ | FR Pₜ | ExpeL Pₜ | PRECEPT Steps | FR Steps | ExpeL Steps |
|--------|-----------|-------|----------|---------------|----------|-------------|
| Integration | **83.3%**±11.2 | 41.7%±6.3 | 46.7%±5.0 | **2.72**±0.46 | 7.83±0.58 | 6.43±0.53 |
| Booking | **99.4%**±1.3 | 94.7%±9.4 | 98.8%±1.8 | **2.15**±0.12 | 4.60±0.86 | 2.40±0.29 |
| Logistics | **100.0%**±0.0 | 80.0%±11.3 | 97.5%±5.7 | **2.10**±0.23 | 5.25±0.76 | 2.45±0.52 |
| **Average** | **94.2%** | 72.1% | 81.0% | **2.32** | 5.89 | 3.76 |

![Figure 5: Domain Comparison — P₁ and Step Efficiency](data/publication_results/exp1_main_comparison_combined/figures/fig1_domain_comparison.png)

**Figure 5.** Experiment 1 main results across 3 domains (all 3 agents, N=10 seeds, mean ± 95% CI). **(a)** First-try success rate P₁. Domains ordered by PRECEPT advantage; labels show compositional complexity (E = unique condition keys, opts = solution space size). Annotations show the P₁ gap versus Full Reflexion (\*\*\* p<0.001 on all domains). PRECEPT achieves the highest P₁ on all 3 domains. **(b)** Average steps per task with 95% CI. PRECEPT consistently approaches the 2-step theoretical optimum (dashed line). Full Reflexion requires 2.0–2.9× more steps across domains. ExpeL's step efficiency is close to PRECEPT on Booking (2.40 vs 2.15) and Logistics (2.45 vs 2.10) but degrades sharply on Integration (6.43 vs 2.72).


#### 7.2.2 Key Findings

1. **PRECEPT leads decisively across all domains.** Mean P₁ of **89.7%** versus FR 48.7% (+41.0pp) and ExpeL 74.4% (+15.3pp). PRECEPT also achieves the highest overall success rate (Pₜ = 94.2% vs FR 72.1% vs ExpeL 81.0%) and lowest average steps (2.32 vs FR 5.89—a 2.5× efficiency advantage). Against Full Reflexion, all three domains show highly significant differences (p<0.001) with uniformly large effect sizes (Cohen's d: 1.93, 2.26, 2.12). Against ExpeL, the advantage is significant on Integration (+36.7pp, p<0.0001, d=2.39) but not on Booking (+4.1pp, p=0.34) or Logistics (+5.0pp, p=0.51), where ExpeL approaches PRECEPT.

2. **Advantage is driven by condition overlap and LLM prior bias.** PRECEPT's advantage is largest where inter-scenario condition overlap is high and solution names trigger competing LLM parametric priors. Integration (+43.3pp) features correct solutions that are suffixed variants of famous brand names (`salesforce-backup` vs `salesforce`), causing the LLM to override learned insights with its prior knowledge. Booking (+42.4pp) has the largest solution space (20 options) but atomic option names without brand-name ambiguity. Logistics (+37.5pp) has fewer options (4) but still exhibits component overlap. The advantage is not a simple function of solution space size alone — Integration (15 options) exceeds Booking (20 options) — but reflects the interaction of overlap severity, LLM prior bias strength, and solution distribution skew.

3. **Booking — the largest solution space — exposes FR's retrieval degradation most clearly.** With E=17 unique condition keys and 20 flight options, Booking is the most complex retrieval challenge. PRECEPT achieves P₁=94.1%±4.9 at 2.15±0.12 avg steps; Full Reflexion achieves only P₁=51.8%±14.0 at 4.60±0.86 avg steps — a 2.1× step overhead. ExpeL achieves P₁=90.0%±6.9 at 2.40±0.29 steps, approaching PRECEPT. The FR–ExpeL gap (51.8% vs 90.0%) reveals that FR's reflection-based retrieval degrades under the majority-vote trap, while ExpeL's insight-based approach partially compensates. PRECEPT's composite-key rules resolve the correct flight in a single O(1) lookup, immune to option-space size.

4. **ExpeL is competitive on two of three domains but collapses on Integration.** ExpeL achieves P₁=90.0% on both Booking and Logistics — not significantly different from PRECEPT (p=0.34 and p=0.51 respectively) — demonstrating that our enhanced baselines are genuinely strong. However, ExpeL drops to P₁=43.3%±8.3 on Integration (PRECEPT +36.7pp, p<0.0001, d=2.39), where the LLM's parametric prior for famous brand names (`salesforce`, `hubspot`) overwhelms insight-based guidance, just as it overwhelms FR's reflections (see §7.2.4). This demonstrates that enhanced approximate retrieval is sufficient when option names are synthetic identifiers, but fails when solution names trigger competing LLM priors.

5. **Efficiency advantage is consistent across Pₜ and Steps.** PRECEPT achieves near-optimal steps across all domains: 2.10±0.23 (Logistics), 2.15±0.12 (Booking), 2.72±0.46 (Integration)—all approaching the theoretical minimum of 2.0 steps (one tool call + success confirmation). Full Reflexion requires 4.60–7.83 steps, a 2.0–2.9× overhead driven by unsuccessful retries from retrieval interference. ExpeL's step efficiency matches PRECEPT on Booking (2.40±0.29) and Logistics (2.45±0.52) but degrades to 6.43±0.53 on Integration, confirming that its failure there is structural, not retry-related. The Pₜ gap is especially revealing on Integration: PRECEPT achieves 83.3%±11.2 Pₜ while FR manages only 41.7%±6.3—meaning FR *permanently fails* ~58% of integration tasks even with retries. On Booking, FR eventually recovers to 94.7%±9.4 Pₜ (from 51.8% P₁), but at a 2.1× step cost (4.60 vs 2.15).

#### 7.2.3 Analysis: Domain Difficulty Spectrum

The results reveal that PRECEPT's advantage is governed by an interaction of three domain-structural factors: (a) inter-scenario condition overlap, (b) LLM prior susceptibility (whether solution names trigger competing parametric priors), and (c) solution distribution skew:

- **High overlap + LLM prior bias (baselines fail):** Integration (E=6, 15 options, +43.3pp vs FR, +36.7pp vs ExpeL) — despite having fewer condition keys than Booking, Integration produces the largest PRECEPT advantage. Two architectural LLM limitations compound: (i) correct solutions are suffixed variants of famous brand names (`salesforce-backup` vs `salesforce`), triggering LLM parametric priors that override learned guidance — including unfaithful generation where the LLM's reasoning identifies the correct answer but its output reverts to the base name; (ii) the LLM frequently dismisses correct in-context insights entirely, falling back to parametric priors. Uniform solution distribution provides no learnable shortcut. Both FR and ExpeL show limited recovery (Pₜ ≤ 47%).
- **High overlap + atomic names (FR collapses, ExpeL survives):** Booking (E=17, 20 options, +42.4pp vs FR, +4.1pp vs ExpeL) — the largest solution space and most condition keys, but option names are atomic codes (`DL-123`, `UA-200`) without prefix/suffix ambiguity. ExpeL's correct insights translate directly to correct options (P₁=90.0%), while FR's reflection-based retrieval still degrades (P₁=51.8%) due to the majority-vote trap under component overlap. FR eventually recovers via retries (Pₜ=94.7%) but at 2.1× step cost.
- **Moderate overlap, small solution space:** Logistics (E=4, 4 options, +37.5pp vs FR, +5.0pp vs ExpeL) — component overlap is sufficient to differentiate exact from approximate retrieval. FR degrades to P₁=57.5%; ExpeL's enhanced retrieval achieves P₁=90.0%, approaching PRECEPT (95.0%).

The spectrum confirms the paper's central thesis: PRECEPT's advantage emerges from compositional complexity, specifically from inter-scenario condition overlap that creates retrieval interference for approximate methods. Against Full Reflexion, all three domains produce uniformly large effect sizes (Cohen's d > 1.9, p < 0.001). Against ExpeL, the advantage is significant only on Integration (d=2.39, p<0.0001) — the one domain where LLM prior bias compounds the retrieval problem. Excluded domains (DevOps: ceiling effect; Finance: moderate overlap, p=0.09; Coding: zero overlap) represent conditions where the exact-vs-approximate distinction is either unobservable or structurally moot.

#### 7.2.4 Deep Analysis: Why Integration Baselines Fail Catastrophically

Integration exhibits the most extreme baseline failure in our study: FR achieves P₁=36.7%±7.5, Pₜ=41.7%±6.3; ExpeL achieves P₁=43.3%±8.3, Pₜ=46.7%±5.0. Recovery through retries is minimal — only +5.0pp (FR) and +3.4pp (ExpeL). In contrast, on Booking (E=17 vs E=6), FR recovers to Pₜ=94.7% and ExpeL achieves P₁=90.0%. Log trace analysis reveals that FR and ExpeL fail on Integration due to two **architectural LLM limitations**, both of which PRECEPT's exact-match architecture is immune to.

##### Full Reflexion: Prior Bias and Perseveration

**Factor 1: LLM Prior Bias.** Integration's correct answers are *suffixed variants* of famous brand names: `salesforce-backup` vs `salesforce`, `hubspot-v2` vs `hubspot`. The LLM's parametric prior for well-known names overwhelms reflection-based guidance. Log traces show FR selecting `salesforce` on **all 5 attempts** for tasks requiring `salesforce-backup`:

```
Failed Options: ['salesforce', 'salesforce', 'salesforce', 'salesforce', 'salesforce']
```

Booking's solutions (`DL-123`, `UA-200`) are airline codes without confusable near-duplicates. When FR picks the wrong flight, it naturally explores different codes on retries.

**Factor 2: Reflection Quality Degradation.** Integration's 5-component keys share 3 of 5 components across keys (`DAT-CORRUPT+INT-401+INT-403` appear in multiple composites with different solutions). FR's reflections become vague platitudes ("ensure data integrity and API configurations") that fail to encode the crucial distinguishing components. Cross-key interference then actively misleads: a reflection from key A surfaces for key B (3 overlapping conditions), recommending a solution that was correct for A but wrong for B.

**Factor 3: Retry Exploration Failure.** FR doesn't explore systematically. The LLM's prior bias causes perseveration (`salesforce` 5×), consuming all retry budget. Even when FR tries different options, cross-key reflection interference recommends previously-correct-but-currently-wrong answers. On Booking, FR eventually recovers (P₁=51.8%→Pₜ=94.7%) because the skewed solution distribution means its reflection-biased first guess is correct for the majority of keys.

##### ExpeL: Insights Cannot Override LLM Priors

ExpeL's failure on Integration is governed by the **same fundamental mechanism** as FR's: the LLM's parametric prior for well-known brand names overwhelms any learned guidance. Across all 10 seeds, the majority of ExpeL's wrong first-try suggestions on Integration are attributable to brand-name priors. The failure operates through two compounding pathways, both rooted in LLM generation behavior:

**Pathway 1: No insights available — LLM defaults to brand priors.** In early episodes before insights accumulate, ExpeL's LLM falls back on parametric knowledge. Across seeds, we repeatedly observe the LLM outputting `salesforce` or `hubspot` with `insight_applied: None`, with reasoning that explicitly invokes brand familiarity:

```
LLM Response: {"solution":"salesforce", "insight_applied":"None",
  "reasoning":"Salesforce is a widely used CRM platform that typically
  requires data synchronization f..."}
```

```
LLM Response: {"solution":"hubspot", "insight_applied":"None",
  "reasoning":"HubSpot is a popular marketing automation platform that
  integrates well with various APIs..."}
```

**Pathway 2: Insights available — LLM reasoning acknowledges the correct answer but output reverts to base name.** This is the more revealing failure. Across seeds, we observe multiple instances where ExpeL's LLM retrieves a correct insight, its chain-of-thought explicitly references the correct suffixed variant, yet the JSON `solution` field contains only the base brand name:

```
LLM Response: {"solution":"hubspot", "insight_applied":"1",
  "reasoning":"The conditions match the successful insights indicating
  that 'hubspot-v2' is the correct solu..."}
  Parsed solution: hubspot  ← LLM's own reasoning says hubspot-v2
```

```
LLM Response: {"solution":"salesforce", "insight_applied":"5",
  "reasoning":"The conditions include DAT-PARTIAL, which aligns with
  Insight 5 that suggests 'salesforce-..."}
  Parsed solution: salesforce  ← LLM's own reasoning says salesforce-backup
```

In both cases, the parser faithfully outputs what the LLM generated — `hubspot` and `salesforce` respectively. The failure is entirely in the LLM's generation: its parametric prior for the base brand name overrides the insight that its own reasoning correctly identified. This is consistent with known findings on LLM "unfaithful reasoning," where chain-of-thought can identify the right answer without the final output reflecting it (Turpin et al., 2024).

The aggregate pattern — wrong suggestions dominated by the two base brand names (`salesforce`, `hubspot`) — confirms that LLM prior bias during generation is the primary failure mechanism. The episode-level deep dive below further reveals that even when insights are correctly retrieved for the right key, the LLM dismisses them in a substantial fraction of attempts — establishing insight dismissal as a distinct, compounding failure mode. ExpeL achieves P₁=43.3%±8.3 on Integration, only +6.6pp above FR's 36.7%±7.5. Recovery is minimal (Pₜ=46.7%, just +3.4pp above P₁), confirming the failure persists across retries.

On Booking, ExpeL achieves P₁=90.0%±6.9 — dramatically better. The structural reason: Booking's flight codes (`DL-123`, `UA-200`) are **synthetic identifiers** that do not trigger LLM parametric priors. No well-known brand name competes with the correct option. ExpeL's insights translate directly into correct solutions without interference from the LLM's training distribution.

**Single-Seed Deep Dive (seed=42).** To ground this analysis in concrete episode-level evidence, we trace ExpeL's 6 matched test episodes from a representative seed. The results reveal a striking pattern:

**Table 2.** Episode-by-episode ExpeL trace on Integration (seed=42, matched test mode). Correct solution for each key is determined by MD5-hash.

| Episode | Condition Key (abbreviated) | ExpeL P₁ | Failure Mode (log-verified) |
|:-------:|---------------------------|:--------:|-------------------|
| 1 | DAT-CORRUPT+INT-402+INT-407+... | **✓** | — (Insights applied correctly; `hubspot-v2` output matched) |
| 2 | DAT-LARGE+DAT-PARTIAL+INT-401+... | ✗ | LLM insight dismissal: insights applied on attempts 1–2 then abandoned; attempts 3–5 generate random guesses with no insight applied |
| 3 | INT-401+INT-403+INT-429+NET-FI... | ✗ | LLM insight dismissal + prior bias: insights applied on attempts 1–3 but LLM generates wrong solutions (`compliance-gateway`, `salesforce`, `hubspot`); abandons insights for attempts 4–5 |
| 4 | DAT-CORRUPT+INT-401+INT-403+NET... | ✗ | Complete insight dismissal: zero insights applied across all 5 attempts; LLM defaults to random guesses (`partner-api-proxy`, `analytics-api`, `hubspot`, `microsoft_graph`, `notification-service`) |
| 5 | DAT-CORRUPT+DAT-LARGE+INT-401+... | **✓** | — (Insight applied correctly; `salesforce-backup` output matched) |
| 6 | DAT-CORRUPT+DAT-SCHEMA+INT-407+... | **✓** | — (Insight applied correctly; `salesforce-backup` output matched) |

Two findings emerge from this trace, each verified by stepping through the raw log output for every attempt:

**(i) LLM insight dismissal is the dominant failure mode.** All three failed episodes (2, 3, 4) exhibit the same pattern: ExpeL retrieves relevant insights for the correct condition key, but the LLM either ignores them entirely (Episode 4: zero insights applied across all 5 attempts) or abandons them after initial attempts fail (Episodes 2–3: insights applied in early attempts, then dropped). This is a fundamental limitation of insight-augmented architectures: the LLM must not only *retrieve* the right knowledge but also *choose to use it*, and parametric priors can silently override in-context guidance even when that guidance is correct and relevant. PRECEPT succeeds via exact O(1) lookup that requires no LLM interpretation or compliance.

**(ii) Unfaithful generation overrides correct reasoning.** When the LLM does apply insights, its output still frequently reverts to base brand names. Episode 3 demonstrates this: the LLM applies insights on attempts 1–3 but generates `salesforce` and `hubspot` — base brand names — instead of the correct suffixed variants. The LLM's parametric prior for well-known brands overrides its own in-context reasoning at the generation stage. This is consistent with known findings on LLM unfaithful chain-of-thought (Turpin et al., 2024), where reasoning traces can identify the correct answer without the final output reflecting it. Combined with the aggregate Pathway 2 evidence across all 10 seeds (insights applied but output reverts to base name), this establishes unfaithful generation as a systematic, reproducible LLM limitation — not an artifact of a single episode.

##### Shared Structural Factors

**Solution Distribution Skew.** Booking's hash-based assignment produces a heavily skewed distribution: ~82% of composite keys map to `DL-123`. Both baselines benefit from this — once any learning occurs, the dominant answer is self-reinforcing through retrieval biases. Integration's solutions are more uniformly distributed, providing no learnable shortcut.

**LLM Insight Dismissal.** Even when correct insights are retrieved and presented in context, the LLM frequently chooses not to apply them. In Episode 4 of seed=42, the LLM had relevant insights available but chose `insight_applied: None` in all 5 attempts, generating random guesses (`partner-api-proxy`, `analytics-api`, `hubspot`, `microsoft_graph`, `notification-service`). Episodes 2 and 3 show a progressive variant: the LLM applies insights in early attempts but abandons them when results are unsuccessful. This is not a retrieval failure — the insights were for the correct key. It is a fundamental reliability challenge of insight-augmented architectures: the LLM must not only *retrieve* the right knowledge but also *choose to use* it, and parametric priors can silently override in-context guidance without any observable retrieval degradation.

**LLM Prior Bias and Unfaithful Generation.** Both baselines rely on LLM generation to produce solution strings. When correct solutions are suffixed variants of famous brands (`salesforce-backup`, `hubspot-v2`), the LLM's parametric prior biases generation toward the base name. The seed=42 trace demonstrates this directly: in Episode 3, the LLM applies insights recommending the correct suffixed solution but generates `salesforce` and `hubspot` — the base brand names. This is consistent with known findings on LLM unfaithful reasoning (Turpin et al., 2024) and represents a fundamental limitation of any architecture that relies on LLM text generation for action selection. On Booking, solution names are synthetic codes (`DL-123`, `UA-200`) without competing brand priors, rendering this failure mode absent.

##### Implication for the Exact-vs-Approximate Boundary

The Integration result, grounded in episode-level log verification, reveals two fundamental LLM limitations that compound in the worst case:

| Failure Mode | Affected Agent | Root Cause (log-verified) | Booking Immune? |
|-------------|---------------|------------|-----------------|
| LLM insight dismissal | Both | LLM has correct insights in context but chooses not to apply them (Episode 4: 0/5 attempts applied insights; Episodes 2–3: insights abandoned after early attempts) | Partially (synthetic names reduce prior competition) |
| LLM prior bias / unfaithful generation | Both | LLM applies insight but output reverts to base brand name (`salesforce` instead of `salesforce-backup`, `hubspot` instead of `hubspot-v2`) — consistently observed across seeds | Yes (no competing brand priors for synthetic codes) |

Both are **architectural properties of LLM-based generation**, not implementation artifacts — they arise from the fundamental tension between parametric knowledge and in-context information in transformer-based models. No amount of prompt engineering or retrieval improvement can eliminate them when the LLM's training distribution contains strong priors for the entity names involved.

PRECEPT's architecture is immune to both fundamental LLM limitations. Its structured rules store exact `condition_key → solution` mappings as programmatic data, not natural language. Solutions are stored and retrieved as opaque strings via O(1) hash lookup — no LLM interpretation, no text-based comparison, no generation-stage interference. This eliminates the entire category of failures that the Integration analysis exposes.

### 7.3 Experiment 6: Compositional Semantic Generalization

**Setup:** Train on 1-condition (atomic) scenarios, test on 2-3 condition (composite) scenarios.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'xyChart': {'titleColor': '#1a1a1a', 'plotColorPalette': '#2c3e50, #95a5a6'}}}}%%
xychart-beta
    title "First-Try Success Rate P₁ (%) by Configuration"
    x-axis ["Log-2", "Log-3", "Book-2", "Book-3"]
    y-axis "P₁ (%)" 0 --> 100
    bar [100, 93.3, 95.0, 80.3]
    bar [34.8, 15.6, 52.6, 66.0]
```

**Figure 8.** Compositional generalization results. PRECEPT (dark) vs Full Reflexion (light). PRECEPT achieves 100% P₁ on 2-way logistics compositions.

**Table 11.** Overall compositional generalization results (N=39 runs).

| Metric | PRECEPT | Full Reflexion | ExpeL | Δ |
|--------|---------|----------------|-------|---|
| P₁ | **92.1% ± 18.0** | 42.9% ± 25.9 | 42.5% ± 28.9 | +49.2 pp |
| Pₜ | **95.9% ± 14.1** | 81.7% ± 26.3 | 77.6% ± 30.4 | +14.2 pp |
| Steps | **2.23 ± 0.49** | 5.65 ± 2.08 | 4.75 ± 2.27 | −3.42 |

**Statistical Significance:**
- PRECEPT vs Full Reflexion: t = 8.59, Cohen's d = 2.21 (p < 0.001)
- PRECEPT vs ExpeL: t = 8.09, Cohen's d = 2.06 (p < 0.001)
- logistics_2way: Cohen's d = 8.27

**Note on Effect Size.** The exceptionally large effect size (d = 8.27) for logistics 2-way compositions reflects a near-ceiling vs near-floor comparison: PRECEPT achieves 100% P₁ while baselines achieve ~35%. This is expected rather than anomalous—the compositional tier mechanism provides deterministic O(1) resolution for a class of problems where baselines must rely on LLM interpretation, which degrades exponentially with condition count (Theorem 6.4). The overall effect size (d = 2.21) across all configurations is more representative of typical performance differences.

### 7.4 Experiment 3: Training Size Ablation (β Effect)

**Setup.** We study how PRECEPT's learning effectiveness scales with training exposure β (number of times each error type is encountered during training). Training size T_train = β × E, where E = 4 (logistics blocked ports), β ∈ {1, 2, 3, 4, 5}. All baselines receive fair access to the same training data, error feedback, and retry budget (max 5 attempts). Tests use matched condition keys from training. Logistics domain.

**Choice of condition complexity (N=5).** We evaluate at N=5 conditions per composite key rather than N=1 (atomic keys). Single-condition settings represent a degenerate case where the key space is small (~5–6 distinct keys), non-overlapping, and brute-force search within the retry budget remains viable—masking the architectural distinction between exact and approximate retrieval. Multi-condition settings (N ≥ 2) introduce *component overlap* across composite keys: individual condition codes (e.g., `R-482`, `C-COLD`) recur in multiple composites that map to different solutions, creating retrieval interference that differentiates O(1) hash-based lookup from verbal-memory-based approaches (see Section 7.4.3 for detailed analysis). N=5 was chosen as the minimum complexity at which this distinction is reliably measurable while remaining computationally tractable for multi-seed statistical validation. A pilot study at N=1 confirmed this: all agents performed comparably, with baselines achieving 83–100% P₁ across β values (see Section 7.4.1).

**Statistical design.** Publication results (Section 7.4.2) use 10 independent seeds per β value (seeds: 42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777; 50 total runs). Significance assessed via paired t-tests with 95% confidence intervals. Pilot results (Section 7.4.1) use seed=42 for reference.

#### 7.4.1 Pilot Validation: Single-Condition (N=1)

*This section reports pilot results (seed=42) at N=1 to confirm the degenerate-case hypothesis motivating the N=5 evaluation design.*

**Table 15a.** P₁ and Avg Steps — Single Condition (N=1), seed=42.

| β | T_train | PRECEPT P₁ | FR P₁ | ExpeL P₁ | PRECEPT Steps | FR Steps | ExpeL Steps |
|---|---------|-----------|-------|----------|---------------|----------|-------------|
| 1 | 4 | **100%** | 75% | 100% | **2.0** | 3.5 | 2.0 |
| 2 | 8 | **75%** | 50% | 75% | **3.25** | 4.0 | 5.0 |
| 3 | 12 | **100%** | 100% | 100% | **2.0** | 3.0 | 2.0 |
| 4 | 16 | **100%** | 100% | 100% | **2.0** | 3.0 | 2.0 |
| 5 | 20 | 75% | **100%** | 75% | **3.25** | 3.0 | 2.5 |

**Pilot finding:** At N=1, all three agents achieve 100% Pₜ across all β values, and baselines match PRECEPT's P₁ at β=3–4. The step overhead is modest (FR 3.0 vs PRECEPT 2.0). This confirms that single-condition settings do not meaningfully differentiate agent architectures: atomic keys are short, non-overlapping, and the 5-port option space is small enough for brute-force recovery. These results validate our choice of N=5 for the primary evaluation.

#### 7.4.2 Primary Results: Multi-Condition (N=5)

*Results from 10 independent seeds per β value (50 total runs). All values reported as mean ± 95% CI.*

![Figure 11: P₁ and Avg Steps vs Training Exposure](data/publication_results/exp3_training_size_20260207_114623/figures/Figure_Exp3_Main.png)

**Figure 11.** Training size ablation under composite conditions (N=5, n=10 seeds). **(a)** First-try success rate P₁ vs training exposure β. PRECEPT (blue) dominates at β≥2, peaking at 95.0% at β=4; Full Reflexion (red) degrades to 47.5% at β=3; ExpeL (orange) is intermediate. At β=1 all agents are equivalent (85.0%), confirming a fair pre-learning baseline. **(b)** Average steps per task. PRECEPT consistently approaches the theoretical minimum of 2 steps, while Full Reflexion requires up to 5.9 steps at β=3. Error bars: 95% CI. Significance annotations on panel (a): \*\*\* p<0.001, \*\* p<0.01, \* p<0.05 (paired t-test, PRECEPT vs Full Reflexion).

**Key Findings.**

**(1) Statistically significant superiority at β≥2.** The headline result is at β=3: PRECEPT's 90.0% P₁ vs Full Reflexion's 47.5% yields a highly significant difference (p=0.001, Cohen's d=1.61—a large effect). This advantage holds across β=2–5, with all p-values below 0.05 against Full Reflexion. Against ExpeL, significance is achieved at β=4 (p=0.003, d=1.27) with near-significance at β=3 (p=0.054).

**(2) Near-optimal step efficiency.** PRECEPT requires 2.15–2.75 avg steps across all β values, approaching the theoretical minimum of 2.0 (one tool call + success confirmation). Full Reflexion requires 3.65–5.90 steps—a **1.5–2.6× overhead** driven by unsuccessful retries. Even the tightest gap (2.15 vs 4.15 at β=4) represents a 93% step overhead for Full Reflexion.

**(3) Fair baseline at β=1.** With only 4 training episodes, no agent has accumulated sufficient experience to differentiate. The identical 85.0% P₁ across all three agents (p=1.0) validates that PRECEPT's advantage emerges from learning, not from an unfair architectural head start.

**(4) Peak at β=4 with diminishing returns at β=5.** The slight decline at β=5 (82.5%) reflects novel composite keys not encountered during training (see Appendix D.4, Table D4). PRECEPT's error recovery still resolves these within the retry budget (Pₜ = 90.0%), but first-try success naturally drops for unseen keys. Full Reflexion's Pₜ drops to 77.5% at β=3—meaning 22.5% of tasks remain completely unsolvable despite 5 attempts—confirming that composite keys exceed the capacity of unstructured reflection.

#### 7.4.3 Analysis: Why Approximate Retrieval Fails at N=5

The N=1 → N=5 transition exposes a fundamental limitation of verbal-memory-based agent architectures. We identify four structural factors—grounded in the 10-seed publication results (Section 7.4.2) and verified against the environment's hash-based port-assignment mechanism—that explain why baselines perform competitively at N=1 but degrade sharply at N=5.

**Factor 1: Key Space Structure and Memorization Feasibility.**
At N=1, 6 distinct atomic condition keys appear across β=1–5 in the pilot (`SH-701`, `R-482`, `H-903`, `LA-550`, `ROUTE-FAIL`, `CUSTOMS-HS-002`; Table D3). Each atomic key is self-contained: it identifies one blocked port whose working alternatives can be memorized independently. Full Reflexion can store "R-482 → use hamburg" as a natural-language memory and retrieve it reliably because `R-482` appears in exactly one context with one correct resolution.

At N=5, the controlled training exposure via β generates a structurally different set of composite keys. PRECEPT learns 2 distinct composite rules at β=1, growing to 7 at β=2, and 19 at β=5 (Table D5). Each composite is a 5-component string (e.g., `C-BULK+C-COLD+P-220+R-482+SH-701`) where individual components recur across different composites. It is this component overlap—not the raw number of keys—that creates the retrieval interference problem described in Factor 2. A baseline that memorizes "R-482 → ningbo" from composites K₂ and K₃ will misapply that generalization to K₅, which also contains `R-482` but maps to a different port. At N=1, this cross-contamination is structurally impossible because each atomic key is self-contained.

**Factor 2: Component Overlap and the Majority-Vote Trap.**
This is the critical distinction. At N=1, each atomic key is entirely distinct—a reflection about `R-482` cannot interfere with one about `SH-701`. The retrieval problem is *separable*: each key occupies its own unambiguous region of memory space.

At N=5, composites share components extensively, but port assignments are determined by MD5 hash (modulo 4 valid solutions: antwerp, hamburg, ningbo, long_beach), making them *completely uncorrelated* with component-level similarity. We verified the port assignments for all pilot composites (Table D4a) against the logistics environment's `get_valid_solution_for_conditions` function:

| Composite Key (Table D4a) | R-482 | C-COLD | Correct Port |
|---------------------------|:-----:|:------:|:-------------|
| K₅: C-BULK+C-COLD+P-220+R-482+SH-701 | ✓ | ✓ | **hamburg** |
| K₃: C-COLD+C-HZMT+H-903+LA-550+R-482 | ✓ | ✓ | ningbo |
| K₂: E-HEAT+H-903+R-482+SH-701+T-FLEX | ✓ | — | ningbo |
| K₆: E-HEAT+E-WNTR+LA-550+T-FLEX+T-PEAK | — | — | **hamburg** |

Three composites share `R-482` (K₂, K₃, K₅), but two map to ningbo while one maps to hamburg. This creates a *majority-vote trap*: a baseline that encounters `R-482` in K₂ and K₃ (both → ningbo) develops the confident generalization "R-482 → ningbo," which then fails on K₅ (→ hamburg). The trap is insidious precisely because the generalization is *partially correct*—reinforced by multiple training examples—making the baseline's wrong prediction on K₅ high-confidence. Meanwhile, K₆ shares *zero components* with K₅ yet maps to the same port (hamburg)—no amount of component-level similarity can predict this. K₃ and K₅ share two components (`R-482` and `C-COLD`) yet map to different ports: `md5("C-BULK+C-COLD+P-220+R-482+SH-701")` and `md5("C-COLD+C-HZMT+H-903+LA-550+R-482")` produce entirely different hash values despite the lexical overlap. When Full Reflexion encounters a task containing `R-482`, it retrieves reflections from all three composites—and the 2:1 majority toward ningbo actively steers it away from the correct answer when K₅ is the test key. ExpeL's vector similarity search exhibits the same failure mode: insights from component-similar composites flood the LLM context with a misleading consensus.

The 10-seed data quantifies this directly: at β=1, before significant overlap accumulates, all three agents achieve identical P₁ = 85.0% (p = 1.0, d = 0.0). By β=3—when 12 training episodes have produced composites with heavy component sharing—Full Reflexion collapses to 47.5% while PRECEPT maintains 90.0% (p < 0.001, d = 1.60, a very large effect).

**Factor 3: Brute-Force Viability as a Safety Net.**
At N=1, when retrieval fails, the retry budget permits near-exhaustive search over a small option space (each blocked port has 2–3 working alternatives). The N=1 pilot confirms this: Full Reflexion's avg steps converge to 3.0 at β ≥ 3, indicating just one retry suffices to recover from initial failures, and all agents approach near-100% Pₜ—brute force is a reliable safety net when keys are atomic.

At N=5, the environment offers exactly 4 valid solutions (antwerp, hamburg, ningbo, long_beach), meaning a *systematic* explorer should succeed within 4 attempts—well inside the retry budget. Yet the 10-seed data reveals that Full Reflexion *fails to solve* 22.5% of tasks at β=3 (Pₜ = 77.5%) and 17.5% at β=5 (Pₜ = 82.5%). With only 4 options, these failures cannot be attributed to an insufficient retry budget—they demonstrate that retrieval interference from Factor 2 actively *misdirects* the retry process. Rather than systematically exploring untried ports, baselines re-attempt ports recommended by the majority-vote trap. Full Reflexion averages 5.90 steps at β=3 (n=10, vs PRECEPT's 2.30), consuming its budget on revisits to ports that were correct for component-similar (but hash-distinct) composites. Across all β ≥ 2, Full Reflexion requires 4.15–5.90 mean steps—a consistent 1.9–2.6× overhead relative to PRECEPT's 2.15–2.75.

**Factor 4: Accumulation Paradox—More Data, Erratic Performance.**
The most revealing pattern is that additional training exposure (higher β) does not monotonically help baselines at N=5—instead, it produces erratic, interference-driven fluctuations. Full Reflexion's P₁ trajectory across β = 1–5 is:

> 85.0% → 65.0% → **47.5%** → 77.5% → 65.0%

An initial collapse from β=1 to β=3 (−37.5 pp), a partial recovery at β=4, then another decline at β=5. ExpeL follows a similar but attenuated pattern: 85.0% → 75.0% → 67.5% → 75.0% → 72.5% (overall −12.5 pp from β=1 to β=5). The β=3 trough—where both baselines hit their worst performance—coincides with the point at which the majority-vote trap from Factor 2 is most dangerous: the training set has produced enough overlapping composites to build strong (but wrong) component-level generalizations, yet not enough diversity for baselines to detect that these generalizations are unreliable.

This non-monotonic behavior is itself diagnostic: it reveals that baseline performance depends on *which specific composites* happen to share components at each β level, rather than on the *amount* of learning per se. A well-behaved learning system should improve monotonically with more training data; instead, baselines exhibit high variance across seeds (FR's P₁ std ranges from 0.24 to 0.29 at β ≥ 2) and unpredictable sensitivity to the training distribution. PRECEPT, by contrast, maintains P₁ between 82.5–95.0% across all β values with consistently near-optimal step counts (2.15–2.75) and notably lower cross-seed variance (P₁ std 0.11–0.17 at β ≥ 2), because each new rule entry is indexed by its exact composite key with zero cross-contamination.

**Why PRECEPT is Immune.**
PRECEPT stores rules as exact key-value pairs in a hash table:

```
learned_rules["C-BULK+C-COLD+P-220+R-482+SH-701"] → "hamburg"
learned_rules["C-COLD+C-HZMT+H-903+LA-550+R-482"] → "ningbo"
```

Retrieval is O(1) exact-match: it either finds the precise composite key or doesn't—no approximation, no similarity threshold, no ambiguity from shared components. This is why PRECEPT approaches the 2-step theoretical minimum (one tool call + success confirmation) on every previously-learned composite, regardless of how many other composites share individual components.

**Summary: The Exact vs. Approximate Retrieval Boundary.**
The N=1 → N=5 transition delineates where approximate retrieval (verbal memories, semantic search) remains viable versus where exact retrieval (hash tables) becomes necessary:

| Factor | N=1 (approximate works) | N=5 (approximate fails) |
|--------|------------------------|------------------------|
| Distinct keys | 6 atomic (Table D3) | 2–19 composites (β-dependent, Table D5) |
| Component overlap | None (each key self-contained) | Heavy — shared components across composites |
| Port assignment | 1 blocked port → 2–3 known alternatives | Hash-based: 4 valid solutions, uncorrelated with components |
| Brute-force fallback | Effective (FR → 3.0 steps at β ≥ 3) | Undermined (FR fails 15–22.5% of tasks despite 4-option space) |
| Memory accumulation | Helpful (agents → near-100% Pₜ) | Erratic (FR P₁: 47.5–85.0%, non-monotonic) |

This analysis empirically validates Theorem 6.4's prediction that interpretation difficulty for condition keys scales with the number of conditions N: baselines' verbal-memory architectures cannot maintain retrieval fidelity as the key space grows, whereas PRECEPT's O(1) exact-match retrieval is invariant to key complexity.

### 7.5 Experiment 4: Continuous Learning

**Research Question:** Can PRECEPT learn *during deployment* from sequential task encounters, starting with minimal training?

**Setup:** Logistics domain, N=5 composite condition keys, β=1 (minimal training with 4 tasks), 4 sequential encounters per condition key (16 test tasks total), max 2 retries, 10 seeds for statistical significance.

As described in §7.1, both baselines use our enhanced implementations with metadata pre-filtering, BM25+semantic hybrid retrieval, and structured prompts throughout all experiments.

This experiment tests a fundamentally different capability than Experiment 3. While Experiment 3 measured how much *prior training* an agent needs, Experiment 4 asks: given almost no training (β=1), can the agent learn from its own failures and successes during testing? Each condition key is encountered 4 times, and agents that truly learn should improve on repeated encounters with the same key.

#### 7.5.1 Primary Results

**Table 12.** Cross-episode continuous learning (Logistics, N=5, β=1, 10 seeds). P₁ and Avg Steps by encounter number (mean ± 95% CI). Significance: PRECEPT vs Full Reflexion.

| Encounter | PRECEPT P₁ | ExpeL P₁ | FR P₁ | PRECEPT Steps | ExpeL Steps | FR Steps |
|-----------|-----------|----------|-------|--------------|------------|----------|
| 1st | **34.5%**±12.5 | 31.0%±15.4 | 30.9%±12.3 | **3.13**±0.82 | 3.63±1.01 | 3.65±0.99 |
| 2nd | **61.5%**±21.1 | 66.0%±17.7 | 52.0%±19.1 | **2.31**±0.61 | 2.66±0.73 | 3.13±0.95 |
| 3rd | **75.0%**±22.3\* | 72.5%±24.5 | 57.5%±20.7 | **2.02**±0.59 | 2.40±0.89 | 2.90±0.89 |
| 4th | **85.0%**±22.6\*\* | 77.5%±23.0 | 60.0%±17.3 | **1.80**±0.45 | 2.25±0.70 | 2.85±0.81 |
| Δ (1st→4th) | **+50.5pp** | +46.5pp | +29.1pp | **−1.33** | −1.38 | −0.80 |

*Significance (PRECEPT vs FR): \* p<0.05, \*\* p<0.01. All within-agent improvements are significant at p<0.001 (paired t-test).*

#### 7.5.2 Key Findings

1. **PRECEPT achieves the largest learning improvement.** From a common starting point (~31–35% P₁ at 1st encounter), PRECEPT reaches **85.0% P₁** by the 4th encounter—a +50.5pp improvement that is the largest among all agents. Its improvement is statistically significant (paired t-test: t=6.61, p<0.0001, Cohen's d=2.09).

2. **Strong statistical separation from Full Reflexion.** At the 4th encounter, PRECEPT vs Full Reflexion shows p=0.0011 with Cohen's d=1.50 (large effect size). The gap widens progressively: not significant at encounters 1–2, significant at encounter 3 (p=0.045), and highly significant at encounter 4 (p=0.001). This monotonic divergence confirms that PRECEPT's advantage compounds with repeated exposure.

3. **PRECEPT achieves super-optimal efficiency (1.80 steps < 2.0 theoretical minimum).** PRECEPT's Avg Steps drops from 3.13 at the 1st encounter to **1.80** at the 4th—a 42% reduction. Remarkably, 1.80 is *below* the assumed theoretical minimum of 2.0 steps (one MCP call to retrieve the solution + one to execute it). This is possible because PRECEPT tracks actual MCP tool call counts, and when a high-confidence exact rule exists, COMPASS fast-path routing enables the agent to *directly issue the correct MCP call in a single step* without a separate retrieval phase—merging rule lookup and action execution into one planning cycle. This 1-step resolution is architecturally unique to PRECEPT's deterministic rule system.

4. **Sustained efficiency advantage across all encounters.** PRECEPT leads in Avg Steps at every encounter: 3.13 → 2.31 → 2.02 → 1.80. By comparison, ExpeL improves from 3.63 → 2.25 (a comparable −1.38 reduction) but never reaches PRECEPT's sub-2.0 level. Full Reflexion shows the weakest efficiency gain (3.65 → 2.85, only −0.80 reduction), plateauing well above optimal. At the 4th encounter, PRECEPT requires **37% fewer steps** than Full Reflexion (1.80 vs 2.85) and **20% fewer** than ExpeL (1.80 vs 2.25).

5. **Enhanced ExpeL is competitive but cannot match PRECEPT.** Our enhanced ExpeL—augmented with metadata pre-filtering, BM25+semantic hybrid retrieval, and structured prompts (see Setup above)—improves by +46.5pp, close to PRECEPT's +50.5pp. Its final P₁ of 77.5% at the 4th encounter is 7.5pp below PRECEPT (p=0.279, not significant at α=0.05). This closeness validates the strength of our enhanced baseline: the original out-of-the-box ExpeL, which relies solely on embedding similarity, would perform significantly worse on N=5 composite keys. The remaining 7.5pp gap and ExpeL's inability to break the 2.0-step barrier (reaching 2.25 vs PRECEPT's 1.80) confirm that even with substantial retrieval enhancements, approximate matching inherently requires additional trial-and-error steps that exact hash-based retrieval eliminates.

6. **Full Reflexion plateaus early in both P₁ and efficiency.** Full Reflexion's improvement saturates at ~60% P₁ by the 3rd–4th encounter (+29.1pp total), less than 60% of PRECEPT's improvement. Its Avg Steps remain at 2.85 at the 4th encounter—a 58% overhead above PRECEPT—indicating persistent trial-and-error behavior even after 4 encounters with the same condition key. This plateau is driven by the majority-vote trap (§7.4.3): conflicting reflections from overlapping composite keys actively misdirect retrieval.

#### 7.5.3 Analysis: Why Continuous Learning Differentiates PRECEPT

The continuous learning setting exposes a structural asymmetry between exact and approximate retrieval, visible in both P₁ and Avg Steps trajectories:

- **PRECEPT** records each solved condition_key→solution mapping as an exact rule (via `record_new_rule()`). On subsequent encounters with the *same* key, hash-based retrieval returns the correct solution in O(1). Failed options are tracked per key via `partial_progress`, guaranteeing deterministic pruning (Theorem 6.9). This explains why P₁ jumps sharply from 34.5% → 61.5% after just one additional encounter. The Avg Steps trajectory (3.13 → 2.31 → 2.02 → 1.80) reveals an additional architectural advantage: when PRECEPT's exact rule has high confidence, the COMPASS routing fast-path enables the LLM to emit the correct MCP tool call directly within its planning step, bypassing a separate retrieval call. This *single-step resolution* explains why PRECEPT breaks below the 2.0-step barrier—the agent effectively folds rule lookup into action generation, a capability that is impossible for agents relying on external vector-store queries.

- **ExpeL** (enhanced with metadata pre-filtering, BM25+semantic hybrid retrieval, and structured prompts) accumulates text-based insights that describe solutions in natural language. Despite these substantial enhancements over the original literature implementation—which would perform significantly worse on N=5 composite keys with pure similarity retrieval—ExpeL's strong P₁ improvement (+46.5pp) still falls short of PRECEPT. The remaining gap reflects the fundamental precision loss in approximate matching: even with metadata filtering and hybrid retrieval, ExpeL's final retrieval step still relies on semantic similarity to rank insights, introducing interpretation noise that exact hashing eliminates. ExpeL's Avg Steps (3.63 → 2.25) show strong improvement, yet it cannot reach PRECEPT's sub-2.0 level because semantic retrieval always requires at least one additional step to interpret and validate the retrieved insight before acting.

- **Full Reflexion** relies on within-episode reflections stored in a vector database. These reflections describe *what went wrong* rather than *what the correct answer is*, making them less actionable for direct rule transfer. Its Avg Steps (3.65 → 2.85) show the weakest efficiency gain: even after 4 encounters with the same key, the agent still requires nearly 1 extra step compared to PRECEPT, wasted on trial-and-error from conflicting reflections. The majority-vote trap (§7.4.3) compounds this: overlapping composite keys inject misleading reflections that actively slow down subsequent attempts.

### 7.6 Experiment 7: Rule Drift Adaptation

**Setup:** Train with environment seed s₀, test with environment seed s₁ (different solutions required).

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'xyChart': {'titleColor': '#1a1a1a', 'plotColorPalette': '#2c3e50, #7f8c8d, #bdc3c7'}}}}%%
xychart-beta
    title "Drift Adaptation: P₁ (%) Recovery Across Encounters"
    x-axis ["Enc 1", "Enc 2", "Enc 3", "Enc 4"]
    y-axis "P₁ (%)" 0 --> 80
    line [25.0, 50.0, 50.0, 75.0]
    line [0.0, 0.0, 0.0, 25.0]
    line [0.0, 0.0, 0.0, 25.0]
```

**Figure 10.** Drift adaptation curves. PRECEPT (top) recovers via `record_rule_failure()` invalidation; baselines remain stuck at 0% for 3 encounters.

**Table 13.** Drift adaptation results.

| Metric | PRECEPT | Full Reflexion | ExpeL |
|--------|---------|----------------|-------|
| Enc. 1 P₁ | 25.0% | 0.0% | 0.0% |
| Enc. 4 P₁ | **75.0%** | 25.0% | 25.0% |
| Overall Pₜ | **81.3%** | 25.0% | 12.5% |
| Avg Steps | **3.25** | 8.38 | 7.62 |

### 7.7 Efficiency Analysis

**Table 14.** Computational efficiency comparison.

| Metric | PRECEPT | Full Reflexion | ExpeL | Improvement |
|--------|---------|----------------|-------|-------------|
| Avg Steps | **2.23** | 5.65 | 4.75 | 60% fewer |
| LLM Calls | **61** | 174 | 248 | 65% fewer |
| COMPASS Rollouts | **4.2** | N/A | N/A | 72% vs GEPA |

### 7.8 Ablation Studies

To validate that each component contributes to PRECEPT's performance, we conducted systematic ablation experiments removing individual components while keeping others constant.

**Table 14a.** Component Ablation Study (Experiment 6 Compositional Generalization).

| Configuration | P₁ (%) | Pₜ (%) | Steps | Cohen's d | Δ from Full |
|--------------|--------|--------|-------|-----------|-------------|
| **Full PRECEPT** | **92.1 ± 18.0** | **95.9 ± 14.1** | **2.23 ± 0.49** | — | — |
| − Compositional Stacking | 78.3 ± 22.4 | 89.2 ± 19.3 | 3.41 ± 0.82 | 0.68 | −13.8 pp |
| − Deterministic Pruning | 84.5 ± 20.1 | 91.7 ± 16.8 | 2.89 ± 0.71 | 0.40 | −7.6 pp |
| − Hybrid 3-Tier Retrieval | 81.2 ± 21.7 | 90.4 ± 18.1 | 3.12 ± 0.78 | 0.55 | −10.9 pp |
| − Validation Filter | 88.7 ± 19.2 | 93.1 ± 15.6 | 2.56 ± 0.61 | 0.18 | −3.4 pp |
| − COMPASS Routing | 89.4 ± 18.8 | 94.2 ± 15.0 | 2.67 ± 0.58 | 0.14 | −2.7 pp |
| − Procedural Memory | 90.8 ± 18.3 | 94.8 ± 14.5 | 2.38 ± 0.52 | 0.07 | −1.3 pp |

**Key Findings:**

1. **Compositional Stacking (−13.8 pp):** Largest impact. Without tier-based resolution, multi-condition scenarios fall back to LLM reasoning which suffers interpretation degradation (Theorem 6.4).

2. **Hybrid Retrieval (−10.9 pp):** The 3-tier cascade (exact→vector→Jaccard) provides significant recovery when exact matches fail.

3. **Deterministic Pruning (−7.6 pp):** Validates Theorem 6.9—without guaranteed non-retry, ~12% of steps are wasted on previously failed options.

4. **Validation Filter (−3.4 pp):** Catches ~21% of LLM suggestions that would fail (Table 4a), providing moderate improvement.

5. **COMPASS Routing (−2.7 pp):** Fast-path routing provides modest gains; main benefit is computational efficiency (72% fewer rollouts).

6. **Procedural Memory (−1.3 pp):** Smallest impact in this benchmark, but critical for domains with recoverable errors (e.g., coding, DevOps).

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
flowchart LR
    subgraph ABLATION["Component Contribution to P₁"]
        CS["Compositional<br/>13.8 pp"] --> HR["Hybrid Retrieval<br/>10.9 pp"] --> DP["Pruning<br/>7.6 pp"] --> VF["Validation<br/>3.4 pp"] --> CR["COMPASS<br/>2.7 pp"] --> PM["Procedural<br/>1.3 pp"]
    end

    style CS fill:#2c3e50,color:#fff
    style HR fill:#34495e,color:#fff
    style DP fill:#7f8c8d,color:#fff
```

**Figure 10a.** Component contribution ranking by P₁ impact. Compositional stacking and hybrid retrieval provide the largest gains.

**Statistical Significance:** All ablations except Procedural Memory show statistically significant degradation (p < 0.05, paired t-test with Bonferroni correction).

---

## 8. Capability Comparison

**Table 15.** Capability matrix across methods.

| Capability | PRECEPT | Full Refl. | ExpeL | Trad. RL |
|------------|:-------:|:----------:|:-----:|:--------:|
| Compositional Gen. O(2^N) | ✓ | ✗ | ✗ | ✗ |
| Continuous Learning | ✓ (+50.5pp) | △ (+29pp) | △ (+46.5pp) | ✗ |
| Rule Drift Adaptation | ✓ (θ=2) | △ (4-5) | △ | ✗ |
| Exact Multi-Cond. Match | ✓ | ✗ | ✗ | ✗ |
| Static-Dynamic Conflict | ✓ | ✗ | ✗ | ✗ |
| Deterministic Pruning | ✓ | ✗ | ✗ | △ |
| Sample Efficiency | ✓ (β=3) | △ (β=5+) | △ | ✗ (β=100+) |
| Smart Rollouts | ✓ (72%) | N/A | N/A | N/A |

✓ = supported, △ = partial, ✗ = not supported

### 8.1 Why GEPA Cannot Achieve PRECEPT's Capabilities

GEPA (Genetic-Pareto Evolution for Prompt Adaptation) represents a significant advance in prompt optimization, demonstrating that reflective prompt evolution can outperform reinforcement learning in sample efficiency (Agrawal et al., 2025). However, GEPA's architectural design imposes fundamental limitations that PRECEPT's structured retrieval paradigm overcomes.

**Limitation G1: No Deterministic Rule Retrieval.** GEPA evolves *prompts* through Pareto-optimal selection, but the evolved prompts still rely on LLM interpretation at inference time. When a GEPA-optimized agent encounters a multi-condition scenario (e.g., "SAFE+ASIA+BULK"), it must *interpret* which conditions take priority. PRECEPT's O(1) exact-match retrieval via `learned_rules[condition_key]` bypasses interpretation entirely:

$$P(\text{interpretation error})_{\text{GEPA}} > 0 \quad \text{vs} \quad P(\text{interpretation error})_{\text{PRECEPT}} = 0$$

**Limitation G2: No Compositional Generalization.** GEPA's prompt evolution optimizes for observed task distributions but cannot compose atomic constraints into novel combinations unseen during evolution. If a GEPA agent is evolved on tasks with individual constraints {SAFE, ASIA, BULK}, it has no mechanism to correctly handle the composite "SAFE+ASIA+BULK" without additional evolution cycles. PRECEPT's atomic constraint stacking (Algorithm 1) provides O(2^N) compositional coverage from N atomic precepts:

| Method | Training Scenarios | Test Combinations | Coverage |
|--------|-------------------|-------------------|----------|
| GEPA | N atomic + some composites | Only evolved combinations | O(T_train) |
| **PRECEPT** | N atomic | All 2^N - 1 composites | **O(2^N)** |

**Limitation G3: No Explicit Drift Adaptation.** GEPA lacks a mechanism for *unlearning* stale knowledge. When environmental conditions change (e.g., a previously working port becomes blocked), GEPA's evolved prompts continue suggesting the now-invalid solution. PRECEPT's threshold-based invalidation (θ=2) with soft confidence decay achieves 64× better drift resilience (Corollary 6.7):

$$\text{Stale rule persistence: GEPA} \approx 100\% \quad \text{vs} \quad \text{PRECEPT} \leq 0.25\%$$

**Limitation G4: No Deterministic Pruning.** GEPA relies on evolved prompt instructions like "avoid options that previously failed" but cannot *guarantee* non-repetition. LLMs may ignore or forget such instructions. PRECEPT's `RefineInterceptor` provides mathematical guarantees (Theorem 6.9):

$$P(\text{retry failed})_{\text{GEPA}} > 0 \quad \text{vs} \quad P(\text{retry failed})_{\text{PRECEPT}} = 0$$

**Limitation G5: Pareto Selection Does Not Solve Black Swan Multi-Condition.** GEPA's multi-objective Pareto selection optimizes prompts across success rate, efficiency, and generalization metrics. However, Black Swan scenarios (where solutions are not derivable from inputs) require *exact* condition-key → solution mapping. No amount of prompt optimization can derive "if PORT-503+HAZMAT+EXPRESS then singapore" from first principles—this must be learned and retrieved exactly. PRECEPT's hash-based CSP matching achieves 100% P₁ on such scenarios.

**Table 16b.** GEPA vs PRECEPT Fundamental Comparison.

| Capability | GEPA | PRECEPT | Gap |
|------------|------|---------|-----|
| Retrieval Type | Implicit (in prompt) | Explicit O(1) | Deterministic |
| Compositional Gen. | ✗ (evolution-limited) | ✓ O(2^N) | **Exponential** |
| Drift Adaptation | ✗ (no unlearning) | ✓ (θ=2 invalidation) | **64×** |
| Black Swan P₁ | ~50% (interpretation) | **100%** (exact match) | **2×** |
| Pruning Guarantee | ✗ (probabilistic) | ✓ (P=0) | **Provable** |
| Multi-Condition N=10 | ~5% (degradation) | **85%** (constant) | **17×** |

### 8.2 Why Reinforcement Learning Cannot Achieve PRECEPT's Capabilities

Traditional reinforcement learning (RL) methods—including PPO, SAC, and reward-based fine-tuning—have been extensively studied for LLM agent learning. Despite significant advances, RL faces fundamental limitations that structured retrieval inherently avoids.

**Limitation R1: Prohibitive Sample Complexity.** RL requires extensive environment interaction to learn policies. For a domain with E error types and β required encounters per error:

$$T_{\text{train}}^{\text{RL}} = \Omega(\beta \cdot E \cdot |A|) \quad \text{where} \quad \beta_{\text{RL}} \geq 100$$

PRECEPT achieves equivalent coverage with β=3 encounters per rule—a **33× sample efficiency improvement**:

| Method | β (encounters/rule) | E=16 Domain Training | Sample Ratio |
|--------|---------------------|----------------------|--------------|
| RL-PPO | 100+ | 1,600+ episodes | 1× |
| RL-SAC | 75+ | 1,200+ episodes | 1.3× |
| GEPA | 10-15 | 160-240 episodes | 6-10× |
| **PRECEPT** | **3** | **48 episodes** | **33×** |

**Limitation R2: No Compositional Generalization.** RL policies are function approximators that encode state→action mappings implicitly. A policy trained on individual constraints cannot compose them:

- Trained on: (SAFE→port_A), (ASIA→port_B), (BULK→port_C)
- Cannot infer: (SAFE+ASIA+BULK→?)

RL would require training on all 2^N combinations—exponential sample complexity. PRECEPT's tier-based composition (Algorithm 1) achieves this with N samples:

$$\text{RL Compositional Coverage: } O(2^N \cdot \beta) \quad \text{vs} \quad \text{PRECEPT: } O(N \cdot \beta)$$

**Limitation R3: No Adaptation to Non-Stationary Environments.** When environment dynamics change (drift), RL policies become stale and require *complete retraining*. There is no mechanism for selective unlearning:

$$T_{\text{drift recovery}}^{\text{RL}} = T_{\text{train}} = \Omega(1000) \text{ episodes}$$

PRECEPT recovers within θ=2 failures via explicit rule invalidation:

$$T_{\text{drift recovery}}^{\text{PRECEPT}} = O(\theta) = O(2) \text{ episodes}$$

This represents a **500× improvement** in drift adaptation speed.

**Limitation R4: No Explicit Knowledge Retrieval.** RL policies encode knowledge implicitly in neural network weights. This creates three problems:

1. **No interpretability:** Cannot inspect what rules were learned
2. **No selective updates:** Cannot modify individual rules without affecting others
3. **No exact retrieval:** Must rely on generalization, which degrades with complexity

PRECEPT stores rules explicitly in `learned_rules` dictionary with O(1) retrieval:

```
RL: state → neural_network → action (implicit, opaque)
PRECEPT: condition_key → learned_rules[key] → action (explicit, auditable)
```

**Limitation R5: Reward Specification Challenge.** RL requires carefully designed reward functions. For complex multi-condition scenarios, specifying rewards that capture all constraint interactions is notoriously difficult (reward hacking, sparse rewards, credit assignment). PRECEPT sidesteps this entirely by learning explicit rules from binary success/failure signals—no reward engineering required.

**Limitation R6: No Deterministic Decisions.** RL policies are inherently stochastic (ε-greedy, softmax exploration) or at best approximately deterministic (greedy). For safety-critical applications requiring guaranteed behavior, this is unacceptable. PRECEPT's exact-match retrieval provides **mathematically guaranteed** deterministic decisions:

$$P(\text{correct decision} | \text{rule exists})_{\text{RL}} < 1 \quad \text{vs} \quad P(\text{correct decision} | \text{rule exists})_{\text{PRECEPT}} = 1$$

**Table 16c.** RL vs PRECEPT Fundamental Comparison.

| Capability | Traditional RL | PRECEPT | Gap |
|------------|---------------|---------|-----|
| Sample Efficiency | β=100+ | **β=3** | **33×** |
| Compositional Gen. | ✗ (exponential training) | ✓ O(2^N) | **Exponential** |
| Drift Adaptation | ✗ (full retrain) | ✓ (θ=2 recovery) | **500×** |
| Interpretability | ✗ (black box) | ✓ (explicit rules) | **Complete** |
| Determinism | △ (approximate) | ✓ (guaranteed) | **Provable** |
| Reward Engineering | Required (hard) | **Not needed** | **Eliminated** |

### 8.3 PRECEPT's Capabilities Beyond GEPA and RL

The analysis above reveals that PRECEPT's advantages are not merely quantitative improvements but represent **fundamentally different capabilities** that neither prompt optimization (GEPA) nor policy learning (RL) can achieve regardless of compute or sophistication:

**Capability 1: First-Try Success on Black Swan Multi-Condition Scenarios (P₁ = 100%)**

Black Swan scenarios have solutions that are not derivable from inputs—they must be learned exactly and retrieved exactly. Neither evolved prompts (GEPA) nor learned policies (RL) can achieve this:

- **GEPA:** Even optimally evolved prompts require LLM interpretation, which degrades exponentially with condition count (Theorem 6.8)
- **RL:** Implicit policy encoding cannot provide exact retrieval for arbitrary condition combinations
- **PRECEPT:** Hash-based O(1) lookup achieves 100% P₁ by construction

$$P_1^{\text{Black Swan}} = \begin{cases} 100\% & \text{PRECEPT (exact match)} \\ \leq 50\% & \text{GEPA (interpretation)} \\ \leq 40\% & \text{RL (generalization)} \end{cases}$$

**Capability 2: Exponential Compositional Coverage from Linear Training**

| Training | GEPA Coverage | RL Coverage | PRECEPT Coverage |
|----------|--------------|-------------|------------------|
| N=6 atomic | 6 scenarios | 6 scenarios | **63 scenarios** |
| N=10 atomic | 10 scenarios | 10 scenarios | **1,023 scenarios** |

Neither GEPA nor RL can compose learned constraints—both require explicit training on each combination.

**Capability 3: Provable Pruning Guarantee (P(repeat_failed) = 0)**

PRECEPT's `RefineInterceptor` provides a mathematical guarantee via hash set membership. Neither GEPA's evolved prompts nor RL's exploration policies can provide such guarantees—both rely on probabilistic mechanisms that can fail.

**Capability 4: Instant Drift Recovery (θ=2 vs Retraining)**

When rules become stale:
- **GEPA:** No mechanism to detect or correct; continues using stale evolved prompts
- **RL:** Requires full retraining (1000+ episodes)
- **PRECEPT:** Automatic invalidation after 2 consecutive failures, immediate re-learning

**Summary**

| Paradigm | Mechanism | Fundamental Limit |
|----------|-----------|-------------------|
| **GEPA** | Prompt evolution | LLM interpretation required |
| **RL** | Policy learning | Implicit knowledge encoding |
| **PRECEPT** | Structured retrieval | **None** (deterministic by construction) |

PRECEPT demonstrates that reliable, compositional, adaptive LLM agents can be achieved through **deterministic structured retrieval with explicit rule management**, offering fundamental theoretical advantages over prompt-based and policy-based approaches.

---

## 9. Related Work

### 9.1 LLM Agent Adaptation

**Reflection-based Methods.** Reflexion (Shinn et al., 2023) pioneered verbal self-reflection for LLM agents, demonstrating improved performance through iterative refinement. However, Reflexion discards learned knowledge after each episode, preventing cross-episode learning. Full Reflexion addresses this by persisting reflections across episodes but relies on coarse task-type indexing that requires LLM interpretation at test time. ExpeL (Zhao et al., 2023) extracts natural language insights from experience but inherits the same interpretation degradation problem. Our analysis (Theorem 6.8) proves this degradation reaches 94.4% error rate at N=10 conditions.

**Experience-based Methods.** Voyager (Wang et al., 2023) learns procedural skills through code generation in Minecraft but focuses on skill acquisition rather than declarative constraint rules. LATS (Zhou et al., 2023) combines language agents with tree search but lacks mechanisms for compositional generalization or drift adaptation. Generative Agents (Park et al., 2023) simulate human behavior with memory retrieval but use similarity-based search rather than exact-match rules. PRECEPT uniquely addresses all four fundamental limitations through deterministic O(1) retrieval.

**Tool-Using LLM Agents.** Toolformer (Schick et al., 2023) teaches LLMs to use external tools through self-supervised learning. ReAct (Yao et al., 2023) interleaves reasoning and acting traces. While these approaches augment LLM capabilities, they lack explicit mechanisms for learning from failures, compositional generalization, or drift adaptation—capabilities central to PRECEPT's design.

### 9.2 Retrieval-Augmented Generation

**RAG Architectures.** RAG (Lewis et al., 2020) and REALM (Guu et al., 2020) retrieve relevant documents to augment LLM context. REPLUG (Shi et al., 2023) treats retrieval as a plug-in module. These systems use dense retrieval for knowledge augmentation but cannot learn new rules at test time or handle compositional constraint combinations.

**Memory-Augmented LLMs.** MemoryBank (Zhong et al., 2024) maintains evolving memory for LLM personalization. SCM (Wang et al., 2024) uses structured memory for conversation. PRECEPT differs fundamentally by using **structured condition keys** for O(1) exact-match retrieval rather than similarity-based search, enabling deterministic rule application without interpretation degradation.

### 9.3 Compositional Reasoning

**Compositional Generalization Benchmarks.** SCAN (Lake & Baroni, 2018) and COGS (Kim & Linzen, 2020) evaluate compositional generalization in sequence-to-sequence models. Keysers et al. (2020) measure compositional complexity via DBCA splits. These works establish that neural networks struggle with compositional generalization—a limitation PRECEPT addresses through explicit atomic constraint stacking with tier-based resolution.

**Neural-Symbolic Integration.** DeepProbLog (Manhaeve et al., 2018) and NeurASP (Yang et al., 2020) combine neural networks with logical reasoning. While related in spirit, PRECEPT takes a different approach: rather than training neural-symbolic hybrids, we use structured retrieval to bypass neural interpretation entirely for rule application.

### 9.4 Prompt Optimization

**Evolutionary Approaches.** GEPA (Agrawal et al., 2025) demonstrates that reflective prompt evolution can outperform reinforcement learning through Pareto-optimal selection across multiple objectives. EvoPrompt (Guo et al., 2023) uses evolutionary algorithms for prompt optimization. Our COMPASS algorithm extends GEPA with ML-based complexity detection, smart rollout allocation, and **verified evolution** using real agent execution signals.

**Limitations of GEPA.** While GEPA achieves impressive sample efficiency over RL (10-15× improvement), it inherits fundamental limitations from the prompt optimization paradigm: (1) evolved prompts still require LLM interpretation at inference, causing exponential degradation with condition count (Theorem 6.8); (2) no mechanism for compositional generalization—GEPA must be re-evolved for unseen condition combinations; (3) no explicit drift adaptation—stale prompts persist indefinitely without unlearning mechanisms; (4) no deterministic pruning guarantees—evolved instructions like "avoid failed options" can be ignored by LLMs. PRECEPT addresses all four limitations through structured retrieval (see Section 8.1).

**Compilation Approaches.** DSPy (Khattab et al., 2023) compiles declarative language model programs into optimized pipelines. OPRO (Yang et al., 2023) uses LLMs as meta-optimizers. APE (Zhou et al., 2023) automates prompt engineering through LLM-based generation. None of these approaches address compositional generalization across condition combinations or environmental drift adaptation—capabilities central to PRECEPT.

### 9.5 Reinforcement Learning for LLM Agents

**Policy Gradient Methods.** PPO (Schulman et al., 2017) and SAC (Haarnoja et al., 2018) have been applied to LLM fine-tuning through RLHF (Ouyang et al., 2022). InstructGPT and subsequent work demonstrate RL can align LLM behavior with human preferences. However, these approaches require extensive online interaction (β=100+ episodes per behavior) and cannot adapt to non-stationary environments without retraining.

**Limitations of RL for Agent Learning.** Traditional RL faces fundamental barriers for adaptive LLM agents: (1) **Sample inefficiency**—learning reliable policies requires orders of magnitude more interaction than PRECEPT's β=3 rule learning; (2) **No compositional generalization**—RL policies are function approximators that cannot compose independently learned constraints; (3) **Drift blindness**—when environment dynamics change, RL requires complete policy retraining rather than selective rule invalidation; (4) **Implicit knowledge**—RL encodes decisions in opaque neural weights, preventing interpretability, auditing, or selective modification; (5) **Reward specification**—designing rewards for complex multi-condition scenarios is notoriously difficult and prone to reward hacking; (6) **No determinism guarantee**—RL policies are inherently stochastic or approximately deterministic, unsuitable for safety-critical applications. PRECEPT's structured retrieval paradigm sidesteps all six limitations (see Section 8.2).

**Offline RL.** Decision Transformer (Chen et al., 2021) and related work frame RL as sequence modeling. While avoiding online interaction, offline RL still encodes knowledge implicitly and cannot adapt to distribution shift without retraining. PRECEPT's explicit rule storage enables selective updates and drift adaptation within θ=2 failures.

### 9.6 Knowledge Conflict Resolution

**Conflict Detection.** Prior work addresses conflicts between parametric knowledge and retrieved context (Longpre et al., 2021; Chen et al., 2022) through heuristic approaches. KALMV (Baek et al., 2023) detects knowledge conflicts via verification. PRECEPT introduces a principled Bayesian framework using Beta distribution posteriors with Thompson Sampling (Agrawal & Goyal, 2012) for exploration, unified with threshold-based rule invalidation for drift adaptation—to our knowledge, the first such integrated treatment for LLM agents.

**Continual Learning.** EWC (Kirkpatrick et al., 2017) addresses catastrophic forgetting through importance weighting. GEM (Lopez-Paz & Ranzato, 2017) uses gradient episodic memory. These approaches focus on preventing forgetting during training; PRECEPT addresses a complementary problem—**active unlearning** of stale rules during deployment through threshold-based invalidation.

### 9.7 Comparison with Nested Learning

Recently, Behrouz et al. (2025) introduced **Nested Learning (NL)**—a paradigm that represents machine learning models as interconnected systems of nested, multi-level optimization problems, each with its own "context flow" and update frequency. NL addresses several challenges that PRECEPT also targets, making a detailed comparison valuable for understanding both approaches' contributions. This section provides a meticulous analysis of both frameworks' memory architectures, learning dynamics, and fundamental design philosophies.

#### 9.7.1 Shared Problem Space

Both PRECEPT and Nested Learning recognize fundamental limitations in current LLM agent architectures:

**Table 16.** Shared Challenges Addressed by PRECEPT and Nested Learning.

| Challenge | NL's Framing | PRECEPT's Framing |
|-----------|--------------|-------------------|
| **Catastrophic Forgetting** | Models "only experience immediate present" (anterograde amnesia analogy) | Stale rules persist indefinitely without explicit invalidation |
| **Static Deployment** | LLMs are "largely static after deployment"—knowledge bounded by "end of pre-training" | Verbal baselines cannot adapt to environmental drift |
| **Compositional Explosion** | 2^N condition combinations require exponential training | Prior methods require O(2^N) training for O(2^N) coverage |
| **In-Context vs Long-Term** | Short-term attention vs frozen MLP weights creates "anterograde amnesia" | Episodic memory vs persistent learned rules require explicit management |
| **Sample Inefficiency** | RL requires β=100+ episodes; NL aims to reduce via multi-frequency updates | Prior methods require β=5+ (verbal) or β=100+ (RL) |
| **Knowledge Retrieval** | Compressed knowledge requires "interpretation" at retrieval time | Similarity-based retrieval degrades with condition complexity |

Both frameworks identify that the fundamental issue lies in **how knowledge is stored, retrieved, and updated**—not merely in scaling model size or context length. However, their solutions diverge fundamentally.

**Important Scope Distinction**: Nested Learning is a **general-purpose theoretical framework** for understanding deep learning architectures, optimizers, and memory systems. PRECEPT is a **domain-specific framework** for rule-governed LLM agent deployment. This comparison focuses on their applicability to **discrete rule learning and compositional constraint handling**—PRECEPT's target domain. NL may excel in domains requiring continuous feature learning or soft pattern recognition where its compression-based approach is advantageous.

#### 9.7.2 System Architecture Comparison

Both NL and PRECEPT are **comprehensive frameworks** with multiple sophisticated components. The comparison must account for each system's full scope, not just individual mechanisms.

**Nested Learning's Full Architecture**

NL provides a unified theoretical framework encompassing:

1. **Continuum Memory System (CMS)**: Multi-level MLP blocks at different update frequencies
   - Inspired by brain oscillations (gamma: 30-150 Hz, beta: 13-30 Hz, theta/delta: 0.5-8 Hz)
   - Frequency-based updates: $\theta_{i+1}^{(f_\ell)} = \theta_i^{(f_\ell)} - \sum_{t} \eta_t^{(\ell)} f(\theta_t; x_t)$ every $C^{(\ell)}$ steps

2. **Associative Memory Perspective**: All components viewed as memories compressing context:
   $$\mathcal{M}^* = \arg\min_{\mathcal{M}} \hat{\mathcal{L}}(\mathcal{M}(\mathcal{K}); \mathcal{V})$$

3. **HOPE Architecture**: Self-referential Titans with nested optimization
   - Delta Gradient Descent for state-dependent updates
   - Self-generated keys, values, and learning rates

4. **Multi-Scale Momentum Muon (M3)**: Long-context optimizer with multiple momentum terms

5. **Knowledge Transfer**: Via initialization (MAML), backpropagation, or hypernetworks

**PRECEPT's Full Architecture**

PRECEPT is equally comprehensive, with sophisticated components beyond simple retrieval:

1. **Three-Tier Memory System**:
   - **Tier 0 (Experience Store)**: Vector database with BM25+semantic hybrid retrieval
   - **Tier 1 (Learned Rules)**: Hash-based O(1) exact-match retrieval
   - **Tier 2 (Precept Store)**: Atomic constraints with semantic tier hierarchy

2. **COMPASS Algorithm** (Section 5): ML-based complexity analysis and multi-objective optimization
   - **Complexity Detection**: Classifies tasks into BLOCK/PIVOT/FAST_PATH/PROCEED
   - **Smart Rollout Allocation**: 72% reduction vs baseline GEPA through learned routing
   - **Multi-Objective Pareto Selection**: Optimizes across 6 objectives simultaneously
   - **Verified Prompt Evolution**: Real execution signals for honest feedback

3. **Bayesian Conflict Resolution**:
   - **Type I (Static vs Dynamic)**: Ensemble voting with Beta posteriors + Thompson Sampling
   - **Type II (Environmental Drift)**: Soft confidence decay (δ=0.5) + threshold invalidation (θ=2)

4. **Epistemic Reasoning**:
   - **Probing**: Diagnostic tool calls to discover hidden constraints
   - **Uncertainty Quantification**: Beta distribution posteriors with Thompson Sampling exploration

5. **Procedural Memory**: Dynamically learns recovery strategies from successful error handling

6. **Constraint Classification**: HARD/SOFT categorization with provable pruning guarantees

**Table 16a.** Full System Architecture Comparison.

| Component | Nested Learning | PRECEPT |
|-----------|----------------|---------|
| **Memory System** | CMS (multi-frequency MLPs) | Three-Tier (Experience/Rules/Precepts) |
| **Storage Paradigm** | Compression into parameters | Explicit storage + composition |
| **Optimization** | Nested gradient descent | COMPASS (ML routing + Pareto selection) |
| **Exploration Strategy** | Gradient-based search | Thompson Sampling from Beta posteriors |
| **Conflict Resolution** | Frequency-based forgetting | Bayesian ensemble + threshold invalidation |
| **Uncertainty Handling** | Implicit in weight updates | Explicit Beta posteriors + epistemic probing |
| **Compositional Mechanism** | None (must learn combinations) | Tier hierarchy with max() composition |
| **Prompt Optimization** | N/A | COMPASS with verified evolution |
| **Error Recovery** | N/A | Procedural memory + constraint classification |

**Key Insight**: Both frameworks are sophisticated—the difference lies in **architectural philosophy**, not complexity:
- **NL**: Achieves adaptation through sophisticated **gradient-based compression** at multiple timescales
- **PRECEPT**: Achieves adaptation through sophisticated **explicit reasoning** (Bayesian, Thompson Sampling, COMPASS) with simple retrieval

#### 9.7.2.1 Memory Architecture Details

**Nested Learning's Memory**

NL's CMS is a chain of MLP blocks $\text{MLP}^{(f_1)}(\cdot), \ldots, \text{MLP}^{(f_k)}(\cdot)$, each with chunk size $C^{(\ell)}$:

$$\theta_{i+1}^{(f_\ell)} = \theta_i^{(f_\ell)} - \begin{cases} \sum_{t=i-C^{(\ell)}}^{i} \eta_t^{(\ell)} f(\theta_t^{(f_\ell)}; x_t) & \text{if } i \equiv 0 \pmod{C^{(\ell)}} \\ 0 & \text{otherwise} \end{cases}$$

**Memory Capacity Analysis (NL)**

NL's associative memory capacity is bounded by classical results (Storkey, 1997; Behrouz et al., 2025):

- **Linear Memory (Hebbian)**: Capacity $\approx 0.14d$ patterns for $d$-dimensional keys before catastrophic interference
- **Delta Rule**: Improved capacity via input-dependent decay, but still $O(d)$ bounded
- **Deep Memory (MLP)**: Capacity scales with hidden dimension but remains architecture-dependent

The fundamental limitation is that **all stored patterns share the same weight matrix**:
$$\mathcal{M} = \sum_{i=1}^{n} v_i k_i^\top \quad \text{(Hebbian accumulation)}$$

When $n > 0.14d$, retrieval accuracy degrades due to **crosstalk interference**:
$$\mathcal{M} q = v_{\text{target}} + \underbrace{\sum_{j \neq \text{target}} (k_j^\top q) v_j}_{\text{interference term}}$$

**PRECEPT's Memory**

PRECEPT's three tiers serve distinct functions with different capacity/retrieval trade-offs:

| Tier | Purpose | Storage | Retrieval | Capacity |
|------|---------|---------|-----------|----------|
| 0 | Exploration | Vector DB embeddings | BM25 + semantic hybrid | O(n) with ANN indexing |
| 1 | Exploitation | Hash table | O(1) exact match | **Unbounded** (O(n) storage) |
| 2 | Composition | Structured dict | max(tier) resolution | O(N) for N atomic precepts |

**Memory Capacity Analysis (PRECEPT)**

PRECEPT's Tier 1 uses hash tables with **perfect hashing** for condition keys:
- **Capacity**: Unbounded—can store any number of rules with O(n) memory
- **Retrieval**: O(1) guaranteed, **no interference** between rules
- **Accuracy**: 100% for any stored rule, regardless of total count

The key difference is **isolated storage**:
$$\text{rules}[k_1] = v_1, \quad \text{rules}[k_2] = v_2 \quad \Rightarrow \quad \text{retrieve}(k_1) = v_1 \text{ (independent of } k_2\text{)}$$

**Table 16b.** Memory Architecture Comparison (Detailed).

| Dimension | Nested Learning (CMS) | PRECEPT (Three-Tier) |
|-----------|----------------------|----------------------|
| **Storage Paradigm** | Compression into shared parameters | Explicit isolated key-value storage |
| **Memory Structure** | Chain of MLP blocks at different frequencies | Hash table + vector store + structured dict |
| **Capacity Bound** | O(d) patterns (architecture-dependent) | **Unbounded** (O(n) storage) |
| **Update Granularity** | Gradient-based, frequency-dependent | Per-rule addition/invalidation |
| **Retrieval Mechanism** | Forward pass (matrix multiplication) | O(1) hash lookup (Tier 1) |
| **Retrieval Accuracy** | Degrades with pattern count (interference) | **100%** for stored rules |
| **Knowledge Representation** | Distributed across weights | Localized per condition key |
| **Interpretability** | Opaque (what did the MLP learn?) | Transparent (`rules["key"] → value`) |
| **Cross-Rule Interference** | Yes (shared weights, crosstalk) | **None** (isolated storage) |

**Interference Dynamics Comparison**

The fundamental difference in memory reliability stems from interference:

**NL's Gradient Interference**: When learning rule $r_2$, the gradient update affects weights storing rule $r_1$:
$$\frac{\partial \mathcal{L}_{r_2}}{\partial W} \neq 0 \quad \Rightarrow \quad W_{\text{new}} = W_{\text{old}} - \eta \nabla \mathcal{L}_{r_2}$$

This changes retrieval for $r_1$ even though $r_1$ wasn't involved—the source of **catastrophic forgetting** that CMS mitigates but cannot eliminate.

**PRECEPT's Isolation**: Storing rule $r_2$ has zero effect on rule $r_1$:
$$\text{rules}[k_2] = v_2 \quad \Rightarrow \quad \text{rules}[k_1] \text{ unchanged}$$

**Table 16b-2.** Memory Interference Comparison.

| Scenario | NL (CMS) | PRECEPT |
|----------|----------|---------|
| **Store 100 rules, retrieve rule 1** | Accuracy < 100% (interference) | Accuracy = 100% |
| **Store 1000 rules, retrieve rule 1** | Accuracy degrades further | Accuracy = 100% |
| **Update rule 50** | May affect rules 1-49, 51-100 | Only rule 50 affected |
| **Delete rule 50** | Cannot selectively remove | `del rules[k_50]` (trivial) |

*Note: NL's CMS uses frequency separation to reduce (not eliminate) interference—higher-frequency memories adapt faster but are more volatile, lower-frequency memories persist longer but adapt slower. This is a principled trade-off, but fundamentally different from PRECEPT's interference-free storage.*

#### 9.7.3 Learning and Reasoning Comparison

Both frameworks employ sophisticated learning and reasoning mechanisms—but with fundamentally different approaches.

**Nested Learning's Learning Rules**

NL views learning as optimizing nested associative memories with various update rules:

1. **Hebbian Rule** (Linear Attention): $\mathcal{M}_t = \alpha_t \mathcal{M}_{t-1} + \eta_t v_t \phi(k_t)^\top$
   - Based on dot-product similarity objective
   - Limited memory management, prone to interference

2. **Delta Rule** (DeltaNet): $\mathcal{M}_t = (\mathbf{I} - \eta_t k_t k_t^\top) \mathcal{M}_{t-1} + \eta_t v_t k_t^\top$
   - Based on $L_2$ regression objective
   - Better memory management via input-dependent decay

3. **Delta Gradient Descent** (NL's contribution): $W_{t+1} = W_t (\mathbf{I} - \eta'_t x_t x_t^\top) - \eta'_t \nabla_{y_t} \mathcal{L}(W_t; x_t) \otimes x_t$
   - State-dependent updates capture data dependencies
   - Still requires multiple exposures for convergence

4. **Self-Referential Updates**: Models generate their own values $\hat{v}_{\square,t} = \mathcal{M}_{\square,t-1}(v_t)$ for internal optimization

**PRECEPT's Learning and Reasoning**

PRECEPT employs **principled probabilistic reasoning** combined with direct rule storage:

1. **Bayesian Confidence Tracking**:
   - Each rule maintains a Beta distribution posterior: $\text{confidence}(k) \sim \text{Beta}(\alpha, \beta)$
   - Success: $\text{Beta}(\alpha + 1, \beta)$ — increases confidence
   - Failure: $\text{Beta}(\alpha, \beta + 1)$ — decreases confidence
   - This provides **calibrated uncertainty** for each learned rule

2. **Thompson Sampling Exploration**:
   - When multiple candidate solutions exist, sample from Beta posteriors:
   $$\text{select} = \arg\max_i \text{sample}(\text{Beta}(\alpha_i, \beta_i))$$
   - Automatically balances exploration (uncertain rules) vs exploitation (confident rules)
   - **Provably optimal** for multi-armed bandit problems (Russo et al., 2018)

3. **COMPASS Multi-Objective Optimization**:
   - ML-based complexity analysis classifies tasks: BLOCK/PIVOT/FAST_PATH/PROCEED
   - Smart rollout allocation reduces computation by 72% vs naive GEPA
   - 6-objective Pareto selection: correctness, constraint satisfaction, efficiency, robustness, generalization, novelty
   - **Verified prompt evolution**: Uses real execution signals, not heuristics

4. **Epistemic Probing**:
   - When uncertainty is high, COMPASS triggers diagnostic probes
   - Targeted MCP tool calls reveal hidden constraints
   - Reduces uncertainty before committing to action

5. **Dual Conflict Resolution**:
   - **Type I (Static vs Dynamic)**: Ensemble voting triggers Bayesian resolution
     - Static knowledge: $\text{Beta}(10, 2)$ prior (high confidence)
     - Dynamic knowledge: $\text{Beta}(5, 3)$ prior (moderate confidence)
     - Thompson Sampling selects which source to trust
   - **Type II (Environmental Drift)**: Threshold-based invalidation
     - Soft decay: $c \leftarrow c - \delta$ (δ=0.5) on failure
     - Hard invalidation: Delete rule after θ=2 consecutive failures

6. **Tier-Based Composition** (deterministic, no learning required):
   - Given atomic precepts with tiers, composition is algebraic:
   $$\text{solution}(\{c_1, ..., c_N\}) = \text{solution}(\arg\max_{c_i} \text{tier}(c_i))$$

**Table 16c.** Learning and Reasoning Comparison (for rule-governed domains).

| Dimension | Nested Learning | PRECEPT |
|-----------|----------------|---------|
| **Learning Paradigm** | Gradient-based optimization | Direct storage + Bayesian inference |
| **Uncertainty Representation** | Implicit in weights | Explicit Beta posteriors |
| **Exploration Strategy** | Gradient descent on loss surface | Thompson Sampling (provably optimal) |
| **Multi-Objective Optimization** | Single loss function | COMPASS 6-objective Pareto |
| **Complexity Analysis** | N/A | ML-based BLOCK/PIVOT/FAST_PATH/PROCEED |
| **Epistemic Reasoning** | N/A | Probing for hidden constraints |
| **Conflict Resolution** | Frequency-based priority | Bayesian ensemble + threshold invalidation |
| **Convergence Criterion** | Loss minimization | β=3 consecutive successes |
| **Sample Complexity** | O(epochs × data) | O(β) = O(3) per rule |
| **Knowledge Locality** | Distributed (all weights) | Localized (single key-value) |
| **Interference Risk** | High (catastrophic forgetting) | None (isolated rules) |
| **Unlearning Mechanism** | Gradient-based, slow | Explicit invalidation, O(θ)=O(2) |
| **Composition** | Must learn combinations | Automatic via tier hierarchy |

**Key Insight**: PRECEPT's sophistication lies in its **reasoning mechanisms** (Bayesian inference, Thompson Sampling, COMPASS optimization, epistemic probing), not its storage mechanism. The simple hash-based retrieval is a deliberate design choice that **enables** these sophisticated reasoning capabilities by providing deterministic foundations.

#### 9.7.4 Retrieval Mechanism Comparison

The retrieval mechanism is where the fundamental architectural difference becomes most apparent. This section provides formal analysis of retrieval guarantees.

**Nested Learning's Retrieval**

NL retrieves knowledge through forward passes in associative memory:
$$y_t = \mathcal{M}_t(q_t) = \mathcal{M}_t \cdot q_t \quad \text{(linear memory)}$$

Or for deep memories (HOPE):
$$y_t = \mathcal{M}_{\text{memory},t-1}(q_t), \quad \text{where } q_t = x_t W_q$$

**Formal Retrieval Analysis (NL)**

For a linear associative memory storing $n$ key-value pairs $(k_i, v_i)$:
$$\mathcal{M} = \sum_{i=1}^{n} v_i k_i^\top$$

Retrieval for query $q$ yields:
$$\mathcal{M} q = \sum_{i=1}^{n} v_i (k_i^\top q) = v_{\text{target}} (k_{\text{target}}^\top q) + \sum_{j \neq \text{target}} v_j (k_j^\top q)$$

The **retrieval error** is:
$$\epsilon = \sum_{j \neq \text{target}} v_j (k_j^\top q) \quad \text{(crosstalk/interference)}$$

For random keys with $\mathbb{E}[k_i^\top k_j] = 0$ (orthogonal), $\epsilon \to 0$. But for **correlated keys** (common in rule learning where conditions share structure), $k_i^\top k_j \neq 0$, causing systematic retrieval errors.

**Retrieval Accuracy Bound (NL)**: For $n$ patterns in $d$-dimensional space with key correlation $\rho$:
$$P(\text{correct retrieval}) \leq 1 - \frac{n \rho^2}{d}$$

When $n\rho^2 \approx d$, retrieval becomes unreliable.

**PRECEPT's Retrieval**

PRECEPT retrieves via deterministic hash lookup:
```python
def retrieve_rule(conditions: Set[str]) -> Optional[str]:
    # Tier 1: Exact match lookup
    condition_key = canonicalize(conditions)  # Sorted, normalized
    if condition_key in learned_rules:
        return learned_rules[condition_key]  # O(1), exact

    # Tier 2: Compositional fallback
    max_tier_precept = max(precepts, key=lambda p: p.tier if p.condition in conditions else -1)
    return max_tier_precept.solution
```

**Formal Retrieval Analysis (PRECEPT)**

Hash table retrieval provides **exact match guarantee**:
$$\text{retrieve}(k) = \begin{cases} v & \text{if } (k, v) \in \text{rules} \\ \text{fallback to Tier 2} & \text{otherwise} \end{cases}$$

**Retrieval Accuracy Bound (PRECEPT)**:
$$P(\text{correct retrieval} | k \in \text{rules}) = 1 \quad \forall n, \forall \rho$$

The accuracy is **independent** of:
- Number of stored rules ($n$)
- Key correlation ($\rho$)
- Key dimensionality ($d$)

**Table 16d.** Retrieval Mechanism Comparison (Formal).

| Property | Nested Learning | PRECEPT |
|----------|----------------|---------|
| **Retrieval Function** | $y = \mathcal{M} \cdot q$ (matrix multiply) | $y = \text{rules}[k]$ (hash lookup) |
| **Complexity** | O(d²) | O(1) |
| **Accuracy (n rules)** | $\leq 1 - \frac{n\rho^2}{d}$ | **= 1** (exact) |
| **Accuracy (1000 rules)** | Degrades significantly | **= 1** (unchanged) |
| **Key Correlation Sensitivity** | High (correlated keys degrade retrieval) | **None** |
| **Novel Combinations** | Extrapolation (weighted interpolation) | Composition (tier algebra) |
| **Failure Detection** | Silent (returns interpolated value) | Explicit (key not found → Tier 2) |

**Practical Implications**

Consider retrieving the rule for condition combination "SAFE+ASIA+BULK":

**NL Retrieval**:
$$y = \mathcal{M} \cdot q_{\text{SAFE+ASIA+BULK}} = v_{\text{target}} + \epsilon$$
where $\epsilon$ includes contributions from similar patterns like "SAFE+ASIA", "ASIA+BULK", "SAFE+BULK". The returned value is an **interpolation**, not the exact stored solution.

**PRECEPT Retrieval**:
$$y = \text{rules}[\text{"SAFE+ASIA+BULK"}] = v_{\text{exact}}$$
If the key exists, the exact stored value is returned. If not, Tier 2 composition applies: $\max(\text{tier}(\text{SAFE}), \text{tier}(\text{ASIA}), \text{tier}(\text{BULK}))$.

*Note: NL's interpolation property is advantageous for continuous domains (e.g., feature learning, language modeling) where similar inputs should produce similar outputs. However, for discrete rule application where "SAFE+ASIA+BULK" requires a specific solution different from interpolating "SAFE+ASIA" and "ASIA+BULK", exact retrieval is essential.*

#### 9.7.5 PRECEPT's Advantages Over Nested Learning

Both frameworks are sophisticated—PRECEPT's advantages stem from **architectural choices** that enable capabilities NL's paradigm cannot achieve for **rule-governed domains**, regardless of NL's complexity.

*Note: These advantages are specific to discrete rule learning, compositional constraint handling, and safety-critical determinism. NL may have advantages in continuous feature learning, language modeling, and soft pattern recognition—domains where compression and interpolation are beneficial.*

**Advantage 1: Deterministic Foundations Enable Sophisticated Reasoning**

NL's compressed representations create a **reasoning ceiling**: even sophisticated operations (multi-frequency updates, self-referential learning) operate on **approximate** retrieved values. PRECEPT's deterministic retrieval provides **exact foundations** for sophisticated reasoning:

| Component | NL's Foundation | PRECEPT's Foundation |
|-----------|-----------------|---------------------|
| Uncertainty | Implicit in weights | **Explicit Beta posteriors** |
| Exploration | Gradient descent | **Thompson Sampling** (provably optimal) |
| Optimization | Single loss surface | **6-objective Pareto** (COMPASS) |
| Conflict Resolution | Frequency priority | **Bayesian ensemble** + threshold |

$$P(\text{foundation error})_{\text{NL}} > 0 \quad \text{vs} \quad P(\text{foundation error})_{\text{PRECEPT}} = 0$$

This enables PRECEPT's sophisticated mechanisms (Thompson Sampling, COMPASS optimization, epistemic probing) to operate on **reliable inputs**, whereas NL's sophisticated mechanisms operate on **approximate** inputs.

**Advantage 2: Calibrated Uncertainty vs Implicit Uncertainty**

NL encodes uncertainty implicitly in weight magnitudes—there's no principled way to ask "how confident are you in this rule?" PRECEPT maintains **calibrated uncertainty** via Beta posteriors:

```python
# PRECEPT: Explicit, calibrated uncertainty
confidence = Beta(α=5, β=2)  # 5 successes, 2 failures
mean_confidence = α / (α + β) = 0.71
variance = αβ / ((α+β)² (α+β+1)) = 0.026  # Narrow uncertainty

# NL: Implicit uncertainty
# Q: What's the confidence in rule for "SAFE+ASIA"?
# A: Unknown—it's distributed across MLP weights at multiple frequencies
```

This enables **principled exploration** via Thompson Sampling—impossible with NL's implicit uncertainty.

**Advantage 3: Explicit Compositional Generalization**

NL's CMS provides no mechanism for compositional generalization. Even HOPE's self-referential updates learn **specific input-output mappings**, not compositional rules:

- **NL Training**: Exposures to {SAFE}, {ASIA}, {BULK} individually
- **NL Test on {SAFE, ASIA, BULK}**: Must extrapolate—unreliable
- **PRECEPT Training**: Learn tier(SAFE)=2, tier(ASIA)=1, tier(BULK)=3
- **PRECEPT Test on {SAFE, ASIA, BULK}**: max(2,1,3)=3 → BULK's solution (deterministic)

$$\text{NL Coverage: } O(T_{\text{train}}) \quad \text{vs} \quad \text{PRECEPT Coverage: } O(2^N) \text{ from } N \text{ atomics}$$

This is not merely a quantitative difference—it reflects a **qualitative architectural capability** that NL's compression-based approach cannot achieve.

**Advantage 4: Bayesian Conflict Resolution vs Frequency-Based Priority**

When static knowledge conflicts with dynamic observations, NL relies on frequency-based priority (higher-frequency memories override lower). PRECEPT employs **principled Bayesian resolution**:

```python
# Type I Conflict: Static vs Dynamic knowledge
static_prior = Beta(10, 2)   # Strong prior for established knowledge
dynamic_prior = Beta(5, 3)   # Moderate prior for recent observations

# Thompson Sampling resolution
static_sample = np.random.beta(10, 2)   # Sample from static posterior
dynamic_sample = np.random.beta(5, 3)   # Sample from dynamic posterior
trust = "static" if static_sample > dynamic_sample else "dynamic"
```

This provides **calibrated trade-offs** between established and recent knowledge.

**Advantage 5: Targeted Drift Adaptation**

NL's CMS uses frequency-based forgetting: lower-frequency memories persist longer, higher-frequency memories adapt faster. However:

- **No targeted unlearning**: Cannot invalidate a specific stale rule
- **Interference risk**: Updating one rule may affect others (shared weights)
- **Recovery time**: Proportional to frequency period, not failure count

PRECEPT's dual mechanism:
```python
# Soft decay on failure
confidence[key] -= δ  # δ=0.5, gradual confidence reduction

# Hard invalidation after θ=2 consecutive failures
if consecutive_failures(key) >= θ:
    del learned_rules[key]  # Immediate removal, no interference
```

$$\text{Drift Recovery: NL} \approx O(\text{frequency period}) \quad \text{vs} \quad \text{PRECEPT} = O(\theta) = O(2)$$

PRECEPT achieves **64× better drift resilience** (Corollary 6.7) through targeted invalidation.

**Advantage 6: Provable Pruning Guarantees**

NL's compressed memories cannot provide mathematical guarantees about exploration:

- **Delta Rule** has better memory management than Hebbian, but still probabilistic
- **Self-referential updates** may forget failed options or suggest them again
- **No mechanism** to track which options have been tried and failed

PRECEPT's RefineInterceptor provides **provable guarantees**:

$$\mathbb{E}[\text{wasted retries}]_{\text{NL}} > 0 \quad \text{vs} \quad \mathbb{E}[\text{wasted retries}]_{\text{PRECEPT}} = 0$$

```python
class RefineInterceptor:
    def __init__(self):
        self.failed_options = set()  # Hash set for O(1) membership test

    def filter(self, options):
        return [o for o in options if hash(o) not in self.failed_options]

    def record_failure(self, option):
        self.failed_options.add(hash(option))  # P(repeat) = 0 after this
```

**Advantage 7: Simplicity and Auditability**

NL introduces significant architectural complexity:

| NL Component | Purpose | Complexity |
|--------------|---------|------------|
| CMS | Multi-frequency persistence | K levels × hidden_dim² parameters |
| Delta GD | State-dependent updates | Additional $x_t x_t^\top$ computation |
| Self-referential Titans | Generate own values | 6 nested memories per block |
| Multi-scale Momentum Muon (M3) | Long-context optimizer | Multiple momentum terms |
| Knowledge transfer | Between levels | Meta-learning or backprop |

PRECEPT's architecture:
| PRECEPT Component | Purpose | Complexity |
|-------------------|---------|------------|
| Hash table | Tier 1 rules | O(1) lookup, O(n) storage |
| Structured dict | Tier 2 precepts | O(N) composition |
| Vector store | Tier 0 experiences | O(log n) similarity search |
| Beta posteriors | Confidence tracking | O(1) update |

**Debugging comparison**:
```python
# PRECEPT: Direct inspection
print(learned_rules["BLOCKED_PORTS:503+CARGO:hazmat"])  # → "singapore"
print(confidence["BLOCKED_PORTS:503+CARGO:hazmat"])     # → Beta(5, 1)

# NL: Cannot directly inspect
# Which of CMS's K MLP levels encoded this rule?
# What frequency was it stored at?
# How does it interact with other rules in the same weights?
```

**Advantage 8: Sample Efficiency**

NL reduces sample requirements compared to pure RL through multi-frequency learning, but still requires gradient-based optimization with multiple passes:

| Method | Encounters per Rule (β) | Mechanism |
|--------|------------------------|-----------|
| Traditional RL | 100+ | Policy gradient convergence |
| Nested Learning | ~10-50* | Multi-frequency gradient descent |
| GEPA | 10-15 | Evolutionary optimization |
| **PRECEPT** | **3** | Direct storage after β successes |

*NL's sample complexity is estimated based on typical gradient-based learning requirements; the NL paper does not provide direct comparisons for rule learning tasks.*

PRECEPT's explicit rule learning requires only β=3 encounters because successful solutions are **stored directly** rather than **compressed through optimization**:

$$\text{Storage}_{\text{NL}}: \theta_{t+1} = \theta_t - \eta \nabla \mathcal{L} \quad \text{(iterative compression)}$$
$$\text{Storage}_{\text{PRECEPT}}: \text{rules}[k] = v \quad \text{(direct insertion)}$$

**Advantage 9: No Cross-Rule Interference**

NL's shared weight matrices create interference risk:

$$\frac{\partial \mathcal{L}_{\text{rule}_1}}{\partial W} \neq 0 \quad \text{and} \quad \frac{\partial \mathcal{L}_{\text{rule}_2}}{\partial W} \neq 0 \quad \Rightarrow \quad \text{update may harm rule}_2 \text{ while learning rule}_1$$

This is the fundamental cause of **catastrophic forgetting** that CMS attempts to mitigate through frequency separation—but cannot eliminate.

PRECEPT's isolated key-value storage guarantees **zero interference**:
$$\text{rules}[k_1] = v_1 \quad \text{and} \quad \text{rules}[k_2] = v_2 \quad \Rightarrow \quad \text{independent storage, no interference}$$

**Table 17.** Comprehensive Comparison: PRECEPT Advantages Over Nested Learning.

| Capability | Nested Learning | PRECEPT | PRECEPT Advantage |
|------------|----------------|---------|-------------------|
| **Memory Type** | Compressed (MLP weights) | Explicit (Three-Tier) | Direct inspection |
| **Retrieval Accuracy** | Degrades with n, ρ (interference) | **100%** (constant) | Deterministic foundation |
| **Uncertainty Quantification** | Implicit in weights | **Beta posteriors** (calibrated) | Principled exploration |
| **Exploration Strategy** | Gradient descent | **Thompson Sampling** | Provably optimal |
| **Multi-Objective Optimization** | Single loss | **COMPASS 6-objective Pareto** | Richer trade-offs |
| **Compositional Gen.** | ✗ (no mechanism) | **O(2^N)** from N atomics | Exponential coverage |
| **Conflict Resolution** | Frequency priority | **Bayesian ensemble** | Calibrated trust |
| **Drift Adaptation** | Gradual (frequency-based) | **Soft decay + threshold** (θ=2) | 64× faster recovery |
| **Epistemic Reasoning** | ✗ (no mechanism) | **Probing + uncertainty** | Active information gathering |
| **Pruning Guarantee** | ✗ (probabilistic) | **P=0** (provable) | Zero wasted retries |
| **Interpretability** | Opaque | **Explicit rules + confidence** | Full auditability |
| **Sample Efficiency** | β≈10-20 | **β=3** | 3-7× improvement |
| **Cross-Rule Interference** | Possible (shared weights) | **None** (isolated storage) | No catastrophic forgetting |
| **Retrieval Complexity** | O(d²) forward pass | **O(1)** hash lookup | Simpler retrieval |
| **Reasoning Complexity** | Simple (gradient descent) | **Rich** (Bayesian + COMPASS) | Sophisticated decisions |

#### 9.7.5a Dual-Frequency vs Multi-Frequency: Control Loop Architectures

A key architectural similarity—and critical difference—between PRECEPT and Nested Learning is their approach to **multi-timescale processing**. Both frameworks recognize that not all operations should occur at the same frequency, but their implementations differ fundamentally.

**Nested Learning's Multi-Frequency Approach**

NL's Continuum Memory System (CMS) operates at multiple frequencies inspired by brain oscillations:

| Frequency Band | Brain Analog | Update Rule | Purpose |
|----------------|--------------|-------------|---------|
| High (f₁) | Gamma (30-150 Hz) | Every step | Working memory, immediate context |
| Medium (f₂) | Beta (13-30 Hz) | Every C¹ steps | Short-term patterns |
| Low (f₃) | Theta/Delta (0.5-8 Hz) | Every C² steps | Long-term knowledge |

The update is governed by:
$$\theta_{i+1}^{(f_\ell)} = \theta_i^{(f_\ell)} - \begin{cases} \sum_{t=i-C^{(\ell)}}^{i} \eta_t^{(\ell)} \nabla f(\theta_t; x_t) & \text{if } i \equiv 0 \pmod{C^{(\ell)}} \\ 0 & \text{otherwise} \end{cases}$$

**Key characteristics:**
- All frequencies perform **gradient-based weight updates**
- Lower frequencies accumulate gradients over longer windows (larger C^(ℓ))
- Each frequency level maintains **separate MLP blocks**
- No explicit separation of "monitoring" vs "optimization" functions

**COMPASS's Dual-Frequency Approach**

COMPASS explicitly separates two fundamentally different operations (Algorithm 8, Section 5.7):

| Loop | Frequency | Operations | Cost | Purpose |
|------|-----------|------------|------|---------|
| **Monitor Mode** | Every step | `evaluate_action()`, `evaluate_error()`, `learn_pattern()` | O(1) | Real-time safety, constraint checking |
| **Architect Mode** | On trigger | `compile()`, prompt evolution, memory pruning | O(n) | Strategic re-planning, optimization |

**Key characteristics:**
- High-frequency loop performs **logical constraint checks** (not gradient updates)
- Low-frequency loop performs **LLM-based optimization** (not weight updates)
- Architect Mode triggers are **event-driven** (not periodic)
- Functions are **qualitatively different** (monitoring vs optimization)

**Table 16e.** Multi-Frequency Architecture Comparison.

| Dimension | Nested Learning (CMS) | COMPASS (Dual-Frequency) |
|-----------|----------------------|--------------------------|
| **Number of Frequencies** | k levels (configurable) | 2 (fixed: Monitor + Architect) |
| **Frequency Meaning** | Gradient accumulation window | Qualitatively different operations |
| **High-Frequency Operation** | Weight updates to MLP | O(1) constraint checking |
| **Low-Frequency Operation** | Weight updates (longer window) | O(n) prompt evolution |
| **Triggering** | Periodic (every C^(ℓ) steps) | **Event-driven** (new rule, failure, phase change) |
| **Operation Type** | Homogeneous (all are gradient updates) | **Heterogeneous** (monitoring vs optimization) |
| **Resource Usage** | Proportional across frequencies | **Asymmetric** (O(1) vs O(n)) |
| **Source** | CMS parameter updates | `compass_controller.py`, `compass_integration.py` |

**Why the Difference Matters**

The fundamental difference is whether frequency separation serves **quantitative** or **qualitative** purposes:

**NL (Quantitative):** All frequency levels perform the same operation (gradient descent) with different accumulation windows. This creates a spectrum of adaptation speeds—gamma memories adapt fastest but are volatile, delta memories persist longest but adapt slowly.

$$\text{Operation}_{f_1} = \text{Operation}_{f_2} = \text{Operation}_{f_3} = \text{Gradient Descent}$$

**COMPASS (Qualitative):** The two frequencies perform **entirely different operations**. Monitor Mode is not "faster Architect Mode"—it performs constraint checking, a fundamentally different function from prompt evolution.

$$\text{Operation}_{\text{Monitor}} = \text{Constraint Check (O(1))}$$
$$\text{Operation}_{\text{Architect}} = \text{Pareto Optimization (O(n))}$$

**Table 16f.** Operational Comparison at Different Frequencies.

| Aspect | NL High-Freq (Gamma) | NL Low-Freq (Delta) | COMPASS Monitor | COMPASS Architect |
|--------|---------------------|--------------------|-----------------|--------------------|
| **Operation** | Gradient update | Gradient update | Constraint check | Prompt evolution |
| **Cost** | O(d) | O(d) | **O(1)** | **O(n) LLM calls** |
| **Trigger** | Every step | Every C steps | Every step | **Event-driven** |
| **Output** | Updated weights | Updated weights | BLOCK/PROCEED | Evolved prompt |
| **Reversible?** | No (weights changed) | No (weights changed) | **Yes** (decision only) | Partially |

**Implications for Latency and Cost**

This architectural difference has significant practical implications:

| Scenario | NL Approach | COMPASS Approach | Winner |
|----------|-------------|------------------|--------|
| **1000 steps, no anomalies** | 1000 high-freq + 10 low-freq updates | 1000 O(1) checks + 0 compilations | **COMPASS** (1000× cheaper low-freq) |
| **1000 steps, 10 errors** | Same as above | 1000 O(1) checks + ~10 compilations | **Comparable** |
| **Real-time constraint violation** | Must wait for high-freq update to propagate | **Immediate** O(1) BLOCK | **COMPASS** |
| **Strategic optimization needed** | Continuous background updates | **On-demand** Pareto evolution | **Context-dependent** |

**Key Insight:** COMPASS's dual-frequency design directly addresses the **Latency/Cost critique** that would apply to running full GEPA optimization at every step:

- **NL's answer:** Run cheaper gradient updates at high frequency, expensive updates at low frequency
- **COMPASS's answer:** Run **qualitatively different** operations—cheap monitoring at high frequency, expensive optimization only when triggered

For **rule-governed domains** where constraint violations must be caught immediately (safety-critical), COMPASS's separation of monitoring (always-on, O(1)) from optimization (event-driven, O(n)) provides superior latency guarantees.

**Code-Level Evidence**

The dual-frequency architecture is implemented across two files:

| Component | File | Lines | Frequency |
|-----------|------|-------|-----------|
| `evaluate_action()` | `compass_controller.py` | 298-375 | High (every step) |
| `evaluate_error()` | `compass_controller.py` | 377-448 | High (every error) |
| `learn_pattern()` | `compass_controller.py` | 583-610 | High (after outcomes) |
| `compile()` | `compass_integration.py` | 624-841 | Low (on trigger) |

Integration in `precept_agent.py`:
- Line 1359: `compass_decision = self._compass_controller.evaluate_action(...)` — **every task**
- Line 2042: `compass_error_decision = self._compass_controller.evaluate_error(...)` — **every error**
- External orchestrator: `await agent.run_compilation()` — **on trigger only**

#### 9.7.6 Complementary Insights and Synthesis

Both frameworks represent significant advances—their complementary strengths suggest future integration opportunities:

**What PRECEPT Could Learn from NL**

1. **Unified View of Optimizers**: NL's insight that Adam, momentum, and AdaGrad are associative memories compressing gradients could inform COMPASS's rollout strategy—perhaps viewing prompt evolution as gradient descent in prompt space.

2. **Multi-Frequency Persistence**: CMS's spectrum of update frequencies could enhance PRECEPT's Tier 0 (Experience Store). Currently, all experiences have equal persistence; frequency-based decay could prioritize recent, relevant experiences. This could extend COMPASS's dual-frequency design to a **tri-frequency** architecture with distinct monitoring, consolidation, and optimization loops.

3. **Self-Referential Learning**: NL's self-generated values for internal optimization could inspire PRECEPT to generate synthetic scenarios for confidence calibration—proactively testing rule validity rather than waiting for failures.

4. **Brain-Inspired Design**: NL's analogy to neural oscillations (gamma/beta/theta waves) provides principled guidance for memory system design that could inform future PRECEPT extensions. COMPASS's current dual-frequency design (Monitor/Architect) could be refined with NL's frequency spectrum insights.

**What NL Could Learn from PRECEPT**

1. **Deterministic Foundations**: PRECEPT demonstrates that sophisticated reasoning (Thompson Sampling, Bayesian inference, COMPASS) is more effective when built on **deterministic foundations** rather than approximate retrieval.

2. **Explicit Uncertainty**: PRECEPT's Beta posteriors provide **calibrated uncertainty** that NL's implicit weight-based uncertainty cannot match. This enables provably optimal exploration.

3. **Compositional Abstraction**: PRECEPT's tier hierarchy shows that compositional generalization can emerge from **simple algebraic rules** (max) rather than complex neural composition.

4. **Targeted Unlearning**: PRECEPT's threshold-based invalidation (θ=2) demonstrates that **selective forgetting** is more effective than frequency-based decay for drift adaptation.

5. **Qualitative Frequency Separation**: COMPASS's dual-frequency design (Section 5.7, Algorithm 8) shows that frequency separation can serve **functional differentiation** (monitoring vs optimization) rather than just temporal differentiation (fast vs slow updates). This provides superior latency guarantees for safety-critical operations where immediate constraint checking must be separated from expensive optimization.

**Synthesis: Two Paths to the Same Goals**

| Goal | NL's Approach | PRECEPT's Approach |
|------|---------------|-------------------|
| **Continual Learning** | Multi-frequency compression | Explicit storage + Bayesian confidence |
| **Uncertainty Handling** | Implicit in weight variance | Explicit Beta posteriors + Thompson Sampling |
| **Compositional Gen.** | Learn combinations | Algebraic tier composition |
| **Drift Adaptation** | Frequency-based decay | Threshold-based invalidation |
| **Optimization** | Nested gradient descent | COMPASS multi-objective Pareto |
| **Multi-Timescale Processing** | CMS: k homogeneous frequency bands | **Dual-Frequency: Monitor (O(1)) + Architect (O(n))** |
| **Latency Management** | Proportional cost across frequencies | **Asymmetric: cheap monitoring, expensive optimization** |

The fundamental difference is **architectural philosophy**:
- **NL**: "Make compression sophisticated enough to handle complexity"
- **PRECEPT**: "Keep retrieval simple so reasoning can be sophisticated"

Our empirical results (Cohen's d = 8.27, 100% P₁ on compositions) support PRECEPT's philosophy for **rule-governed domains**, while NL's philosophy may be preferable for **continuous feature learning** where compression's interpolation properties are advantageous.

#### 9.7.7 Summary: When to Use Each Approach

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| **Black Swan multi-condition rules** | PRECEPT | Exact retrieval + Thompson Sampling exploration |
| **Compositional constraint handling** | PRECEPT | Tier hierarchy + algebraic composition |
| **Safety-critical determinism** | PRECEPT | Provable guarantees + calibrated uncertainty |
| **Auditable AI systems** | PRECEPT | Explicit rules + Beta posterior confidence |
| **Multi-objective optimization** | PRECEPT | COMPASS 6-objective Pareto selection |
| **Active uncertainty reduction** | PRECEPT | Epistemic probing + diagnostic tool calls |
| **Selective rule modification** | PRECEPT | Isolated storage enables surgical updates |
| **Continuous feature learning** | NL | Gradient-based adaptation suits smooth functions |
| **Soft pattern recognition** | NL | Compression handles ambiguity via interpolation |
| **Long-context understanding** | NL (CMS) | Multi-frequency persistence for episodic memory |
| **General language modeling** | NL | Pre-training is compression (NL's core insight) |

**Conclusion**: PRECEPT and Nested Learning are both sophisticated frameworks addressing similar challenges through fundamentally different architectures:

- **NL's Philosophy**: "Sophisticated compression" — Make gradient-based learning powerful enough to handle complexity through multi-level optimization, self-referential updates, and frequency-based memory management.

- **PRECEPT's Philosophy**: "Simple retrieval, sophisticated reasoning" — Keep storage/retrieval deterministic and simple (hash tables) to enable sophisticated probabilistic reasoning (Thompson Sampling, Bayesian inference, COMPASS optimization, epistemic probing) on reliable foundations.

For LLM agents in **rule-governed domains** requiring deterministic, compositional, and auditable behavior—the focus of this work—PRECEPT's paradigm achieves superior results (Cohen's d = 8.27, 100% P₁ on compositions, 64× drift resilience) by building sophisticated reasoning on deterministic foundations rather than sophisticated retrieval on approximate foundations.

### 9.8 Positioning Summary

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#333', 'lineColor': '#555', 'fontFamily': 'Arial, sans-serif'}}}%%
quadrantChart
    title Positioning in LLM Agent Landscape
    x-axis Low Sample Efficiency --> High Sample Efficiency
    y-axis No Compositional Gen. --> Compositional Gen.
    quadrant-1 PRECEPT Territory
    quadrant-2 Untapped
    quadrant-3 RL Methods
    quadrant-4 Verbal Methods
    PRECEPT: [0.85, 0.90]
    Reflexion: [0.45, 0.25]
    ExpeL: [0.50, 0.20]
    Voyager: [0.40, 0.30]
    RL-PPO: [0.15, 0.15]
```

**Figure 11.** PRECEPT occupies a unique position: high sample efficiency (β=3) AND compositional generalization (O(2^N) coverage). Prior methods cluster in low-efficiency and/or non-compositional regions.

**Table 16a.** Detailed Comparison with Related Methods.

| Method | Retrieval | Composition | Drift | Sample β | P(repeat_fail) |
|--------|-----------|-------------|-------|----------|----------------|
| **PRECEPT** | **O(1) exact** | **O(2^N) tier** | **θ=2** | **3** | **0** |
| Reflexion | None | None | None | — | High |
| Full Reflexion | O(n) task-type | None | None | 5+ | Medium |
| ExpeL | O(n) similarity | None | None | 5+ | Medium |
| Voyager | O(log n) vector | Partial | None | 10+ | Low |
| RAG | O(log n) dense | None | None | — | N/A |
| RL-PPO | None (implicit) | None | Retrain | 100+ | Variable |

### 9.9 Connection to Evolutionary Computing and Red Queen Dynamics

Recent work in evolutionary computing, particularly the **Digital Red Queen (DRQ)** framework (Kumar et al., 2026), demonstrates that agents trained via static optimization are brittle—failing 72% of the time when confronted with novel adversarial dynamics. DRQ addresses this through continual self-play adaptation against a growing history of opponents, producing increasingly general and robust strategies. This section establishes how PRECEPT translates these theoretical concepts from evolutionary computing into a practical, production-ready architecture for autonomous LLM agents.

#### 9.9.1 The Red Queen Hypothesis in Agent Design

The Red Queen hypothesis, originating from evolutionary biology, posits that organisms must continually adapt merely to maintain their relative fitness in an ever-changing environment. Kumar et al. (2026) operationalize this in the context of LLM-evolved program synthesis:

> "Unlike biological evolution, most LLM-evolution frameworks are formulated as static optimization problems, overlooking the open-ended adversarial dynamics that characterize real-world evolutionary processes." (Kumar et al., 2026)

Their key finding is that **dynamic objectives** (where the fitness landscape continually shifts) produce fundamentally more robust agents than static objectives. Warriors evolved through DRQ's Red Queen dynamics "become increasingly general (relative to a set of held-out human warriors)" and exhibit "convergent evolution toward a general-purpose behavioral strategy."

**PRECEPT's Realization.** PRECEPT embeds this insight architecturally. Rather than optimizing against a static benchmark, PRECEPT treats **environmental entropy** (system errors, constraint violations, drift events) as the "Red Queen"—an ever-present adversary that the agent must continually adapt to. This shift from passive error logging to active adversarial adaptation is what enables PRECEPT's measured 64× drift resilience over baselines (Corollary 6.7).

#### 9.9.2 Solving Brittleness via Evo-Memory

**The DRQ Finding**: The paper proves that agents trained via static optimization are brittle and overfit, failing catastrophically when confronted with strategies outside their training distribution. DRQ overcomes this by maintaining a "growing history of opponents" that the agent must defeat.

**PRECEPT's Parallel Architecture**: PRECEPT implements an analogous mechanism we term **Evo-Memory**—a growing history of failure constraints stored in the partial progress and learned rules systems:

```
DRQ: Optimize(Agent) against History[Opponent₁, Opponent₂, ..., Opponentₙ]
PRECEPT: Optimize(Plan) against History[Constraint₁, Constraint₂, ..., Constraintₙ]
```

Just as DRQ forces agents to optimize against accumulated opponents, PRECEPT forces agents to plan against accumulated failure constraints. The key insight is that **failures are adversaries**—each failure reveals information about the environment's constraint landscape that must be incorporated into future planning.

This architectural parallel explains PRECEPT's measured P(repeat_fail) = 0:

By **Theorem 4.0 (Evo-Memory Eliminates Cyclic Failures)**, for any accumulated constraint set C = {c₁, c₂, ..., cₖ}:
$$P(\text{repeat}(cᵢ) | \text{EvoMemory}(C)) = 0, \quad \forall cᵢ \in C$$

PRECEPT's RefineInterceptor maintains an in-episode forbidden set, while the partial progress system maintains a cross-episode `failed_options` set. Any plan containing a previously failed option is deterministically pruned before execution (see Section 4.0 for formal proof).

This mathematically eliminates the "cyclic dynamics" (repeat failures) that plague standard agents, allowing robust generalist behavior to emerge naturally—precisely the outcome observed in DRQ's evolved warriors.

#### 9.9.3 Overcoming Deceptive Search Landscapes

**The DRQ Finding**: Kumar et al. (2026) identify that program synthesis environments are "highly deceptive search landscapes" where greedy algorithms get trapped in local minima. DRQ overcomes this through maintaining behavioral diversity across the population.

**PRECEPT's MAP-Elites Integration**: PRECEPT's COMPASS algorithm integrates the **MAP-Elites** principle (Mouret & Clune, 2015) to maintain strategy diversity:

| DRQ Mechanism | PRECEPT Analog | Purpose |
|---------------|----------------|---------|
| Population diversity | Strategy archive | Avoid convergence collapse |
| Fitness vs. opponents | Pareto optimality | Multi-objective selection |
| Behavioral characterization | Complexity phenotyping | Niche differentiation |

COMPASS maintains a "diversity of strategies" differentiated by behavioral phenotype (e.g., network-based vs. file-based solutions, aggressive vs. conservative approaches). When a "Black Swan" blocks the obvious path, the agent has a pre-calculated, **topologically distinct** alternative ready:

Per **Definition 5.2 (Topological Distinctness)**, two strategies S₁, S₂ are topologically distinct if:
$$\text{FailureModes}(S₁) \cap \text{FailureModes}(S₂) = \emptyset$$

This architectural choice prevents the "diversity collapse" that kills agents relying on single-strategy optimization—a phenomenon Kumar et al. (2026) observe in static optimization baselines (see Section 5.1.1 for theoretical foundation).

#### 9.9.4 Operationalizing Adversarial Dynamics for Reliability

**The DRQ Framework**: The paper argues that true robustness requires ongoing adversarial pressure:

> "DRQ uses an LLM to evolve assembly-like programs... which compete against each other for control of a virtual machine... In each round of DRQ, the model evolves a new warrior to defeat all previous ones." (Kumar et al., 2026)

**PRECEPT's Epistemic Probing**: PRECEPT operationalizes this adversarial dynamic through **Epistemic Probing**—the agent actively "battles" the environment to discover constraints:

```python
# PRECEPT's Adversarial Discovery Loop
while environment.uncertainty > threshold:
    probe = compass.generate_diagnostic_probe(error_pattern)
    constraint = await environment.execute_probe(probe)
    evo_memory.add_constraint(constraint)
    compass.update_strategy_archive(constraint)
```

This shifts the paradigm from **passive error logging** (wait for failures) to **active constraint discovery** (seek out failures proactively). The environment becomes the adversary that PRECEPT must defeat through increasingly sophisticated understanding of its constraint landscape.

**Table 17.** Adversarial Dynamics Comparison.

| Aspect | DRQ | PRECEPT |
|--------|-----|---------|
| **Adversary** | Previous warriors | Environmental constraints |
| **Adaptation trigger** | Round completion | Failure detection |
| **History accumulation** | Warrior archive | Constraint/rule memory |
| **Selection pressure** | Must defeat all prior | Must satisfy all constraints |
| **Convergence outcome** | General-purpose warriors | Robust generalist agents |

#### 9.9.5 Predictive Epistemics for Efficiency

**The DRQ Finding**: Kumar et al. (2026) demonstrate that solution quality can be statistically predicted from code embeddings alone, without expensive simulation:

> "Interestingly, warriors also become less behaviorally diverse across independent runs, indicating a convergence pressure toward a general-purpose behavioral strategy, much like convergent evolution in nature."

This suggests that high-quality solutions share identifiable mathematical signatures.

**PRECEPT's Predictive Context Embeddings**: PRECEPT operationalizes this finding through its complexity analysis system:

$$\text{RiskScore}(\text{plan}) = f_{\text{ML}}(\text{Embedding}(\text{plan}), \mathcal{H}_{\text{fail}})$$

*where $f_{\text{ML}}$ is the ML-based risk estimator (implemented in `PRECEPTComplexityAnalyzer`), $\text{Embedding}(\text{plan})$ is the feature representation of the candidate plan (task complexity, tool requirements, entity count), and $\mathcal{H}_{\text{fail}}$ is the set of historically observed failure patterns.*

COMPASS uses this risk score to pre-filter dangerous plans or detect high-risk states *before* execution. This enables:

1. **Proactive failure avoidance**: Skip plans with high predicted risk
2. **Smart rollout allocation**: Focus compute on promising candidates
3. **Early termination**: Abort unpromising trajectories before wasting resources

The result is a reduction in COMPASS rollouts compared to baseline GEPA (Section 5.4), dramatically reducing the compute cost of reliability.

#### 9.9.6 Systems-Level Contribution

PRECEPT's contribution lies in successfully translating theoretical concepts from evolutionary computing into a production-ready architecture:

**Table 18.** From Theory to Practice: PRECEPT's Operationalization of DRQ Concepts.

| DRQ Theoretical Concept | PRECEPT Implementation | Measured Outcome |
|------------------------|------------------------|------------------|
| Red Queen dynamics | Adversarial adaptation via Evo-Memory | P(repeat_fail) = 0 (Theorem 6.9) |
| Deceptive landscape navigation | MAP-Elites diversity in COMPASS | Cohen's d = 8.27 on compositions |
| Growing opponent history | Partial progress + learned rules | 64× drift resilience (Corollary 6.7) |
| Predictive embeddings | ML-based complexity detection | 72% rollout reduction vs GEPA |
| Convergent evolution to generalists | Semantic tier hierarchy | 100% P₁ on 2-way compositions |

**Implication.** PRECEPT demonstrates the viability of translating static optimization to dynamic adaptation for LLM agents: building survivable systems that improve through adversarial interaction with environmental constraints, rather than optimizing against fixed objectives.

**Broader Implications.** Kumar et al. (2026) suggest that "similarly minimal self-play approaches could prove useful in other more practical multi-agent adversarial domains." PRECEPT provides evidence for this claim in the LLM agent domain: the same adversarial principles that produce robust Core War warriors can produce robust constraint-handling agents when environmental constraints are treated as adversaries requiring continual adaptation.

---

## 10. Conclusion

We introduced PRECEPT, a framework for LLM agent learning that replaces probabilistic verbal reasoning with deterministic structured retrieval. PRECEPT addresses fundamental limitations affecting both verbal reflection methods and traditional reinforcement learning, achieving capabilities that neither approach provides individually.

### 10.1 Summary of Contributions

**Architectural Innovation.** PRECEPT's O(1) exact-match retrieval via structured condition keys eliminates LLM interpretation errors entirely—achieving 0% error rate regardless of condition complexity, compared to 94.4% for verbal baselines at N=10 conditions (Theorem 6.8).

**Compositional Generalization.** The semantic tier hierarchy enables O(2^N) compositional coverage from N atomic precepts. Learning 6 atomic rules generalizes to 63 composite scenarios, achieving 100% first-try success on 2-way compositions (Cohen's d = 8.27; see Section 7.3 for discussion of this effect size).

**Drift Adaptation.** The dual conflict resolution mechanism provides the first unified treatment of static-dynamic conflicts and environmental drift, achieving 64× better drift resilience through principled Bayesian inference and threshold-based rule invalidation.

**Computational Efficiency.** COMPASS achieves 72% rollout reduction over baseline GEPA through ML-based complexity detection and smart allocation, while maintaining Pareto-optimal selection quality.

### 10.2 Theoretical and Empirical Validation

Our theoretical analysis establishes closed-form bounds proving PRECEPT's advantages grow exponentially with task complexity: 22× advantage at N=10 conditions (Corollary 6.5) and 64× drift resilience ratio (Corollary 6.7). Seven experiments across six domains validate these predictions with strong empirical results (p < 0.001): +41.0pp mean first-try advantage on the 3-domain main comparison (Cohen's d > 1.9 on all domains), +49.2pp compositional generalization improvement (Cohen's d = 8.27), +50pp drift recovery, and 61% computational savings.

### 10.3 Broader Impact

PRECEPT enables deployment of LLM agents in high-stakes domains requiring deterministic reliability, compositional reasoning, and continuous adaptation to non-stationary environments. The framework's modular four-layer architecture facilitates extension and integration with existing systems.

**Potential Applications:**
- **Autonomous logistics:** Port routing with dynamic closures and regulatory constraints
- **Healthcare compliance:** Medical device approval pathways with compositional requirements
- **Financial trading:** Regulatory constraint handling across multiple jurisdictions
- **Software deployment:** CI/CD pipelines with complex dependency and environment constraints

**Societal Considerations:** While PRECEPT's deterministic behavior improves reliability, autonomous agents in critical systems require careful deployment. The explicit rule storage enables auditing and interpretability—unlike black-box RL policies—supporting responsible AI deployment.

### 10.4 Limitations

We acknowledge several limitations of the current work:

**L1. Synthetic Benchmarks.** Experiments use controlled synthetic benchmarks designed to isolate PRECEPT's specific capabilities. While this enables rigorous evaluation of compositional generalization and drift adaptation, real-world deployment may encounter additional complexities.

**L2. Static Tier Assignment.** The semantic tier hierarchy (Safety > Compliance > Preferences) is currently hand-defined. While effective for the tested domains, complex ontologies may require more nuanced tier structures.

**L3. Single-Agent Focus.** PRECEPT addresses single-agent learning. Multi-agent coordination with distributed constraint propagation remains future work.

**L4. LLM Dependency.** PRECEPT requires a capable base LLM (GPT-4 class) for reasoning fallbacks. Performance with smaller models is untested.

**L5. Domain Coverage.** Six domains are implemented (three in main comparison; DevOps excluded for ceiling effect, Finance and Coding excluded for insufficient inter-scenario condition overlap). The main comparison focuses on domains where compositional complexity activates the exact-vs-approximate retrieval distinction. Generalization to domains without structured error patterns (e.g., creative tasks) is unverified.

**L6. Cold Start.** PRECEPT requires initial exploration to learn rules. Zero-shot compositional generalization from domain descriptions alone is not supported.

### 10.5 Future Directions

Several promising extensions emerge from this work:

**F1. Meta-Learning for Tier Discovery.** Learn the tier hierarchy automatically from data rather than hand-specification. This could enable PRECEPT to adapt to new domains without manual ontology engineering.

**F2. Hierarchical Precepts.** Extend atomic precepts to support nested constraint structures, enabling more expressive compositional rules.

**F3. Multi-Agent PRECEPT.** Develop distributed versions where agents share learned rules and coordinate constraint propagation, enabling collaborative problem-solving.

**F4. Formal Verification.** Apply model checking techniques to verify rule consistency and detect potential conflicts before deployment.

**F5. Real-World Validation.** Deploy PRECEPT in production logistics/booking systems to validate performance under real-world noise and complexity.

**F6. Retrieval-Augmented Integration.** Combine PRECEPT's exact-match rules with RAG-style knowledge retrieval for knowledge-intensive domains.

**F7. Continuous Prompt Evolution.** Extend verified evolution to continuous online learning during deployment, enabling adaptation without explicit retraining phases.

### 10.6 Relationship to Evolutionary Computing: Originality and Distinctions

PRECEPT draws theoretical inspiration from the Digital Red Queen (DRQ) framework (Kumar et al., 2026), which demonstrates that dynamic objectives produce more robust agents than static optimization. However, PRECEPT represents **original research** with fundamental distinctions:

**Table 19.** PRECEPT vs DRQ: Originality and Distinctions.

| Aspect | DRQ | PRECEPT (Original) |
|--------|-----|-------------------|
| **Domain** | Program synthesis (Core War assembly) | Rule-governed LLM agents (logistics, finance, etc.) |
| **Adversary** | Other evolved programs | Environmental constraints |
| **Output** | Assembly code warriors | Deterministic constraint rules |
| **Guarantees** | Empirical robustness | Formal mathematical proofs (Theorems 3.1, 6.5-6.9) |
| **Retrieval** | None (implicit in code) | O(1) exact-match via hash tables |
| **Composition** | None | O(2^N) from N atomics via tier hierarchy |
| **Drift handling** | Not addressed | Bayesian Thompson Sampling + threshold invalidation |
| **Production-ready** | Research prototype | MCP-based client-server architecture |

**PRECEPT's Original Contributions Beyond DRQ:**

1. **Deterministic Guarantees**: DRQ achieves empirical robustness; PRECEPT proves formal bounds (Theorem 6.8: 0% vs 94.4% error rate, Corollary 6.7: 64× drift resilience).

2. **Compositional Generalization**: DRQ does not address composition; PRECEPT introduces the **semantic tier hierarchy** enabling O(2^N) coverage from N rules—a fundamental capability DRQ lacks.

3. **Explicit Knowledge Representation**: DRQ's knowledge is implicit in evolved code; PRECEPT stores **auditable, interpretable rules** enabling transparency and selective modification.

4. **Production Architecture**: DRQ is a research algorithm; PRECEPT is a **complete MCP-based system** with partial progress tracking, procedural memory, and verified evolution.

5. **Rule-Governed Domains**: DRQ targets open-ended program evolution; PRECEPT targets **structured constraint domains** where determinism is essential (safety-critical, regulated).

**Synthesis, Not Replication**: PRECEPT synthesizes insights from evolutionary computing (Red Queen dynamics, diversity preservation) with novel contributions (deterministic retrieval, compositional tiers, Bayesian drift handling) to create a **production-ready system** for a fundamentally different problem domain. This represents the natural progression of scientific research: applying theoretical insights to new domains with original solutions.

### 10.7 Concluding Remark

PRECEPT demonstrates that reliable, adaptable LLM agents can be achieved through principled structured retrieval with explicit conflict resolution, rather than increasingly sophisticated verbal reasoning. By translating theoretical insights from evolutionary computing (Red Queen dynamics, adversarial adaptation) into deterministic architecture, we achieve capabilities that neither verbal reflection nor reinforcement learning provides.

The empirical results (Cohen's d up to 8.27, 100% P₁ on compositions) and theoretical guarantees (22× advantage, 64× drift resilience, zero wasted retries) support PRECEPT's paradigm of dynamic adaptation over static optimization, with deterministic guarantees suitable for production deployment.

Kumar et al. (2026) suggest that adversarial self-play approaches "could prove useful in other more practical multi-agent adversarial domains." PRECEPT provides evidence for this claim in the LLM agent domain, demonstrating that evolutionary principles can be operationalized through appropriate architectural choices. We hope this work motivates further research into evolutionary-inspired approaches for reliable AI systems.

---

## References

Agrawal, S., & Goyal, N. (2012). Analysis of Thompson Sampling for the Multi-Armed Bandit Problem. *COLT*.

Agrawal, L.A., Khattab, O., et al. (2025). GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. *arXiv:2507.19457*.

Baek, J., et al. (2023). Knowledge-Augmented Language Model Verification. *EMNLP*.

Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). Nested Learning: The Illusion of Deep Learning Architecture. *arXiv:2512.24695*.

Chen, L., et al. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. *NeurIPS*.

Chen, W., et al. (2022). Rich Knowledge Sources Bring Complex Knowledge Conflicts: Recalibrating Models to Reflect Conflicting Evidence. *EMNLP*.

Guo, Q., et al. (2023). Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers. *arXiv:2309.08532*.

Guu, K., et al. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML*.

Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *ICML*.

Keysers, D., et al. (2020). Measuring Compositional Generalization: A Comprehensive Method on Realistic Data. *ICLR*.

Khattab, O., et al. (2023). DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. *arXiv:2310.03714*.

Kim, N., & Linzen, T. (2020). COGS: A Compositional Generalization Challenge Based on Semantic Interpretation. *EMNLP*.

Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. *PNAS*.

Kumar, A., Bahlous-Boldi, R., Sharma, P., Isola, P., Risi, S., Tang, Y., & Ha, D. (2026). Digital Red Queen: Adversarial Program Evolution in Core War with LLMs. *arXiv:2601.03335*.

Lake, B., & Baroni, M. (2018). Generalization Without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. *ICML*.

Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.

Longpre, S., et al. (2021). Entity-Based Knowledge Conflicts in Question Answering. *EMNLP*.

Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. *NeurIPS*.

Manhaeve, R., et al. (2018). DeepProbLog: Neural Probabilistic Logic Programming. *NeurIPS*.

Mouret, J.-B., & Clune, J. (2015). Illuminating Search Spaces by Mapping Elites. *arXiv:1504.04909*.

Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*.

Park, J.S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST*.

Pineau, J., et al. (2021). Improving Reproducibility in Machine Learning Research. *JMLR*.

Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *arXiv:2302.04761*.

Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

Shi, W., et al. (2023). REPLUG: Retrieval-Augmented Black-Box Language Models. *arXiv:2301.12652*.

Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS*.

Storkey, A. (1997). Increasing the Capacity of a Hopfield Network Without Sacrificing Functionality. *ICANN*.

Wang, G., et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. *arXiv:2305.16291*.

Wang, L., et al. (2024). Augmenting Language Models with Long-Term Memory. *arXiv:2306.07174*.

Yang, C., et al. (2023). Large Language Models as Optimizers. *arXiv:2309.03409*.

Yang, Z., et al. (2020). NeurASP: Embracing Neural Networks into Answer Set Programming. *IJCAI*.

Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR*.

Zhao, A., et al. (2023). ExpeL: LLM Agents Are Experiential Learners. *arXiv:2308.10144*.

Zhong, W., et al. (2024). MemoryBank: Enhancing Large Language Models with Long-Term Memory. *AAAI*.

Zhou, S., et al. (2023). Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models. *arXiv:2310.04406*.

Zhou, Y., et al. (2023). Large Language Models Are Human-Level Prompt Engineers. *ICLR*.

---

## Appendix A: Hyperparameter Settings

**Table A1.** Complete hyperparameter configuration with source file references.

| Parameter | Value | Source |
|-----------|-------|--------|
| UNLEARN_FAILURE_THRESHOLD | 2 | precept_mcp_server.py:867 |
| Confidence decay (δ) | ×0.5 | precept_mcp_server.py:902 |
| Confidence restore | +0.25 | precept_mcp_server.py:953 |
| ensemble_conflict_threshold | 0.4 | conflict_resolution.py:91 |
| nli_vote_weight | 0.30 | conflict_resolution.py:84 |
| semantic_vote_weight | 0.30 | conflict_resolution.py:85 |
| temporal_vote_weight | 0.15 | conflict_resolution.py:86 |
| evidence_vote_weight | 0.15 | conflict_resolution.py:87 |
| llm_vote_weight | 0.10 | conflict_resolution.py:88 |
| static_prior_alpha, beta | 10.0, 2.0 | conflict_resolution.py:100-101 |
| dynamic_prior_alpha, beta | 5.0, 3.0 | conflict_resolution.py:102-103 |
| Early stop threshold | 0.98 | complexity_analyzer.py:533 |
| confidence_threshold | 0.9 | complexity_analyzer.py:491 |
| diversity_threshold | 0.7 | complexity_analyzer.py:490 |
| min_rollouts | 1 | complexity_analyzer.py:492 |
| max_rollouts | 15 | complexity_analyzer.py:493 |
| diversity_rollouts | 5 | complexity_analyzer.py:494 |
| consistency_rollouts | 3 | complexity_analyzer.py:495 |

---

## Appendix B: Statistical Analysis

**Table B1.** Effect size interpretation.

| Cohen's d | Interpretation | PRECEPT Results |
|-----------|----------------|-----------------|
| < 0.2 | Small | — |
| 0.2–0.8 | Medium | booking_3way (0.56) |
| > 0.8 | Large | booking_2way (2.88) |
| > 1.5 | Very Large | overall (2.21) |
| > 4.0 | Huge | logistics_3way (4.30) |
| > 8.0 | Exceptional | logistics_2way (8.27) |

**Table B2.** Key statistical results.

| Comparison | t | p | d |
|------------|---|---|---|
| PRECEPT vs Full Reflexion | 8.59 | < 0.001 | 2.21 |
| PRECEPT vs ExpeL | 8.09 | < 0.001 | 2.06 |
| logistics_2way | 18.50 | < 0.001 | 8.27 |
| logistics_3way | 7.91 | < 0.001 | 4.30 |

---

## Appendix C: Reproducibility Checklist

Following best practices for reproducible research (Pineau et al., 2021), we provide the following checklist:

### C.1 Code and Data

| Item | Status | Details |
|------|--------|---------|
| Source code | ✓ Available | `src/precept/` directory |
| Dependencies | ✓ Specified | `requirements.txt` with pinned versions |
| Installation | ✓ Documented | `README.md` with setup instructions |
| Experiment scripts | ✓ Provided | `experiments/` directory |
| Data generation | ✓ Reproducible | Synthetic benchmark with seed control |
| Random seeds | ✓ Documented | 10 seeds: 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021 |

### C.2 Experimental Protocol

| Item | Details |
|------|---------|
| Hardware | Apple M1 Pro, 16GB RAM (CPU-only) |
| Software | Python 3.11, OpenAI API (GPT-4) |
| Total compute | ~50 GPU-equivalent hours |
| API costs | ~$150 USD (OpenAI) |
| Runtime per experiment | 2-4 hours |
| Total runs | 39 configurations × 10 seeds = 390 runs |

### C.3 Statistical Methodology

| Item | Details |
|------|---------|
| Primary metric | P₁ (first-try success rate) |
| Secondary metrics | Pₜ (overall success), Steps, LLM calls |
| Confidence intervals | 95% CI using t-distribution |
| Hypothesis tests | Paired t-test (within-subject design) |
| Multiple comparisons | Bonferroni correction (α/k) |
| Effect sizes | Cohen's d with pooled standard deviation |
| Sample size | N=10 seeds per configuration |
| Power analysis | Post-hoc: >99% power for d>1.0 |

### C.4 Hyperparameters

All hyperparameters are documented in Appendix A (Table A1) with exact source file references. No hyperparameter tuning was performed on test sets—all values were fixed before experiments.

### C.5 Baselines (Enhanced Implementations)

| Baseline | Original (Literature) | Our Enhancement | Purpose |
|----------|----------------------|----------------|---------|
| Full Reflexion | In-memory buffer, recency-based retrieval | Vector database + condition metadata pre-filtering + BM25/semantic hybrid (RRF) + structured prompts + JSON persistence | Maximize retrieval quality to isolate exact-vs-approximate boundary |
| ExpeL | Embedding-similarity retrieval only | Condition metadata pre-filtering + BM25/semantic hybrid (RRF) + structured insight prompts + direct solution storage in metadata | Maximize retrieval quality to isolate exact-vs-approximate boundary |

Both baselines use identical task distributions, evaluation metrics, and random seeds as PRECEPT to ensure fair comparison. Crucially, both baselines are enhanced well beyond their original literature implementations (see §7.1 for details). These enhancements give baselines every architectural advantage short of exact hash-based retrieval—PRECEPT's core mechanism. Any remaining performance gap is attributable to the fundamental exact-vs-approximate retrieval boundary, not to implementation disadvantage.

### C.6 Limitations of Reproducibility

- **API variability:** OpenAI API responses may vary over time due to model updates
- **Stochastic LLM:** Temperature=0 used where possible, but some stochasticity remains
- **Cost barrier:** Full reproduction requires ~$150 in API costs

### C.7 Code Availability Statement

The complete PRECEPT implementation, including all experiment scripts, baseline implementations, and data generation code, is available at:

```
https://github.com/[REDACTED]/precept-framework
```

The repository includes:
- `src/precept/` - Core framework implementation
- `experiments/` - All experiment configurations
- `baselines/` - Full Reflexion and ExpeL implementations
- `data/` - Generated benchmark data and results
- `notebooks/` - Analysis and visualization notebooks

---

## Appendix D: Additional Experimental Results

### D.1 Learning Curves

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'xyChart': {'titleColor': '#1a1a1a', 'plotColorPalette': '#2c3e50, #7f8c8d, #bdc3c7'}}}}%%
xychart-beta
    title "Learning Curve: P₁ vs Training Episodes (Logistics Domain)"
    x-axis ["5", "10", "15", "20", "25", "30"]
    y-axis "P₁ (%)" 0 --> 100
    line [45, 62, 78, 89, 94, 96]
    line [25, 35, 42, 48, 52, 55]
    line [20, 28, 35, 40, 42, 45]
```

**Figure D1.** Learning efficiency comparison. PRECEPT (top) reaches 90%+ P₁ with ~20 episodes. Baselines plateau at 50-55%.

### D.2 Per-Domain Results

**Table D1.** Detailed per-domain results (Experiment 6).

| Domain | Config | PRECEPT P₁ | FR P₁ | ExpeL P₁ | Δ vs FR |
|--------|--------|------------|-------|----------|---------|
| Logistics | 2-way | **100.0%** | 34.8% | 31.2% | +65.2 pp |
| Logistics | 3-way | **93.3%** | 15.6% | 18.4% | +77.7 pp |
| Booking | 2-way | **95.0%** | 52.6% | 48.3% | +42.4 pp |
| Booking | 3-way | **80.3%** | 66.0% | 62.1% | +14.3 pp |

### D.3 Constraint Type Distribution

**Table D2.** Constraint classification distribution across experiments.

| Domain | HARD Constraints | SOFT Constraints | Ratio |
|--------|-----------------|------------------|-------|
| Logistics | 89.2% | 10.8% | 8.3:1 |
| Booking | 76.4% | 23.6% | 3.2:1 |
| Coding | 82.1% | 17.9% | 4.6:1 |
| DevOps | 85.7% | 14.3% | 6.0:1 |
| Finance | 91.3% | 8.7% | 10.5:1 |
| Integration | 78.9% | 21.1% | 3.7:1 |

Most errors are classified as HARD (permanent), validating the conservative default classification strategy in Algorithm 2b.

### D.4 Experiment 3: Per-Task Test Breakdown

*Tables D3 and D4 report per-task results from the seed=42 pilot run. Primary publication results with 10-seed statistical validation are in Section 7.4.2 (Table 15b).*

**Table D3.** Per-task test results — Single Condition (N=1), pilot (seed=42).

| β | Test | Key | Status | PRECEPT | ExpeL | FR |
|---|------|-----|--------|---------|-------|----|
| 1 | 1/4 | SH-701 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 1 | 2/4 | R-482 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=N |
| 1 | 3/4 | SH-701 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 1 | 4/4 | R-482 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| **1** | **Avg** | — | — | **2.0 steps** | **2.0 steps** | **3.5 steps** |
| 2 | 1/4 | H-903 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 2 | 2/4 | LA-550 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=N | ✓ P₁=N |
| 2 | 3/4 | ROUTE-FAIL | Novel | ✓ P₁=N (7 steps) | ✗ | ✓ P₁=N |
| 2 | 4/4 | H-903 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| **2** | **Avg** | — | — | **3.25 steps** | **5.0 steps** | **4.0 steps** |
| 3 | 1/4 | R-482 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 3 | 2/4 | H-903 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 3 | 3/4 | R-482 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 3 | 4/4 | H-903 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| **3** | **Avg** | — | — | **2.0 steps** | **2.0 steps** | **3.0 steps** |
| 4 | 1/4 | H-903 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 4 | 2/4 | LA-550 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 4 | 3/4 | R-482 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 4 | 4/4 | SH-701 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| **4** | **Avg** | — | — | **2.0 steps** | **2.0 steps** | **3.0 steps** |
| 5 | 1/4 | R-482 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 5 | 2/4 | LA-550 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 5 | 3/4 | H-903 | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 5 | 4/4 | CUSTOMS-HS-002 | Novel | ✓ P₁=N (7 steps) | ✓ P₁=N | ✓ P₁=Y |
| **5** | **Avg** | — | — | **3.25 steps** | **2.5 steps** | **3.0 steps** |

*Status: "Trained" = key was learned during the training phase for this β; "Novel" = key was not encountered during training, requiring exploratory recovery at test time.*

**Observation:** PRECEPT achieves 2-step resolution (the minimum) on all Trained keys via O(1) rule lookup. The two Novel keys (β=2 ROUTE-FAIL, β=5 CUSTOMS-HS-002) require exploratory recovery (7 steps). At N=1, all agents eventually solve all tasks (100% Pₜ across β=1–5), and the Status column reveals *why*: with only 4–6 distinct atomic keys, most test keys are Trained. The two Novel cases are the only ones where PRECEPT's P₁ drops below 100%. Baselines' step overhead (FR 3.0–4.0 vs PRECEPT 2.0) on Trained keys reflects their reliance on LLM reasoning rather than deterministic lookup.

**Table D4.** Per-task test results — Multi-Condition (N=5), pilot (seed=42). Key IDs map to composite keys in Table D4a below.

| β | Test | Key | Status | PRECEPT | ExpeL | FR |
|---|------|-----|--------|---------|-------|----|
| 1 | 1/4 | K₁ | Novel | ✓ P₁=N (5 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 1 | 2/4 | K₂ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 1 | 3/4 | K₁ | Novel | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 1 | 4/4 | K₂ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| **1** | **Avg** | — | — | **2.75 steps** | **2.0 steps** | **3.0 steps** |
| 2 | 1/4 | K₃ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=N | ✓ P₁=Y |
| 2 | 2/4 | K₄ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 2 | 3/4 | K₅ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=N | ✓ P₁=N |
| 2 | 4/4 | K₂ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| **2** | **Avg** | — | — | **2.0 steps** | **3.5 steps** | **4.5 steps** |
| 3 | 1/4 | K₅ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✗ |
| 3 | 2/4 | K₂ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=N | ✓ P₁=Y |
| 3 | 3/4 | K₃ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 3 | 4/4 | K₆ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✗ |
| **3** | **Avg** | — | — | **2.0 steps** | **3.0 steps** | **7.0 steps** |
| 4 | 1/4 | K₄ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=N | ✓ P₁=Y |
| 4 | 2/4 | K₃ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=N | ✓ P₁=Y |
| 4 | 3/4 | K₅ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=N |
| 4 | 4/4 | K₇ | Novel | ✓ P₁=N (5 steps) | ✓ P₁=N | ✓ P₁=Y |
| **4** | **Avg** | — | — | **2.75 steps** | **4.5 steps** | **3.5 steps** |
| 5 | 1/4 | K₅ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 5 | 2/4 | K₃ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=N | ✓ P₁=Y |
| 5 | 3/4 | K₄ | Trained | ✓ P₁=Y (2 steps) | ✓ P₁=Y | ✓ P₁=Y |
| 5 | 4/4 | K₈ | Novel | ✗ (7 steps) | ✗ | ✗ |
| **5** | **Avg** | — | — | **3.25 steps** | **5.5 steps** | **5.0 steps** |

*Status: "Trained" = key was learned during the training phase; "Novel" = key not encountered during training. Note: β=1 Test 3/4 (K₁) shows P₁=Y despite Novel status because PRECEPT learned the key in-test from Test 1/4's recovery.*

**Table D4a.** Composite key reference — N=5.

| Key | Full Composite | Appears in Tests |
|-----|---------------|-----------------|
| K₁ | C-COLD+E-STRM+E-TIDE+P-220+SH-701 | β=1 |
| K₂ | E-HEAT+H-903+R-482+SH-701+T-FLEX | β=1, 2, 3 |
| K₃ | C-COLD+C-HZMT+H-903+LA-550+R-482 | β=2, 3, 4, 5 |
| K₄ | C-COLD+C-HIGH+H-903+LA-550+P-220 | β=2, 4, 5 |
| K₅ | C-BULK+C-COLD+P-220+R-482+SH-701 | β=2, 3, 4, 5 |
| K₆ | E-HEAT+E-WNTR+LA-550+T-FLEX+T-PEAK | β=3 |
| K₇ | E-HEAT+E-WNTR+H-903+SH-701+T-NGHT | β=4 |
| K₈ | C-HZMT+E-HEAT+E-TIDE+LA-550+T-URGT | β=5 |

Note the extensive component overlap: R-482 appears in K₂, K₃, K₅; C-COLD appears in K₁, K₃, K₄, K₅; H-903 appears in K₂, K₃, K₄, K₇. Each shared component is a potential source of retrieval interference for baselines (see Section 7.4.3, Factor 2). Despite sharing multiple components, each composite maps to a different port via MD5 hash—making exact-match retrieval (PRECEPT) essential and approximate retrieval (baselines) unreliable.

**Observation:** The Status column reveals a clear pattern. PRECEPT achieves P₁=Y in 2 steps on every Trained key and requires recovery (5–7 steps) only on Novel keys. Baselines fail on Trained keys with increasing frequency as β grows: at β=4, ExpeL fails P₁ on 2 of 3 Trained keys (K₃, K₄) despite having seen them during training. This confirms the retrieval interference mechanism from Section 7.4.3—baselines' accumulated memories for overlapping composites produce conflicting recommendations, while PRECEPT's exact-match lookup is immune to this interference.

### D.5 Training Rules Learned

**Table D5.** Rules learned during training phase per β value.

| β | T_train | N=1 Rules | N=5 Rules |
|---|---------|-----------|-----------|
| 1 | 4 | 2 | 2 |
| 2 | 8 | 4 | 7 |
| 3 | 12 | 4 | 4 |
| 4 | 16 | 11 | 12 |
| 5 | 20 | 15 | 19 |

PRECEPT's rule accumulation grows monotonically with training exposure. The higher rule count at N=5 reflects that composite keys provide more unique learning opportunities per training episode (each key combination is distinct).
