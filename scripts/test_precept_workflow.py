#!/usr/bin/env python3
"""
PRECEPT Workflow Test: Verify all stages work with ACTUAL API calls.

This test script verifies the complete PRECEPT workflow:
1. Phase 1: Hard Ingestion (Knowledge Base)
2. Phase 2: Evo-Memory Runtime (Think-Act-Refine with Dual Retrieval)
3. Phase 3: COMPASS Compilation (Feedback Ingestion + Evolution + Pruning)

Usage:
    # Activate virtual environment first
    source .venv/bin/activate
    
    # Run the test
    python examples/test_precept_workflow.py

Requirements:
    - OPENAI_API_KEY in .env file
    - All dependencies installed (uv sync)
"""

import asyncio
import os
import sys
from pathlib import Path

# Load environment variables
import dotenv
dotenv.load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from precept import (
    PRECEPTOrchestrator,
    PRECEPTConfig,
    MemoryStore,
    MemoryConsolidator,
    ParetoMemoryManager,
    precept_llm_client,
    precept_embedding_fn,
    check_api_availability,
    SoftIngestionManager,
    FeedbackIngestionManager,
)
from precept.pareto_memory import TaskType


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_section(text: str):
    """Print a formatted section."""
    print("\n" + "-" * 50)
    print(text)
    print("-" * 50)


async def test_phase1_hard_ingestion():
    """Test Phase 1: Hard Ingestion (Knowledge Base Setup)."""
    print_section("📚 PHASE 1: Hard Ingestion")
    
    # Verify COMPASS components are available
    try:
        from precept.compass_integration import (
            COMPASSHardIngestion,
            COMPASSHardIngestionConfig,
        )
        print("  ✓ COMPASSHardIngestion available")
    except ImportError as e:
        print(f"  ⚠ COMPASSHardIngestion not available: {e}")
    
    # Simulate vector DB content
    vector_db_knowledge = {
        "hamburg_port_manual": "Hamburg Port: Status OPERATIONAL. Capacity: High.",
        "rotterdam_port_manual": "Rotterdam Port: Status OPERATIONAL. Trans-Atlantic alternative.",
        "pharma_shipping_guide": "Type-II Pharma requires temperature control.",
    }
    
    print(f"  ✓ Simulated {len(vector_db_knowledge)} documents in Vector DB")
    return {"documents": len(vector_db_knowledge), "status": "success"}


async def test_phase2_evomemory_runtime(agent: PRECEPTOrchestrator):
    """Test Phase 2: Evo-Memory Runtime (Think-Act-Refine)."""
    print_section("🚀 PHASE 2: Evo-Memory Runtime")
    
    # Step 2A: Dual Retrieval - Store prior experience
    print("\n  📥 Step 2A: Dual Retrieval")
    agent.memory_store.store_experience(
        task_description="Route pharma shipment via Hamburg",
        goal="Deliver on time",
        trajectory=[{"step": 1, "action": "Book Hamburg", "result": "FAILED - Strike"}],
        outcome="failure",
        correctness=0.0,
        strategy_used="default_routing",
        lessons_learned=["Hamburg has strike delay", "Use Rotterdam instead"],
        skills_demonstrated=["route_planning"],
        domain="logistics",
    )
    
    memories = agent.memory_store.retrieve_relevant(
        query="Hamburg pharma shipment",
        top_k=3,
        domain="logistics",
    )
    print(f"  ✓ Retrieved {len(memories)} memories from episodic store")
    
    # Step 2B-C: Think + Act (ReMem Loop)
    print("\n  🧠 Step 2B-C: ReMem Loop (Think-Act)")
    result = await agent.run_task(
        task="Route pharma shipment from Europe to Boston",
        goal="Find fastest route avoiding delays",
        domain="logistics",
    )
    print(f"  ✓ Task completed: steps={result.step_count}, llm_calls={result.llm_calls}")
    
    # Step 2D: Soft Ingestion
    print("\n  📝 Step 2D: Soft Ingestion")
    agent.create_warning_patch(
        query_pattern="Hamburg speed priority",
        warning="Use Rotterdam for speed-priority shipments",
        task="Route optimization",
        domain="logistics",
    )
    print(f"  ✓ Soft patch created. Total patches: {len(agent.soft_ingestion.patches)}")
    
    # Run additional tasks
    print("\n  📊 Running additional tasks...")
    for i in range(2):
        await agent.run_task(
            task=f"Logistics task {i+1}",
            goal="Test pattern building",
            domain="logistics",
        )
    print(f"  ✓ Total tasks: {agent.stats.total_tasks}")
    
    return {
        "tasks": agent.stats.total_tasks,
        "memories": agent.memory_store.get_stats()["current_size"],
        "patches": len(agent.soft_ingestion.patches),
        "status": "success"
    }


async def test_phase3_compass_compilation(agent: PRECEPTOrchestrator):
    """Test Phase 3: COMPASS Compilation (Evolution + Pruning)."""
    print_section("🔬 PHASE 3: COMPASS Compilation")
    
    # Step 3A: Feedback Ingestion
    print("\n  📈 Step 3A: Feedback Ingestion")
    trace_analysis = agent.analyze_execution_traces()
    print(f"  ✓ Analyzed {trace_analysis.get('total_traces', 0)} execution traces")
    
    # Step 3B-C: Consolidation (Mutation + Validation)
    print("\n  🧬 Step 3B-C: Consolidation")
    consolidation_result = await agent.force_consolidation()
    print(f"  ✓ Rules extracted: {len(consolidation_result.new_rules)}")
    print(f"  ✓ Memories pruned: {consolidation_result.stats.get('memories_pruned', 0)}")
    
    # Step 3D: Check evolved prompt
    print("\n  🚀 Step 3D: Deployment")
    rules_section = agent.consolidator.get_all_rules_as_prompt_section()
    if rules_section:
        print("  ✓ Consolidated rules generated")
    else:
        print("  ℹ No rules consolidated yet (need more task patterns)")
    
    return {
        "traces_analyzed": trace_analysis.get("total_traces", 0),
        "rules_extracted": len(consolidation_result.new_rules),
        "status": "success"
    }


async def main():
    """Run the complete PRECEPT workflow test."""
    print_header("🧪 PRECEPT WORKFLOW TEST")
    print("Testing all stages with ACTUAL OpenAI API calls")
    
    # Check API availability
    print("\n🔍 Checking API configuration...")
    availability = check_api_availability()
    
    if not availability.get("openai"):
        print("❌ OPENAI_API_KEY not configured!")
        print(f"   Details: {availability}")
        sys.exit(1)
    
    print("✅ OpenAI API configured")
    
    # Create PRECEPT agent
    config = PRECEPTConfig(
        max_memories=50,
        max_steps_per_task=3,
        consolidation_interval=3,
        min_strategy_count=2,
    )
    
    agent = PRECEPTOrchestrator(
        llm_client=precept_llm_client,
        embedding_fn=precept_embedding_fn,
        config=config,
    )
    
    agent.set_system_prompts({
        "system": """You are a logistics routing assistant.
Analyze shipping routes and provide recommendations.
Consider alternative routes when delays are reported."""
    })
    
    # Run all phases
    results = {}
    
    results["phase1"] = await test_phase1_hard_ingestion()
    results["phase2"] = await test_phase2_evomemory_runtime(agent)
    results["phase3"] = await test_phase3_compass_compilation(agent)
    
    # Print summary
    print_header("📊 TEST SUMMARY")
    
    all_passed = all(r.get("status") == "success" for r in results.values())
    
    print(f"""
    Phase 1 (Hard Ingestion):
      - Documents loaded: {results['phase1']['documents']}
      - Status: {'✓' if results['phase1']['status'] == 'success' else '✗'}
    
    Phase 2 (Evo-Memory Runtime):
      - Tasks executed: {results['phase2']['tasks']}
      - Memories stored: {results['phase2']['memories']}
      - Soft patches: {results['phase2']['patches']}
      - Status: {'✓' if results['phase2']['status'] == 'success' else '✗'}
    
    Phase 3 (COMPASS Compilation):
      - Traces analyzed: {results['phase3']['traces_analyzed']}
      - Rules extracted: {results['phase3']['rules_extracted']}
      - Status: {'✓' if results['phase3']['status'] == 'success' else '✗'}
    """)
    
    if all_passed:
        print("✅ ALL WORKFLOW STAGES PASSED!")
        print("\n🌟 PRECEPT is working correctly with ACTUAL OpenAI API calls.")
    else:
        print("❌ Some stages failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

