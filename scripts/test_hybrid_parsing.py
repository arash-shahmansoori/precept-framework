#!/usr/bin/env python3
"""
Test script for PRECEPT's hybrid task parsing capability.

This tests the improved LLM-enabled parsing for complex task descriptions.
The hybrid parser combines:
1. Fast rule-based parsing (deterministic)
2. LLM-assisted parsing with structured outputs (for complex/ambiguous tasks)

Usage:
    uv run python tests/test_hybrid_parsing.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src and examples to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "examples"))

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from config.experiment import SERVER_SCRIPT

from precept import PRECEPTAgent, get_domain_strategy
from precept.config import PreceptConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_hybrid_parsing")


# Complex task descriptions to test parsing
COMPLEX_TASKS = {
    "logistics": [
        # Simple task (HIGH confidence - rule-based)
        "Ship cargo from rotterdam to boston",
        # Complex: multiple conditions embedded (HIGH confidence - rule-based)
        "Ship hazardous materials from Rotterdam port during night shift hours with refrigerated cargo type R-482",
        # ═══════════════════════════════════════════════════════════════════════
        # AMBIGUOUS TASKS - Should trigger LLM fallback (LOW confidence)
        # ═══════════════════════════════════════════════════════════════════════
        # No clear action or entity
        "Please handle the European logistics situation for our Boston customer",
        # Very vague
        "Fix the package thing for the overseas destination",
        # No domain keywords at all
        "The thing from place A to place B needs to go urgently",
        # Indirect reference with no explicit ports
        "Our client's items need to reach their final destination somehow",
    ],
    "booking": [
        # Simple
        "Book flight AA-999 for John Doe",
        # Complex
        "Find an alternative flight to replace the cancelled AA-999 booking for passenger Jane Smith traveling on the night schedule",
        # Ambiguous
        "The customer needs to rebook their phantom inventory affected reservation",
    ],
    "coding": [
        # Simple
        "Install numpy package",
        # Complex
        "The build is failing because requests package version 2.28 has a CVE vulnerability - need to upgrade or find alternative",
        # Multi-dependency
        "Fix the ImportError: cannot import name 'AsyncClient' from 'httpx' - might need to update or use aiohttp instead",
    ],
    "devops": [
        # Simple
        "Deploy to production",
        # Complex
        "The Kubernetes pod api-server-7b8c9 is stuck in CrashLoopBackOff state - investigate and remediate",
        # IAM issue
        "CloudFormation stack deployment-stack-123 failed due to insufficient IAM permissions for lambda:CreateFunction",
    ],
    "finance": [
        # Simple
        "Execute trade for AAPL",
        # Complex
        "Process the high-frequency trade for volatile symbol NVDA during market hours with compliance check bypass",
        # Data issue
        "The price feed for BTC/USD is showing stale data from 2 hours ago - need alternative data source",
    ],
    "integration": [
        # Simple
        "Connect OAuth to github",
        # Complex
        "The webhook endpoint https://api.example.com/webhook is returning 503 errors - need to configure retry with exponential backoff",
        # Token issue
        "OAuth token for provider salesforce expired and refresh failed - need to re-authenticate or use alternative",
    ],
}


async def run_parsing_for_domain(domain: str, tasks: list[str]) -> dict:
    """Test parsing capability for a specific domain."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"🔬 TESTING DOMAIN: {domain.upper()}")
    logger.info(f"{'=' * 80}")

    strategy = get_domain_strategy(domain)
    config = PreceptConfig()

    # Enable hybrid parsing
    config.agent.enable_hybrid_parsing = True
    config.agent.verbose_llm = True

    results = {
        "domain": domain,
        "tasks": [],
        "rule_based_confidence": [],
        "llm_used": [],
    }

    # Create and connect agent (no async context manager)
    agent = PRECEPTAgent(
        domain_strategy=strategy,
        config=config,
        server_script=SERVER_SCRIPT,
    )
    await agent.connect()

    try:
        for i, task in enumerate(tasks, 1):
            logger.info(f"\n📝 Task {i}/{len(tasks)}: {task[:60]}...")

            # Test rule-based parsing first
            rule_parsed = strategy.parse_task(task)
            logger.info(
                f"   📋 Rule-based: action={rule_parsed.action}, entity={rule_parsed.entity}"
            )

            # Calculate rule-based confidence using agent's method
            confidence = agent._assess_parsing_quality(task, rule_parsed)
            logger.info(f"   📊 Rule confidence: {confidence:.2f}")

            # Test hybrid parsing
            if hasattr(agent, "_hybrid_parse_task"):
                try:
                    hybrid_parsed = await agent._hybrid_parse_task(task)
                    logger.info(
                        f"   🤖 Hybrid: action={hybrid_parsed.action}, "
                        f"entity={hybrid_parsed.entity}, "
                        f"source={getattr(hybrid_parsed, 'source', 'N/A')}, "
                        f"target={getattr(hybrid_parsed, 'target', 'N/A')}"
                    )

                    # Check if LLM was used
                    llm_used = confidence < 0.8
                    if llm_used:
                        logger.info(
                            f"   ✨ LLM-assisted parsing was triggered (confidence {confidence:.2f} < 0.8)"
                        )
                    else:
                        logger.info(
                            f"   ⚡ Pure rule-based parsing used (confidence {confidence:.2f} >= 0.8)"
                        )

                    results["tasks"].append(
                        {
                            "task": task,
                            "rule_based": {
                                "action": rule_parsed.action,
                                "entity": rule_parsed.entity,
                            },
                            "hybrid": {
                                "action": hybrid_parsed.action,
                                "entity": hybrid_parsed.entity,
                                "source": getattr(hybrid_parsed, "source", None),
                                "target": getattr(hybrid_parsed, "target", None),
                            },
                            "confidence": confidence,
                            "llm_used": llm_used,
                        }
                    )
                    results["rule_based_confidence"].append(confidence)
                    results["llm_used"].append(llm_used)

                except Exception as e:
                    logger.warning(f"   ❌ Hybrid parsing failed: {e}")
                    results["tasks"].append(
                        {
                            "task": task,
                            "error": str(e),
                        }
                    )
    finally:
        # Cleanup: close MCP client connection
        # Suppress MCP cleanup errors (they are harmless)
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        if hasattr(agent, "mcp_client") and agent.mcp_client:
            try:
                await agent.mcp_client.close()
            except Exception:
                pass  # MCP cleanup errors are expected

    return results


async def main():
    """Run hybrid parsing tests across all domains."""
    logger.info("=" * 80)
    logger.info("🧪 PRECEPT HYBRID PARSING TEST")
    logger.info("=" * 80)
    logger.info("Testing the improved LLM-enabled parsing for complex tasks")
    logger.info("Hybrid = Rule-based (fast) + LLM fallback (smart)")
    logger.info("")

    # Test ONE domain at a time to avoid MCP connection issues
    # Change this to test different domains
    test_domain = "logistics"

    if test_domain not in COMPLEX_TASKS:
        logger.error(f"Unknown domain: {test_domain}")
        return

    try:
        results = await run_parsing_for_domain(test_domain, COMPLEX_TASKS[test_domain])

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 SUMMARY")
        logger.info("=" * 80)

        if "rule_based_confidence" in results and results["rule_based_confidence"]:
            avg_conf = sum(results["rule_based_confidence"]) / len(
                results["rule_based_confidence"]
            )
            llm_count = sum(results["llm_used"])
            total = len(results["llm_used"])

            logger.info(f"\n{test_domain.upper()}:")
            logger.info(f"  📊 Average rule-based confidence: {avg_conf:.2f}")
            logger.info(
                f"  🤖 LLM-assisted parsing used: {llm_count}/{total} tasks ({100 * llm_count / total:.0f}%)"
            )

            logger.info("\n📋 DETAILED RESULTS:")
            for task_result in results["tasks"]:
                task_short = task_result.get("task", "")[:50] + "..."
                rule_action = task_result.get("rule_based", {}).get("action", "N/A")
                hybrid_entity = task_result.get("hybrid", {}).get("entity", "N/A")
                hybrid_source = task_result.get("hybrid", {}).get("source", "N/A")
                hybrid_target = task_result.get("hybrid", {}).get("target", "N/A")
                conf = task_result.get("confidence", 0)
                llm = "✨LLM" if task_result.get("llm_used") else "⚡Rule"
                logger.info(f"  [{llm}] {task_short}")
                logger.info(
                    f"       → entity={hybrid_entity}, source={hybrid_source}, target={hybrid_target} (conf={conf:.2f})"
                )

        logger.info("\n✅ Hybrid parsing test complete!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
