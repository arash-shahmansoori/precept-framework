"""
Dynamic Static Knowledge Generator for PRECEPT.

Generates static knowledge that conflicts with dynamic learning based on num_conditions.
This enables testing of PRECEPT's conflict resolution capabilities at any condition level.

CRITICAL DESIGN: Wrong solutions are drawn from the SAME in-vocabulary option set
that agents actually choose from (MULTI_CONDITION_VALID_SOLUTIONS). This ensures
adversarial SK can genuinely mislead agents, not just add irrelevant noise.

Conflict Types:
1. FULL CONFLICT: Same N conditions, different (but in-vocabulary) solutions
   - Tests: PRECEPT's Bayesian conflict resolution
   - Baseline Trap: May follow the wrong in-vocabulary recommendation

2. PARTIAL CONFLICT - Subset: Static has fewer conditions
   - Tests: PRECEPT's exact-match requirement (should NOT apply partial match)
   - Baseline Trap: May incorrectly apply partial match

3. PARTIAL CONFLICT - Superset: Static has more conditions
   - Tests: Different condition sets should not match

Usage:
    generator = DynamicStaticKnowledgeGenerator(domain="logistics", num_conditions=3)
    knowledge_items = generator.generate()
"""

import random
from dataclasses import dataclass
from typing import Dict, List

from .multi_condition import (
    BookingConditions,
    CodingConditions,
    DevOpsConditions,
    FinanceConditions,
    IntegrationConditions,
    LogisticsConditions,
    MultiConditionConfig,
)


@dataclass
class ConflictingKnowledge:
    """A piece of static knowledge designed to conflict with dynamic learning."""

    condition_key: str
    content: str
    solution: str
    conflict_type: str  # "full", "partial_subset", "partial_superset"
    intended_dynamic_key: str  # The dynamic key this is designed to conflict with
    metadata: Dict


def _get_domain_config(domain: str):
    """Import and return the domain config class (avoids circular imports)."""
    if domain == "logistics":
        from .logistics import LogisticsConfig
        return LogisticsConfig
    elif domain == "integration":
        from .integration import IntegrationConfig
        return IntegrationConfig
    elif domain == "booking":
        from .booking import BookingConfig
        return BookingConfig
    else:
        return None


class DynamicStaticKnowledgeGenerator:
    """
    Generate static knowledge that conflicts with dynamic learning.

    The generator creates knowledge items that will conflict with rules
    learned during training, enabling rigorous testing of PRECEPT's
    conflict resolution vs baseline's vulnerability to adversarial knowledge.
    """

    DOMAIN_CONDITIONS = {
        "logistics": LogisticsConditions,
        "booking": BookingConditions,
        "coding": CodingConditions,
        "devops": DevOpsConditions,
        "finance": FinanceConditions,
        "integration": IntegrationConditions,
    }

    DOMAIN_TEMPLATES = {
        "logistics": {
            "full_conflict": (
                "VERIFIED SOLUTION: When error conditions [{conditions}] are active simultaneously, "
                "the ONLY working port is {solution}. All other ports will reject shipments under "
                "these conditions. This has been confirmed by the Maritime Authority and shipping "
                "operations team. Confidence: HIGH. Source: Maritime Operations Database 2024."
            ),
            "partial_subset": (
                "Port operations advisory for {subset_condition}: Route all shipments through "
                "{solution}. This port has confirmed availability and priority handling for this "
                "condition. Do NOT use other ports. Source: Port Operations Manual v12."
            ),
            "partial_superset": (
                "Extended protocol for [{superset_conditions}]: The designated port is {solution}. "
                "This routing has been tested under all listed conditions and is the only reliable "
                "option. Source: International Shipping Guidelines 2024."
            ),
            "outdated": (
                "ESTABLISHED PROCEDURE: Under conditions [{conditions}], always use {solution}. "
                "This routing was validated across 500+ shipments with 100% success rate. "
                "Source: Logistics Best Practices Archive."
            ),
        },
        "booking": {
            "full_conflict": "Airline policy: For conditions [{conditions}], use {solution}. Source: IATA Guidelines 2024.",
            "partial_subset": "Quick reference for {subset_condition}: {solution} is recommended. Source: Booking System Manual.",
            "partial_superset": "Complex booking scenario [{superset_conditions}]: Apply {solution}. Source: Travel Agent Handbook.",
            "outdated": "Legacy procedure: With [{conditions}], {solution} was previously standard (deprecated).",
        },
        "coding": {
            "full_conflict": "Best practice: When [{conditions}] detected, use {solution}. Source: Developer Guidelines 2024.",
            "partial_subset": "Quick fix for {subset_condition}: Try {solution}. Source: Stack Overflow Top Answer.",
            "partial_superset": "Advanced debugging [{superset_conditions}]: Apply {solution}. Source: Senior Dev Notes.",
            "outdated": "Historical solution: For [{conditions}], {solution} was recommended (now deprecated).",
        },
        "devops": {
            "full_conflict": "Runbook: Conditions [{conditions}] require {solution}. Source: SRE Handbook 2024.",
            "partial_subset": "Quick response for {subset_condition}: Execute {solution}. Source: Incident Response Guide.",
            "partial_superset": "Escalation protocol [{superset_conditions}]: Initiate {solution}. Source: Platform Team Docs.",
            "outdated": "Legacy runbook: Under [{conditions}], {solution} was standard (superseded).",
        },
        "finance": {
            "full_conflict": "Trading rule: Conditions [{conditions}] trigger {solution}. Source: Risk Management Policy 2024.",
            "partial_subset": "Market guidance for {subset_condition}: Consider {solution}. Source: Trading Floor Manual.",
            "partial_superset": "Complex scenario [{superset_conditions}]: Execute {solution}. Source: Quantitative Strategy Doc.",
            "outdated": "Historical strategy: With [{conditions}], {solution} was preferred (pre-regulation change).",
        },
        "integration": {
            "full_conflict": (
                "VERIFIED SOLUTION: When conditions [{conditions}] occur together, the ONLY "
                "working endpoint is {solution}. All other endpoints will fail with authentication "
                "or gateway errors under this condition combination. This has been validated by "
                "the platform engineering team. Confidence: HIGH. Source: Integration Runbook 2024."
            ),
            "partial_subset": (
                "Integration advisory for {subset_condition}: Use {solution} exclusively. "
                "This endpoint has been tested and confirmed working for this condition. "
                "Other endpoints are known to fail. Source: Developer Portal Knowledge Base."
            ),
            "partial_superset": (
                "Enterprise scenario [{superset_conditions}]: The designated endpoint is "
                "{solution}. This has been validated under the full condition set. "
                "Source: Architecture Review Board Decision 2024."
            ),
            "outdated": (
                "ESTABLISHED PROCEDURE: For conditions [{conditions}], always use {solution}. "
                "This endpoint was validated across 1000+ API calls with 100% success rate. "
                "Source: Integration Best Practices Archive."
            ),
        },
    }

    def __init__(
        self,
        domain: str,
        num_conditions: int = 2,
        num_full_conflicts: int = 4,
        num_partial_subset: int = 2,
        num_partial_superset: int = 2,
        num_outdated: int = 2,
    ):
        self.domain = domain
        self.num_conditions = max(1, min(10, num_conditions))
        self.num_full_conflicts = num_full_conflicts
        self.num_partial_subset = num_partial_subset
        self.num_partial_superset = num_partial_superset
        self.num_outdated = num_outdated

        condition_class = self.DOMAIN_CONDITIONS.get(domain)
        if condition_class:
            self.conditions = condition_class()
        else:
            raise ValueError(f"Unknown domain: {domain}")

        self.templates = self.DOMAIN_TEMPLATES.get(domain, {})

        self._domain_config = _get_domain_config(domain)
        if self._domain_config and hasattr(self._domain_config, "MULTI_CONDITION_VALID_SOLUTIONS"):
            self._valid_solutions = list(self._domain_config.MULTI_CONDITION_VALID_SOLUTIONS)
        else:
            self._valid_solutions = []

    def _get_wrong_solution_for_key(self, condition_key: str) -> str:
        """Get an in-vocabulary wrong solution for a specific condition_key.

        Computes the correct answer via the domain config, then returns
        a DIFFERENT valid solution from the same pool.
        """
        if not self._valid_solutions or len(self._valid_solutions) < 2:
            return random.choice(self._valid_solutions) if self._valid_solutions else "alternative"

        if self._domain_config and hasattr(self._domain_config, "get_valid_solution_for_conditions"):
            correct = self._domain_config.get_valid_solution_for_conditions(condition_key)
        else:
            correct = None

        wrong_options = [s for s in self._valid_solutions if s != correct]
        if not wrong_options:
            wrong_options = self._valid_solutions
        return random.choice(wrong_options)

    def _get_random_wrong_solution(self) -> str:
        """Get a random in-vocabulary solution (used when no specific key is available)."""
        if self._valid_solutions:
            return random.choice(self._valid_solutions)
        return "alternative"

    def _generate_full_conflicts(self) -> List[Dict]:
        """
        Generate FULL CONFLICT items.

        Same N conditions as dynamic learning, but a DIFFERENT in-vocabulary solution.
        The wrong solution is a real, choosable option — genuinely adversarial.
        """
        items = []

        for i in range(self.num_full_conflicts):
            conditions = self.conditions.get_random_conditions(self.num_conditions)
            condition_key = MultiConditionConfig.generate_condition_key(conditions)
            condition_str = " + ".join(conditions)

            wrong_solution = self._get_wrong_solution_for_key(condition_key)

            content = self.templates.get("full_conflict", "").format(
                conditions=condition_str,
                solution=wrong_solution,
            )

            items.append(
                {
                    "content": content,
                    "metadata": {
                        "type": "factual_knowledge",
                        "domain": self.domain,
                        "conflict_type": "full",
                        "condition_key": condition_key,
                        "conditions_str": "+".join(conditions),
                        "num_conditions": self.num_conditions,
                        "solution": wrong_solution,
                        "reliability": 0.85,
                        "source": "static_knowledge_base",
                        "note": "ADVERSARIAL - in-vocabulary wrong solution",
                    },
                }
            )

        return items

    def _generate_partial_subset_conflicts(self) -> List[Dict]:
        """
        Generate PARTIAL CONFLICT (Subset) items.

        Static has FEWER conditions than dynamic learning will have.
        Tests PRECEPT's exact-match requirement.
        """
        items = []

        if self.num_conditions <= 1:
            return items

        for i in range(self.num_partial_subset):
            full_conditions = self.conditions.get_random_conditions(self.num_conditions)
            full_key = MultiConditionConfig.generate_condition_key(full_conditions)

            subset_size = max(1, self.num_conditions // 2)
            subset_conditions = full_conditions[:subset_size]
            subset_key = MultiConditionConfig.generate_condition_key(subset_conditions)
            subset_str = (
                subset_conditions[0]
                if len(subset_conditions) == 1
                else " + ".join(subset_conditions)
            )

            wrong_solution = self._get_wrong_solution_for_key(full_key)

            content = self.templates.get("partial_subset", "").format(
                subset_condition=subset_str,
                solution=wrong_solution,
            )

            items.append(
                {
                    "content": content,
                    "metadata": {
                        "type": "factual_knowledge",
                        "domain": self.domain,
                        "conflict_type": "partial_subset",
                        "condition_key": subset_key,
                        "conditions_str": "+".join(subset_conditions),
                        "num_conditions": len(subset_conditions),
                        "intended_dynamic_key": full_key,
                        "intended_dynamic_conditions_str": "+".join(full_conditions),
                        "solution": wrong_solution,
                        "reliability": 0.85,
                        "source": "static_knowledge_base",
                        "note": "PARTIAL MATCH TRAP - in-vocabulary wrong solution",
                    },
                }
            )

        return items

    def _generate_partial_superset_conflicts(self) -> List[Dict]:
        """
        Generate PARTIAL CONFLICT (Superset) items.

        Static has MORE conditions than dynamic learning will have.
        """
        items = []

        if self.num_conditions >= 10:
            return items

        for i in range(self.num_partial_superset):
            base_conditions = self.conditions.get_random_conditions(self.num_conditions)
            base_key = MultiConditionConfig.generate_condition_key(base_conditions)

            extra_count = min(2, 10 - self.num_conditions)
            all_conds = list(self.conditions.get_all_conditions().keys())
            available = [c for c in all_conds if c not in base_conditions]
            extra = random.sample(available, min(extra_count, len(available)))

            superset_conditions = base_conditions + extra
            superset_key = MultiConditionConfig.generate_condition_key(superset_conditions)
            superset_str = " + ".join(superset_conditions)

            wrong_solution = self._get_wrong_solution_for_key(base_key)

            content = self.templates.get("partial_superset", "").format(
                superset_conditions=superset_str,
                solution=wrong_solution,
            )

            items.append(
                {
                    "content": content,
                    "metadata": {
                        "type": "factual_knowledge",
                        "domain": self.domain,
                        "conflict_type": "partial_superset",
                        "condition_key": superset_key,
                        "conditions_str": "+".join(superset_conditions),
                        "num_conditions": len(superset_conditions),
                        "intended_dynamic_key": base_key,
                        "intended_dynamic_conditions_str": "+".join(base_conditions),
                        "solution": wrong_solution,
                        "reliability": 0.80,
                        "source": "static_knowledge_base",
                        "note": "SUPERSET - in-vocabulary wrong solution",
                    },
                }
            )

        return items

    def _generate_outdated_knowledge(self) -> List[Dict]:
        """
        Generate OUTDATED knowledge items with in-vocabulary wrong solutions.
        """
        items = []

        for i in range(self.num_outdated):
            conditions = self.conditions.get_random_conditions(self.num_conditions)
            condition_key = MultiConditionConfig.generate_condition_key(conditions)
            condition_str = " + ".join(conditions)

            wrong_solution = self._get_wrong_solution_for_key(condition_key)

            content = self.templates.get("outdated", "").format(
                conditions=condition_str,
                solution=wrong_solution,
            )

            items.append(
                {
                    "content": content,
                    "metadata": {
                        "type": "factual_knowledge",
                        "domain": self.domain,
                        "conflict_type": "outdated",
                        "condition_key": condition_key,
                        "conditions_str": "+".join(conditions),
                        "num_conditions": self.num_conditions,
                        "solution": wrong_solution,
                        "reliability": 0.80,
                        "source": "static_knowledge_base",
                        "timestamp": "2023-01-01",
                        "note": "OUTDATED - in-vocabulary wrong solution",
                    },
                }
            )

        return items

    def _generate_agreement_knowledge(self) -> List[Dict]:
        """
        Generate AGREEMENT knowledge items.

        These items agree with what dynamic learning will discover,
        boosting confidence when both sources align.
        """
        items = []

        for i in range(2):
            conditions = self.conditions.get_random_conditions(self.num_conditions)
            condition_key = MultiConditionConfig.generate_condition_key(conditions)
            condition_str = " + ".join(conditions)

            content = (
                f"Verified best practice: For conditions [{condition_str}], the recommended "
                f"approach has been validated through multiple sources. Source: Industry Standards 2024."
            )

            items.append(
                {
                    "content": content,
                    "metadata": {
                        "type": "factual_knowledge",
                        "domain": self.domain,
                        "conflict_type": "agreement",
                        "condition_key": condition_key,
                        "conditions_str": "+".join(conditions),
                        "num_conditions": self.num_conditions,
                        "reliability": 0.9,
                        "source": "static_knowledge_base",
                        "note": "AGREEMENT - boosts confidence when dynamic aligns",
                    },
                }
            )

        return items

    def generate(self) -> List[Dict]:
        """Generate all static knowledge items."""
        items = []

        items.extend(self._generate_full_conflicts())
        items.extend(self._generate_partial_subset_conflicts())
        items.extend(self._generate_partial_superset_conflicts())
        items.extend(self._generate_outdated_knowledge())
        items.extend(self._generate_agreement_knowledge())

        print(f"\n📚 Generated Dynamic Static Knowledge for {self.domain}:")
        print(f"   num_conditions: {self.num_conditions}")
        print(f"   Valid solution pool: {self._valid_solutions}")
        print(f"   Full conflicts: {self.num_full_conflicts}")
        print(f"   Partial (subset): {self.num_partial_subset}")
        print(f"   Partial (superset): {self.num_partial_superset}")
        print(f"   Outdated: {self.num_outdated}")
        print("   Agreement: 2")
        print(f"   Total items: {len(items)}")

        return items

    def generate_json(self) -> str:
        """Generate knowledge items as JSON string for ingestion."""
        import json

        return json.dumps(self.generate())


def generate_dynamic_static_knowledge(
    domain: str,
    num_conditions: int = 2,
) -> List[Dict]:
    """
    Convenience function to generate dynamic static knowledge.

    Args:
        domain: Domain name
        num_conditions: Number of conditions to match

    Returns:
        List of knowledge items
    """
    generator = DynamicStaticKnowledgeGenerator(
        domain=domain,
        num_conditions=num_conditions,
    )
    return generator.generate()
