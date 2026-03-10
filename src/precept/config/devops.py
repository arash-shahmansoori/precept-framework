"""
DevOps Domain Configuration for PRECEPT.

Single source of truth for all DevOps-related configuration including:
- Stuck CloudFormation stacks and recovery actions
- IAM roles with propagation delays
- Kubernetes pod issues and fixes
- Infrastructure error codes
- Scenario generation templates

Usage:
    from precept.config import DevOpsConfig

    # Access configuration
    config = DevOpsConfig
    stuck_stacks = config.STUCK_STACKS
    iam_roles = config.IAM_ROLES
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple


@dataclass(frozen=True)
class DevOpsConfig:
    """
    Centralized configuration for DevOps domain.

    SINGLE SOURCE OF TRUTH for all DevOps-related data:
    - CloudFormation stack configurations and error codes
    - IAM role propagation issues
    - Kubernetes pod problems
    - Infrastructure scenario templates

    COHERENCE GUARANTEE: Each resource has consistent attributes:
    - error_code: The error for THIS specific resource
    - recovery_action: What works when THIS resource fails
    - lesson: The lesson specific to THIS failure mode
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # STUCK CLOUDFORMATION STACKS
    # Maps stack → (error_code, recovery_action, stack_type, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "CFN-881" means rollback failed.
    # Error codes do NOT hint at the solution.
    # ═══════════════════════════════════════════════════════════════════════════
    STUCK_STACKS: ClassVar[Dict[str, Dict]] = {
        "prod-api-stack": {
            "error_code": "CFN-881",  # Vague: doesn't reveal rollback state
            "recovery_action": "wait_for_completion",  # Changed to balance with other stacks
            "working_alternatives": ["wait_for_completion"],  # ONLY this works
            "stack_type": "API Gateway + Lambda",
            "lesson": "prod-api-stack has pending operations, wait_for_completion first",
            "description": "production API infrastructure",
            "error_message": "Stack operation failed. Error: CFN-881. Check CloudFormation console.",
        },
        "data-pipeline-stack": {
            "error_code": "CFN-882",  # Vague: different code, same underlying issue
            "recovery_action": "wait_for_completion",  # Changed to make unique per stack
            "working_alternatives": [
                "wait_for_completion"
            ],  # ONLY wait_for_completion works
            "stack_type": "Step Functions + Glue",
            "lesson": "data-pipeline-stack has pending operations, wait_for_completion then retry",
            "description": "ETL pipeline infrastructure",
            "error_message": "Deployment blocked. Reference: CFN-882.",
        },
        "auth-service-stack": {
            "error_code": "CFN-550",  # Vague: doesn't reveal creation failure
            "recovery_action": "delete_and_recreate",
            "working_alternatives": ["delete_and_recreate"],
            "stack_type": "Cognito + Lambda",
            "lesson": "auth-service-stack creation failed, must delete and recreate",
            "description": "authentication infrastructure",
            "error_message": "Stack provisioning error. Code: CFN-550.",
        },
        "monitoring-stack": {
            "error_code": "CFN-420",  # Vague: doesn't reveal in-progress state
            "recovery_action": "continue_update_rollback",  # Changed to balance stacks
            "working_alternatives": ["continue_update_rollback"],  # ONLY this works
            "stack_type": "CloudWatch + SNS",
            "lesson": "monitoring-stack needs rollback continuation before retry",
            "description": "monitoring and alerting",
            "error_message": "Cannot proceed. Error: CFN-420. Try again later.",
        },
        "vpc-network-stack": {
            "error_code": "CFN-661",  # Vague: doesn't reveal dependency issue
            "recovery_action": "remove_dependencies",
            "working_alternatives": ["remove_dependencies", "force_delete"],
            "stack_type": "VPC + Subnets",
            "lesson": "vpc-network-stack has dependencies, remove attached resources first",
            "description": "network infrastructure",
            "error_message": "Stack cleanup failed. Reference: CFN-661.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # IAM ROLES WITH PROPAGATION DELAY
    # Maps role → (error_code, wait_time, service, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "IAM-201" means propagation delay.
    # ═══════════════════════════════════════════════════════════════════════════
    IAM_ROLES: ClassVar[Dict[str, Dict]] = {
        "LambdaExecutionRole": {
            "error_code": "IAM-201",  # Vague: doesn't reveal propagation issue
            "wait_time": "15-30 seconds",
            "wait_seconds": 30,
            "service": "Lambda",
            "lesson": "IAM role needs 15-30s propagation delay before Lambda can assume it",
            "recovery_action": "exponential_backoff",
            "error_message": "Role assumption failed. Error: IAM-201.",
        },
        "ECSTaskRole": {
            "error_code": "IAM-302",  # Vague: doesn't reveal timing issue
            "wait_time": "30-60 seconds",
            "wait_seconds": 60,
            "service": "ECS",
            "lesson": "ECS task role needs longer propagation (30-60s), add exponential backoff",
            "recovery_action": "exponential_backoff",
            "error_message": "Task configuration error. Code: IAM-302.",
        },
        "GlueServiceRole": {
            "error_code": "IAM-403",  # Vague: doesn't reveal access timing
            "wait_time": "60-120 seconds",
            "wait_seconds": 120,
            "service": "Glue",
            "lesson": "Glue role propagation is slow (up to 2 min), wait before job start",
            "recovery_action": "wait_and_retry",
            "error_message": "Permission check failed. Reference: IAM-403.",
        },
        "StepFunctionsRole": {
            "error_code": "IAM-504",  # Vague: doesn't reveal role readiness
            "wait_time": "30-45 seconds",
            "wait_seconds": 45,
            "service": "Step Functions",
            "lesson": "Step Functions role needs 30-45s propagation before state machine creation",
            "recovery_action": "exponential_backoff",
            "error_message": "State machine configuration error. Code: IAM-504.",
        },
        "CodeBuildRole": {
            "error_code": "IAM-605",  # Vague: doesn't reveal input error
            "wait_time": "20-40 seconds",
            "wait_seconds": 40,
            "service": "CodeBuild",
            "lesson": "CodeBuild role needs propagation delay, retry after 20-40 seconds",
            "recovery_action": "wait_and_retry",
            "error_message": "Build configuration error. Code: IAM-605.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # KUBERNETES POD ISSUES
    # Maps issue → (error_code, fix_action, resource_type, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "K8S-101" means increase memory limits.
    # ═══════════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════════
    # K8S_ISSUES - Maps pod name to issue details
    # IMPORTANT: Pod names MUST match KNOWN_PODS and MCP PROBLEM_PODS!
    # Fix actions MUST match what MCP server expects (restart, increase_resources)
    # ═══════════════════════════════════════════════════════════════════════════
    K8S_ISSUES: ClassVar[Dict[str, Dict]] = {
        "api-server-pod": {
            "error_code": "K8S-137",  # Vague: just exit code
            "fix_action": "increase_resources",  # Matches MCP PROBLEM_PODS
            "working_alternatives": ["increase_resources"],
            "resource_type": "memory",
            "lesson": "api-server-pod failed with K8S-137, use increase_resources",
            "error_message": "Pod terminated unexpectedly. Error: K8S-137.",
        },
        "worker-pod": {
            "error_code": "K8S-250",  # Vague: doesn't reveal eviction
            "fix_action": "restart",  # Matches MCP PROBLEM_PODS
            "working_alternatives": ["restart"],
            "resource_type": "storage",
            "lesson": "worker-pod failed with K8S-250, use restart",
            "error_message": "Pod removed from node. Code: K8S-250.",
        },
        "frontend-pod": {
            "error_code": "K8S-101",  # Vague: doesn't reveal OOM
            "fix_action": "increase_resources",  # Matches MCP PROBLEM_PODS
            "working_alternatives": ["increase_resources"],
            "resource_type": "memory",
            "lesson": "frontend-pod failed with K8S-101, use increase_resources",
            "error_message": "Performance degraded. Reference: K8S-101.",
        },
        "cache-pod": {
            "error_code": "K8S-202",  # Vague: doesn't reveal eviction
            "fix_action": "restart",  # Matches MCP PROBLEM_PODS
            "working_alternatives": ["restart"],
            "resource_type": "storage",
            "lesson": "cache-pod failed with K8S-202, use restart",
            "error_message": "Pod evicted. Error: K8S-202.",
        },
        "database-pod": {
            "error_code": "K8S-303",  # Vague: doesn't reveal throttle
            "fix_action": "increase_resources",  # Matches MCP PROBLEM_PODS
            "working_alternatives": ["increase_resources"],
            "resource_type": "cpu",
            "lesson": "database-pod failed with K8S-303, use increase_resources",
            "error_message": "Pod throttled. Code: K8S-303.",
        },
        "ingress-pod": {
            "error_code": "K8S-404",  # Vague: doesn't reveal image issue
            "fix_action": "restart",  # Matches MCP PROBLEM_PODS
            "working_alternatives": ["restart"],
            "resource_type": "image",
            "lesson": "ingress-pod failed with K8S-404, use restart",
            "error_message": "Container failed. Reference: K8S-404.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # PROVIDERS - Cloud providers vocabulary
    # ═══════════════════════════════════════════════════════════════════════════
    PROVIDERS: ClassVar[List[str]] = ["aws", "gcp", "azure"]

    # ═══════════════════════════════════════════════════════════════════════════
    # REGIONS - Available deployment regions
    # ═══════════════════════════════════════════════════════════════════════════
    REGIONS: ClassVar[List[str]] = [
        "us-east-1",
        "us-west-2",
        "eu-west-1",
        "ap-southeast-1",
        "ap-northeast-1",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # STACK STATES - CloudFormation states vocabulary
    # ═══════════════════════════════════════════════════════════════════════════
    STACK_STATES: ClassVar[List[str]] = [
        "CREATE_COMPLETE",
        "UPDATE_COMPLETE",
        "ROLLBACK_FAILED",
        "DELETE_FAILED",
        "UPDATE_IN_PROGRESS",
        "CREATE_FAILED",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # STACK ACTIONS - Available recovery actions
    # ═══════════════════════════════════════════════════════════════════════════
    # STACK_ACTIONS - 9 actions to match logistics difficulty (1/9 = 11% random success)
    # Only 2 of these work for multi-condition (hash-mapped to 1 per condition_key)
    STACK_ACTIONS: ClassVar[List[str]] = [
        "update",
        "create",
        "continue_update_rollback",
        "delete_and_recreate",
        "wait_for_completion",
        "remove_dependencies",
        "force_cleanup",  # Added for difficulty parity
        "manual_intervention",  # Added for difficulty parity
        "skip_validation",  # Added for difficulty parity
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # KNOWN STACKS - Vocabulary for task parsing
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_STACKS: ClassVar[List[str]] = [
        "prod-api-stack",
        "data-pipeline-stack",
        "auth-service-stack",
        "monitoring-stack",
        "vpc-network-stack",
        "web-stack",
        "api-stack",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # KNOWN ROLES - Vocabulary for task parsing
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_ROLES: ClassVar[List[str]] = [
        "LambdaExecutionRole",
        "ECSTaskRole",
        "GlueServiceRole",
        "StepFunctionsRole",
        "CodeBuildRole",
        "deploy-role",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # KNOWN PODS - Vocabulary for task parsing
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_PODS: ClassVar[List[str]] = [
        "api-server-pod",
        "worker-pod",
        "frontend-pod",
        "cache-pod",
        "database-pod",
        "ingress-pod",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # K8S NAMESPACES
    # ═══════════════════════════════════════════════════════════════════════════
    K8S_NAMESPACES: ClassVar[List[str]] = [
        "production",
        "staging",
        "development",
        "monitoring",
        "data-pipeline",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # UPDATE TYPES - CloudFormation update contexts
    # ═══════════════════════════════════════════════════════════════════════════
    UPDATE_TYPES: ClassVar[Dict[str, str]] = {
        "ec2_instance": "EC2 instance type change",
        "lambda_config": "Lambda configuration update",
        "rds_scaling": "RDS scaling modification",
        "s3_policy": "S3 bucket policy update",
        "vpc_subnet": "VPC subnet configuration",
        "iam_policy": "IAM policy modification",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK REASON → LESSON TEMPLATE MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    BLOCK_REASON_TEMPLATES: ClassVar[Dict[str, str]] = {
        "rollback": "{stack} is stuck in rollback. Use {action} to recover.",
        "in_progress": "{stack} has operation in progress. Wait or cancel first.",
        "dependencies": "{stack} has dependencies. Remove attached resources before delete.",
        "propagation": "{role} needs propagation delay. Wait {wait_time} before using.",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # CLOUDFORMATION TASK TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    CFN_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Update {stack} with {update_type}",
        "Deploy changes to {stack} for {update_type}",
        "Modify {stack} configuration ({update_type})",
        "Push {update_type} to {stack}",
    ]

    CFN_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Apply {update_type} update to {stack}",
        "Roll out {update_type} for {stack}",
        "Execute {stack} deployment ({update_type})",
        "Finalize {stack} changes for {update_type}",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # IAM TASK TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    IAM_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Create {role} and assign to {service}",
        "Set up {role} for {service} access",
        "Configure {role} permissions for {service}",
        "Deploy {role} for {service} function",
    ]

    IAM_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Use {role} with {service} immediately after creation",
        "Apply {role} to new {service} resource",
        "Assign {role} to {service} deployment",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # KUBERNETES TASK TEMPLATES
    # NOTE: Templates MUST include {pod} placeholder so task parser can extract pod name
    # ═══════════════════════════════════════════════════════════════════════════
    K8S_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Debug {pod} in {namespace} namespace",
        "Troubleshoot {pod} deployment in {namespace}",
        "Investigate {pod} issues in {namespace} cluster",
        "Check health of {pod} in {namespace}",
    ]

    K8S_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Fix {pod} in {namespace} namespace",
        "Diagnose {pod} problems in {namespace}",
        "Resolve {pod} issues in {namespace} cluster",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # REGULATORY REQUIREMENTS - For static knowledge scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    REGULATORY_REQUIREMENTS: ClassVar[Dict[str, Dict]] = {
        "backup_before_delete": {
            "training_templates": [
                "Delete {stack} without backup",
                "Remove {stack} immediately - skip backup",
            ],
            "test_templates": [
                "Quick delete {stack} - no backup needed",
            ],
            "lesson": "Production stacks require backup before deletion. Compliance requirement.",
            "error": "BACKUP-REQUIRED-403",
            "stacks": ["prod-api-stack", "data-pipeline-stack"],
        },
        "change_approval": {
            "training_templates": [
                "Deploy {update_type} to production without approval",
                "Push {update_type} changes directly",
            ],
            "test_templates": [
                "Urgent {update_type} deployment - skip approval",
            ],
            "lesson": "Production changes require change approval. SOC2 compliance.",
            "error": "APPROVAL-REQUIRED-403",
            "stacks": ["prod-api-stack", "auth-service-stack"],
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # AGREEMENT SCENARIOS - Static and dynamic align
    # ═══════════════════════════════════════════════════════════════════════════
    RECOVERY_PAIRS: ClassVar[List[Tuple[str, str]]] = [
        ("prod-api-stack", "continue_update_rollback"),
        ("data-pipeline-stack", "continue_update_rollback"),
        ("auth-service-stack", "delete_and_recreate"),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-CONDITION VALID STRATEGIES - Strict enforcement for fair comparison
    # ═══════════════════════════════════════════════════════════════════════════
    # For multi-condition scenarios, ONLY these 2 strategies are valid.
    # Each condition_key is deterministically mapped to exactly ONE strategy.
    # This matches logistics domain difficulty where only antwerp/hamburg work.
    # ═══════════════════════════════════════════════════════════════════════════
    MULTI_CONDITION_VALID_STRATEGIES: ClassVar[List[str]] = [
        "continue_update_rollback",
        "wait_for_completion",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # SEMANTIC CONDITIONS - For compositional generalization
    # Maps semantic condition codes to solutions with tier priorities
    # Higher tier wins when conditions conflict (same as logistics)
    # ═══════════════════════════════════════════════════════════════════════════
    SEMANTIC_CONDITIONS: ClassVar[Dict[str, Dict]] = {
        # Tier 3 (Highest): Compliance - non-negotiable
        "PCI": {
            "solution": "wait_for_completion",
            "tier": 3,
            "meaning": "PCI-DSS compliant payment deployment",
        },
        "HIPAA": {
            "solution": "continue_update_rollback",
            "tier": 3,
            "meaning": "HIPAA-compliant healthcare deployment",
        },
        # Tier 2 (Middle): Reliability requirements
        "SCALE": {
            "solution": "continue_update_rollback",
            "tier": 2,
            "meaning": "High-scale deployment for traffic spikes",
        },
        "TEST": {
            "solution": "wait_for_completion",
            "tier": 2,
            "meaning": "Testing/staging deployment for validation",
        },
        # Tier 1 (Lowest): Performance/cost preferences
        "FAST": {
            "solution": "wait_for_completion",
            "tier": 1,
            "meaning": "Time-critical express deployment",
        },
        "CHEAP": {
            "solution": "continue_update_rollback",
            "tier": 1,
            "meaning": "Cost-optimized deployment",
        },
    }

    @classmethod
    def get_valid_strategy_for_conditions(cls, condition_key: str) -> str:
        """
        Get the ONLY valid strategy for a given condition_key.

        For SEMANTIC conditions (PCI, HIPAA, FAST, etc.):
        - Uses priority-based resolution (highest tier wins)
        - Solutions are DERIVABLE from atomic precepts
        - Enables P₁ > 0% through compositional reasoning

        For BLACK SWAN conditions (CFN-420, IAM-403, etc.):
        - Uses deterministic hash
        - Solutions are NOT derivable (require exploration)
        - P₁ = 0% expected (first-try success unlikely)
        """
        # Check if this is a semantic condition
        conditions = condition_key.split("+")
        is_semantic = all(
            c.strip().upper() in cls.SEMANTIC_CONDITIONS for c in conditions
        )

        if is_semantic:
            # SEMANTIC MODE: Priority-based resolution (highest tier wins)
            best_cond = None
            best_tier = -1
            for cond in sorted(conditions):
                cond = cond.strip().upper()
                if cond in cls.SEMANTIC_CONDITIONS:
                    tier = cls.SEMANTIC_CONDITIONS[cond]["tier"]
                    if tier > best_tier:
                        best_tier = tier
                        best_cond = cond
            if best_cond:
                return cls.SEMANTIC_CONDITIONS[best_cond]["solution"]
            return "wait_for_completion"  # Fallback
        else:
            # BLACK SWAN MODE: Hash-based (solutions NOT derivable)
            # FIX: Use hashlib.md5 for deterministic hashing across sessions
            import hashlib

            hash_bytes = hashlib.md5(condition_key.encode()).digest()
            hash_val = int.from_bytes(hash_bytes[:8], byteorder="big")
            idx = hash_val % len(cls.MULTI_CONDITION_VALID_STRATEGIES)
            return cls.MULTI_CONDITION_VALID_STRATEGIES[idx]


# Convenience function to get config
def get_devops_config() -> type:
    """Get the DevOps configuration class."""
    return DevOpsConfig
