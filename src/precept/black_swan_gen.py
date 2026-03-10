import json
import uuid
import random
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# ==============================================================================
# 1. VARIATIONAL ARTIFACT ENGINE (COMPLETE)
# ==============================================================================
# Contains authentic log templates for every category, including "Cryptic/Opaque" variants.

REAL_WORLD_LOG_VARIANTS = {
    
    # --- CODING & DEVELOPMENT ---
    "PIP_INSTALL_FAIL": [
        """ERROR: Could not find a version that satisfies the requirement {lib_name} (from versions: none)""",
        """WARNING: Retrying (Retry(total=0...)) after connection broken by 'NewConnectionError': [Errno -2] Name or service not known.""",
        """PackagesNotFoundError: The following packages are not available from current channels: - {lib_name}"""
    ],
    "PYTHON_IMPORT_ERROR": [
        """ModuleNotFoundError: No module named '{lib_name}'""",
        """ImportError: cannot import name 'Core' from '{lib_name}' (unknown location)""",
        """ImportError: cannot import name '{lib_name}' from partially initialized module '{lib_name}' (circular import)"""
    ],
    "PYTHON_SEGFAULT": [
        """Segmentation fault (core dumped)""",
        """Bus error (core dumped)""",
        """# A fatal error has been detected by the Java Runtime Environment: SIGSEGV (0xb)"""
    ],
    "CONCURRENCY_ERROR": [
        """sqlalchemy.exc.IntegrityError: (psycopg2.errors.UniqueViolation) duplicate key value violates unique constraint""",
        """mysql.connector.errors.DatabaseError: 1205 (HY000): Lock wait timeout exceeded""",
        """redis.exceptions.WatchError: Watched variable changed."""
    ],

    # --- DEVOPS & INFRASTRUCTURE ---
    "AWS_ROLLBACK_FAILED": [
        """Stack:{stack_name} is in UPDATE_ROLLBACK_FAILED state and can not be updated.""",
        """{{ "ResourceStatus": "UPDATE_ROLLBACK_FAILED", "ResourceStatusReason": "Failed to update [EC2Instance]." }}""",
        """botocore.exceptions.WaiterError: Waiter StackUpdateComplete failed: terminal failure state UPDATE_ROLLBACK_FAILED."""
    ],
    "AWS_IAM_RACE": [
        """botocore.errorfactory.NoSuchEntityException: The role with name {role_name} cannot be found.""",
        """ClientError: An error occurred (AccessDenied) when calling the AssumeRole operation: User not authorized to perform sts:AssumeRole."""
    ],
    "AWS_MASKED_403": [
        """An error occurred (AccessDenied) when calling the CreateBucket operation: Access Denied""",
        """<html><body><h1>403 Forbidden</h1><center>Request ID: {req_id}</center></body></html>""", # WAF style
        """An error occurred (AccessDeniedException) when calling Decrypt: The ciphertext refers to a key that does not exist or you cannot access.""" # KMS style
    ],
    "K8S_EVICTION": [
        """Event(v1.ObjectReference{{Kind:"Pod", Name:"{pod_name}"}}): type: 'Warning' reason: 'Evicted' low on resource: ephemeral-storage.""",
        """State: Terminated | Reason: OOMKilled | Exit Code: 137"""
    ],

    # --- LOGISTICS & SUPPLY CHAIN ---
    "EDI_REJECTION": [
        """EDI 997 FUNCTIONAL ACKNOWLEDGMENT | Status: REJECTED (R) | Error Code: 110 - Vessel Not Authorized for Port {port_code}.""",
        """{{ "status": "EXCEPTION", "exception_code": "SANCTION_BLOCK_404", "message": "Port {port_code} is embargoed." }}"""
    ],
    "CUSTOMS_HOLD": [
        """STATUS UPDATE: HELD_BY_CUSTOMS | Reason: HS Code {hs_code} requires Permit Type C.""",
        """ALERT: Shipment {shipment_id} flagged for inspection. Clearance Status: PENDING_DOCS ({port_name})"""
    ],

    # --- BOOKING & TRAVEL ---
    "HTTP_200_LIE": [
        """HTTP/1.1 200 OK\n{{ "data": null, "errors": [ {{ "code": "INVENTORY_ALLOCATION_FAIL", "status": "DECLINED" }} ] }}""",
        """HTTP/1.1 200 OK\n{{ "success": false, "error": "Seat selection not confirmed." }}""",
        """HTTP/1.1 200 OK\n{{ "booking_ref": "{req_id}", "status": "FAILED_BACKEND_TIMEOUT" }}"""
    ],
    "PAYMENT_GATEWAY_TIMEOUT": [
        """Gateway Error 504: Upstream Provider Timeout | Transaction ID: {txn_id}""",
        """urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='payment.api'): Read timed out."""
    ],

    # --- FINANCE & TRADING ---
    "FIX_REJECT": [
        """8=FIX.4.4|35=3|58=REJECT: Order {order_id} exceeds MaxNotional limit.|10=044""",
        """8=FIX.4.2|35=3|58=Incorrect NumInGroup count for repeating group|10=129"""
    ],
    "MARKET_DATA_GAP": [
        """WARNING: Data Gap Detected | Symbol: {symbol} | Gap Duration: 4500ms""",
        """[Strategy-VWAP] Critical: Last tick for {symbol} received 10s ago. Halting execution."""
    ],

    # --- INTEGRATION & API ---
    "OAUTH_ZOMBIE": [
        """HTTP/1.1 401 Unauthorized\nWWW-Authenticate: Bearer error="invalid_token" """,
        """HTTP/1.1 403 Forbidden\n{{ "error": "access_denied", "message": "Token expired at {timestamp}" }}"""
    ],
    "API_OPAQUE_500": [
        """502 Bad Gateway\nnginx/1.18.0 (Ubuntu)""",
        """Error 524: A timeout occurred. The origin web server timed out.""",
        """HTTP/1.1 500 Internal Server Error\n{{ "error": "InternalServerError", "requestId": "{req_id}" }}"""
    ],
    "API_THROTTLE_SILENT": [
        """HTTP/1.1 429 Too Many Requests\nRetry-After: 3600""",
        """[INFO] Request to {entity} completed in 45.2s (Normal avg: 0.2s).""" # Soft throttle
    ]
}

# ==============================================================================
# 2. MASTER ONTOLOGY (ALL CATEGORIES & TRAPS)
# ==============================================================================

BLACK_SWAN_DEFINITIONS = {
    "Coding": {
        "Dependency_Zombie": {
            "description": "Agent tries to install a deleted/missing library.",
            "log_type": "PIP_INSTALL_FAIL",
            "trap_template": "Install '{entity}' for the project.",
            "precept_lesson": "If pip fails, switch to writing raw code."
        },
        "Opaque_Crash": {
            "description": "Agent triggers a Segfault/Bus Error with no stack trace.",
            "log_type": "PYTHON_SEGFAULT",
            "trap_template": "Run the optimized C-wrapper for '{entity}'.",
            "precept_lesson": "Segfaults require 'faulthandler' or library version rollback."
        },
        "Concurrency_Race": {
            "description": "Race condition in check-then-act logic.",
            "log_type": "CONCURRENCY_ERROR",
            "trap_template": "Register '{entity}' if unique.",
            "precept_lesson": "Use DB Constraints, not Python checks."
        },
        "Import_Hell": {
            "description": "Circular imports or missing submodules.",
            "log_type": "PYTHON_IMPORT_ERROR",
            "trap_template": "Refactor '{entity}' imports.",
            "precept_lesson": "Check for circular dependencies."
        }
    },
    "DevOps": {
        "Zombie_Stack": {
            "description": "Agent tries to update a stuck CloudFormation stack.",
            "log_type": "AWS_ROLLBACK_FAILED",
            "trap_template": "Update Stack '{entity}'.",
            "precept_lesson": "If ROLLBACK_FAILED, use 'continue_update_rollback'."
        },
        "Consistency_Race": {
            "description": "Agent ignores IAM propagation delay.",
            "log_type": "AWS_IAM_RACE",
            "trap_template": "Create Role '{entity}' and use immediately.",
            "precept_lesson": "Implement waiters/sleeps after IAM creation."
        },
        "Hidden_Policy_Block": {
            "description": "Agent hits SCP/WAF block masked as generic 403.",
            "log_type": "AWS_MASKED_403",
            "trap_template": "Deploy '{entity}' in restricted region.",
            "precept_lesson": "Generic AccessDenied implies SCP/WAF. Check Org limits."
        },
        "Pod_Eviction": {
            "description": "Pod dies due to ephemeral storage/OOM.",
            "log_type": "K8S_EVICTION",
            "trap_template": "Debug Pod '{entity}' restart loop.",
            "precept_lesson": "Check resource limits and storage."
        }
    },
    "Logistics": {
        "Sovereign_Blockade": {
            "description": "Sanctions contradict RAG SOPs.",
            "log_type": "EDI_REJECTION",
            "trap_template": "Ship '{entity}' to embargoed port.",
            "precept_lesson": "EDI signals override static RAG docs."
        },
        "Hazmat_Drift": {
            "description": "Regulatory change halts shipment.",
            "log_type": "CUSTOMS_HOLD",
            "trap_template": "Clear customs for '{entity}'.",
            "precept_lesson": "Verify HS Code regulations dynamically."
        }
    },
    "Booking": {
        "Phantom_Inventory": {
            "description": "HTTP 200 OK but body is error.",
            "log_type": "HTTP_200_LIE",
            "trap_template": "Book flight '{entity}'.",
            "precept_lesson": "Validate JSON body 'status', ignore HTTP code."
        },
        "Gateway_Timeout": {
            "description": "Payment hangs upstream.",
            "log_type": "PAYMENT_GATEWAY_TIMEOUT",
            "trap_template": "Charge card for '{entity}'.",
            "precept_lesson": "Use Idempotency Keys on retries."
        }
    },
    "Finance": {
        "Volatility_Reject": {
            "description": "FIX Protocol rejects due to volatility.",
            "log_type": "FIX_REJECT",
            "trap_template": "Execute market order for '{entity}'.",
            "precept_lesson": "Parse FIX tag 58 for logic."
        },
        "Stale_Data": {
            "description": "Agent trades on stale data (gaps).",
            "log_type": "MARKET_DATA_GAP",
            "trap_template": "Calculate VWAP for '{entity}'.",
            "precept_lesson": "Detect data gaps >1s and halt."
        }
    },
    "Integration": {
        "Auth_Zombie": {
            "description": "Token expires; agent retries without refreshing.",
            "log_type": "OAUTH_ZOMBIE",
            "trap_template": "Sync '{entity}' data.",
            "precept_lesson": "401/403 means Re-Auth, never Retry."
        },
        "Gateway_Masking": {
            "description": "Generic 500/502 masks upstream failure.",
            "log_type": "API_OPAQUE_500",
            "trap_template": "Connect to ERP '{entity}'.",
            "precept_lesson": "500/502 is environmental. Do not change code."
        },
        "Silent_Throttling": {
            "description": "Latency spikes instead of 429.",
            "log_type": "API_THROTTLE_SILENT",
            "trap_template": "Bulk upload to '{entity}'.",
            "precept_lesson": "Rising latency is backpressure. Back off."
        }
    }
}

# ==============================================================================
# 3. GENERATOR ENGINE
# ==============================================================================

@dataclass
class SyntheticSample:
    id: str
    category: str
    sub_category: str
    difficulty: int
    user_query: str
    ground_truth_log: str
    hidden_trap: Dict[str, Any]
    precept_instinct: str

class UniversalDataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.categories = list(BLACK_SWAN_DEFINITIONS.keys())

    def _fill_log_template(self, log_type: str, context: Dict) -> str:
        """Selects a random variant and fills dynamic data."""
        variants = REAL_WORLD_LOG_VARIANTS.get(log_type, [f"Error: {log_type}"])
        template = random.choice(variants)
        
        # Real-world randomness
        context['uuid'] = str(uuid.uuid4())[:8]
        context['req_id'] = f"req_{uuid.uuid4()}"
        context['txn_id'] = f"txn_{uuid.uuid4()}"
        context['timestamp'] = int(datetime.datetime.utcnow().timestamp())
        
        return template.format(**context)

    def generate_sample(self, category, sub_type, definition) -> SyntheticSample:
        # Realistic Entities
        entities = {
            "Coding": ["fast_xml", "auth_lib_v1", "numpy_mkl"],
            "DevOps": ["Prod-Stack", "Deploy-Role", "redis-pod-x"],
            "Logistics": ["SHP-992", "CNTR-440", "Maersk-X"],
            "Booking": ["Resv-882", "Flight-99", "Hotel-Z"],
            "Finance": ["ORD-220", "AAPL", "BTC-USD"],
            "Integration": ["Salesforce_Sync", "Twitter_Stream", "Oracle_ERP"]
        }
        entity = random.choice(entities.get(category, ["Entity_X"]))
        
        # Context Variables
        log_vars = {
            "lib_name": entity, "stack_name": entity, "entity": entity,
            "role_name": "DeployRole_v1", "shipment_id": f"SHP-{entity}",
            "port_code": "CN-SHA", "port_name": "Shanghai", "hs_code": "8542.31",
            "pod_name": f"{entity}-pod", "order_id": f"ORD-{entity}", 
            "symbol": entity, "entity_id": "U-1001"
        }
        
        real_log = self._fill_log_template(definition['log_type'], log_vars)
        
        return SyntheticSample(
            id=str(uuid.uuid4())[:8],
            category=category,
            sub_category=sub_type,
            difficulty=4, # "Cryptic/Black Swan" is max difficulty
            user_query=definition['trap_template'].format(entity=entity),
            ground_truth_log=real_log.strip(),
            hidden_trap={
                "trigger": "Execution", 
                "response": real_log.strip(),
                "root_cause": definition['description']
            },
            precept_instinct=definition['precept_lesson']
        )

    def run(self, output_file="precept_ultimate_training.jsonl"):
        dataset = []
        samples_per_cat = max(1, self.num_samples // len(self.categories))
        
        print(f"Generating {self.num_samples} samples across {len(self.categories)} categories...")
        
        for category in self.categories:
            sub_cats = BLACK_SWAN_DEFINITIONS[category]
            for _ in range(samples_per_cat):
                sub_type = random.choice(list(sub_cats.keys()))
                sample = self.generate_sample(category, sub_type, sub_cats[sub_type])
                dataset.append(asdict(sample))
        
        random.shuffle(dataset)
        
        with open(output_file, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        
        print(f"Done. Saved {len(dataset)} scenarios to {output_file}.")

if __name__ == "__main__":
    gen = UniversalDataGenerator(num_samples=100)
    gen.run()
