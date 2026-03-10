"""
Simulation Framework for PRECEPT Testing.

Provides simulation environments with configurable "black swan" events
for testing agent learning and adaptation capabilities.

Key Components:
- SimulationWorld: Base class for simulation environments
- BlackSwanWorld: Logistics simulation with hidden rules
- SimulationRule: Defines hidden constraints the agent must discover
- ToolExecutor: Factory for creating domain-specific tool executors

Black Swan events are rare, unexpected situations that:
1. Are not in the agent's training data
2. Cause initial task failures
3. Can be learned from through experience
4. Test the agent's adaptation capabilities

Usage:
    from precept.simulation import BlackSwanWorld, create_logistics_tool_executor
    
    # Configure black swan rules
    world = BlackSwanWorld()
    world.CURRENT_DAY = "Tuesday"  # Triggers Rotterdam closure
    
    # Create tool executor
    tool_executor = create_logistics_tool_executor(world)
    
    # Run agent with tools
    result = await agent.run_task("Ship from Rotterdam to Boston")
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


# =============================================================================
# SIMULATION TYPES
# =============================================================================

class RuleType(Enum):
    """Type of simulation rule."""
    TIME_BASED = "time_based"  # Rule depends on time/day
    CARGO_BASED = "cargo_based"  # Rule depends on cargo type
    CAPACITY_BASED = "capacity_based"  # Rule depends on capacity
    COMPLIANCE_BASED = "compliance_based"  # Rule depends on regulations


@dataclass
class SimulationRule:
    """
    A hidden rule in the simulation that the agent must discover.
    
    Attributes:
        rule_id: Unique identifier for the rule
        rule_type: Type of rule (time, cargo, capacity, compliance)
        description: Human-readable description
        trigger_condition: Lambda that returns True when rule applies
        failure_message: Message shown when rule triggers
        alternative: Suggested alternative when rule triggers
        learning_opportunity: Whether agent can learn from this failure
    """
    rule_id: str
    rule_type: RuleType
    description: str
    trigger_condition: Callable[[Dict], bool]
    failure_message: str
    alternative: Optional[str] = None
    learning_opportunity: bool = True
    encounters_to_learn: int = 2


@dataclass
class SimulationResult:
    """Result of a simulation action."""
    success: bool
    message: str
    rule_triggered: Optional[str] = None
    alternative: Optional[str] = None
    learning_opportunity: bool = False
    can_retry_with: Optional[str] = None


# =============================================================================
# BLACK SWAN WORLD
# =============================================================================

class BlackSwanWorld:
    """
    Simulated world with configurable BLACK SWAN rules.
    
    This simulation environment contains hidden rules that:
    1. Are not documented in the static knowledge base
    2. Cause task failures when triggered
    3. Provide learning opportunities after multiple encounters
    
    Default black swan scenarios:
    - Rotterdam closed on Tuesdays
    - Antwerp rejects hazmat cargo
    - Hamburg at capacity after 20th of month
    - NorthStar carrier unavailable on 1st/15th
    - Pharma to US requires QDNA-7 certificate
    
    Usage:
        world = BlackSwanWorld()
        world.CURRENT_DAY = "Tuesday"
        
        result = world.check_port("rotterdam", "standard")
        # Returns failure with alternative suggestion
        
        # After enough encounters, rules are learned
        learned = world.get_learned_rules()
    """
    
    # Configurable world state
    CURRENT_DAY: str = "Tuesday"
    CURRENT_DATE: int = 15
    
    # Tracking (class-level for persistence across instances)
    encounter_count: Dict[str, int] = {}
    learned_rules: Dict[str, str] = {}
    
    # Multi-objective tracking
    objective_scores: Dict[str, List[float]] = {
        "success_rate": [],
        "step_efficiency": [],
        "adaptation_speed": [],
        "rule_generalization": [],
    }
    
    @classmethod
    def reset(cls):
        """Reset all world state (for baseline comparisons)."""
        cls.encounter_count = {}
        cls.learned_rules = {}
        cls.objective_scores = {k: [] for k in cls.objective_scores}
    
    @classmethod
    def reset_encounters(cls):
        """Reset only encounter counts (keep learned rules)."""
        cls.encounter_count = {}
    
    @classmethod
    def check_port(cls, port: str, cargo_type: str = "standard") -> Dict:
        """
        Check if a port is available for the given cargo type.
        
        Args:
            port: Port name (rotterdam, hamburg, antwerp, boston)
            cargo_type: Type of cargo (standard, pharma, hazmat)
        
        Returns:
            Dictionary with success status, message, and alternatives
        """
        port = port.lower()
        
        # BLACK SWAN 1: Rotterdam closed Tuesdays
        if port == "rotterdam" and cls.CURRENT_DAY == "Tuesday":
            cls.encounter_count["rotterdam_tuesday"] = cls.encounter_count.get("rotterdam_tuesday", 0) + 1
            if cls.encounter_count["rotterdam_tuesday"] >= 2:
                cls.learned_rules["rotterdam_tuesday"] = "Rotterdam closed Tuesdays → use Hamburg"
                return {
                    "success": False,
                    "message": f"ERROR: Rotterdam CLOSED (Tuesday maintenance). ALTERNATIVE: Hamburg/Antwerp.",
                    "alternative": "hamburg",
                    "reason": "rotterdam_tuesday",
                    "learning_opportunity": True,
                }
            return {
                "success": False,
                "message": f"ERROR: Rotterdam is CLOSED today.",
                "reason": "rotterdam_tuesday",
            }
        
        # BLACK SWAN 2: Antwerp hazmat restricted
        if port == "antwerp" and "hazmat" in cargo_type.lower():
            cls.encounter_count["antwerp_hazmat"] = cls.encounter_count.get("antwerp_hazmat", 0) + 1
            if cls.encounter_count["antwerp_hazmat"] >= 2:
                cls.learned_rules["antwerp_hazmat"] = "Antwerp rejects hazmat → use Hamburg"
                return {
                    "success": False,
                    "message": f"ERROR: Antwerp REJECTS hazmat. ALTERNATIVE: Hamburg.",
                    "alternative": "hamburg",
                    "reason": "antwerp_hazmat",
                    "learning_opportunity": True,
                }
            return {"success": False, "message": f"ERROR: Antwerp cannot accept hazmat cargo."}
        
        # BLACK SWAN 3: Hamburg capacity limits
        if port == "hamburg" and cls.CURRENT_DATE > 20:
            cls.encounter_count["hamburg_capacity"] = cls.encounter_count.get("hamburg_capacity", 0) + 1
            if cls.encounter_count["hamburg_capacity"] >= 2:
                cls.learned_rules["hamburg_capacity"] = "Hamburg at capacity after 20th → use Rotterdam (if not Tuesday)"
            return {"success": False, "message": f"ERROR: Hamburg at capacity end of month."}
        
        return {"success": True, "message": f"Port {port.title()} is OPERATIONAL."}
    
    @classmethod
    def check_carrier(cls, carrier: str) -> Dict:
        """
        Check if a carrier is available.
        
        Args:
            carrier: Carrier name (northstar, maersk, msc)
        
        Returns:
            Dictionary with success status and alternatives
        """
        carrier = carrier.lower()
        
        # BLACK SWAN: NorthStar blackout on 1st/15th
        if "northstar" in carrier and cls.CURRENT_DATE in [1, 15]:
            cls.encounter_count["northstar_blackout"] = cls.encounter_count.get("northstar_blackout", 0) + 1
            if cls.encounter_count["northstar_blackout"] >= 2:
                cls.learned_rules["northstar_blackout"] = "NorthStar unavailable 1st/15th → use Maersk"
                return {
                    "success": False,
                    "message": f"ERROR: NorthStar BLACKOUT. ALTERNATIVE: Maersk/MSC.",
                    "alternative": "maersk",
                    "learning_opportunity": True,
                }
            return {"success": False, "message": f"ERROR: NorthStar unavailable."}
        
        return {"success": True, "message": f"Carrier {carrier.title()} AVAILABLE."}
    
    @classmethod
    def check_compliance(cls, cargo_type: str, destination: str) -> Dict:
        """
        Check regulatory compliance for cargo to destination.
        
        Args:
            cargo_type: Type of cargo (standard, pharma, hazmat)
            destination: Destination port/country
        
        Returns:
            Dictionary with compliance status
        """
        cargo = cargo_type.lower()
        dest = destination.lower()
        
        # BLACK SWAN: QDNA-7 for pharma to US
        if "pharma" in cargo and dest in ["boston", "new_york", "us", "usa"]:
            cls.encounter_count["qdna7"] = cls.encounter_count.get("qdna7", 0) + 1
            if cls.encounter_count["qdna7"] >= 2:
                cls.learned_rules["qdna7"] = "Pharma to US requires QDNA-7 certificate"
                return {
                    "success": False,
                    "message": f"ERROR: Missing QDNA-7 for pharma to US. REQUIREMENT: Obtain QDNA-7.",
                    "requirement": "qdna7_certificate",
                    "learning_opportunity": True,
                }
            return {"success": False, "message": f"ERROR: Compliance failed for pharma to {dest}."}
        
        return {"success": True, "message": f"Compliance PASSED for {cargo} to {dest}."}
    
    @classmethod
    def book_shipment(
        cls,
        origin: str,
        destination: str,
        cargo_type: str,
        carrier: str = "standard",
    ) -> Dict:
        """
        Book a shipment from origin to destination.
        
        Args:
            origin: Origin port
            destination: Destination port
            cargo_type: Type of cargo
            carrier: Carrier to use
        
        Returns:
            Dictionary with booking result
        """
        # Check port availability
        port_check = cls.check_port(origin, cargo_type)
        if not port_check["success"]:
            if "alternative" in port_check:
                return {
                    "success": False,
                    "message": f"{port_check['message']}",
                    "can_retry_with": port_check["alternative"],
                }
            return {"success": False, "message": f"BOOKING FAILED: {port_check['message']}"}
        
        # Check carrier availability
        carrier_check = cls.check_carrier(carrier)
        if not carrier_check["success"]:
            if "alternative" in carrier_check:
                return {
                    "success": False,
                    "message": f"{carrier_check['message']}",
                    "can_retry_with": carrier_check["alternative"],
                }
            return {"success": False, "message": f"BOOKING FAILED: {carrier_check['message']}"}
        
        # Check compliance
        compliance_check = cls.check_compliance(cargo_type, destination)
        if not compliance_check["success"]:
            return {"success": False, "message": f"BOOKING FAILED: {compliance_check['message']}"}
        
        # Success!
        booking_id = f"BK-{random.randint(100000, 999999)}"
        return {
            "success": True,
            "message": f"BOOKING CONFIRMED! ID: {booking_id}. {origin.title()} → {destination.title()}. Task completed successfully.",
        }
    
    @classmethod
    def get_learned_rules(cls) -> List[str]:
        """Get all learned rules as a list."""
        return list(cls.learned_rules.values())
    
    @classmethod
    def get_learned_rules_dict(cls) -> Dict[str, str]:
        """Get learned rules as a dictionary."""
        return cls.learned_rules.copy()
    
    @classmethod
    def record_objective_score(cls, objective: str, score: float):
        """Record a score for an objective."""
        if objective in cls.objective_scores:
            cls.objective_scores[objective].append(score)
    
    @classmethod
    def get_state_summary(cls) -> Dict:
        """Get summary of current world state."""
        return {
            "current_day": cls.CURRENT_DAY,
            "current_date": cls.CURRENT_DATE,
            "encounters": cls.encounter_count.copy(),
            "learned_rules": cls.learned_rules.copy(),
            "rules_count": len(cls.learned_rules),
        }


# =============================================================================
# TOOL EXECUTOR FACTORY
# =============================================================================

def create_logistics_tool_executor(
    world: type = BlackSwanWorld,
    auto_retry_alternatives: bool = True,
) -> Callable:
    """
    Create a tool executor for logistics simulation.
    
    Args:
        world: The simulation world class to use
        auto_retry_alternatives: Whether to auto-retry with alternatives on failure
    
    Returns:
        Async tool executor function
    """
    
    async def tool_executor(action_type: str, action_content: str, state: Any = None) -> str:
        """
        Execute logistics tool actions.
        
        Supports:
        - check port [name] - Check port status
        - check carrier [name] - Check carrier availability
        - check compliance [cargo] [dest] - Check regulations
        - book shipment from [origin] to [dest] - Execute booking
        """
        action = f"{action_type} {action_content}".lower()
        
        # Detect cargo type
        cargo_type = "standard"
        if "pharma" in action:
            cargo_type = "pharma"
        elif "hazmat" in action:
            cargo_type = "hazmat"
        
        # BOOKING - with smart retry
        if "book" in action:
            origin = dest = carrier = None
            
            # Parse origin
            for port in ["rotterdam", "antwerp", "hamburg"]:
                if f"from {port}" in action or port in action.split()[:3]:
                    origin = port
                    break
            
            # Parse destination
            for port in ["boston", "new_york", "shanghai"]:
                if f"to {port}" in action or port in action:
                    dest = port
                    break
            
            # Parse carrier
            for c in ["northstar", "maersk", "msc"]:
                if c in action:
                    carrier = c
            
            if origin and dest:
                result = world.book_shipment(origin, dest, cargo_type, carrier or "standard")
                
                # Auto-retry with alternative if available
                if auto_retry_alternatives and not result["success"] and "can_retry_with" in result:
                    alt_origin = result["can_retry_with"]
                    alt_result = world.book_shipment(alt_origin, dest, cargo_type, carrier or "standard")
                    if alt_result["success"]:
                        return f"Original port blocked. AUTO-REROUTED via {alt_origin.title()}. {alt_result['message']}"
                    return f"{result['message']} Tried alternative {alt_origin} but also failed: {alt_result['message']}"
                
                return result["message"]
            
            # Fallback - use Hamburg if no origin specified
            if dest and not origin:
                result = world.book_shipment("hamburg", dest, cargo_type, carrier or "standard")
                return f"Using Hamburg as origin. {result['message']}"
            
            return "Specify: book from [origin] to [destination]"
        
        # CHECK PORT
        if any(p in action for p in ["check port", "port status", "verify port"]):
            for port in ["rotterdam", "antwerp", "hamburg", "boston"]:
                if port in action:
                    result = world.check_port(port, cargo_type)
                    if not result["success"] and "alternative" in result:
                        return f"{result['message']} RECOMMENDATION: Use {result['alternative'].title()} instead and book directly."
                    return result["message"]
            return "Specify port: rotterdam, hamburg, antwerp, boston"
        
        # CHECK CARRIER
        if any(p in action for p in ["check carrier", "carrier status", "verify carrier"]):
            for carrier in ["northstar", "maersk", "msc"]:
                if carrier in action:
                    result = world.check_carrier(carrier)
                    if not result["success"] and "alternative" in result:
                        return f"{result['message']} RECOMMENDATION: Use {result['alternative'].title()} instead."
                    return result["message"]
            return "Specify carrier: northstar, maersk, msc"
        
        # CHECK COMPLIANCE
        if "compliance" in action:
            dest = "boston" if "boston" in action else "new_york" if "new_york" in action else "us"
            result = world.check_compliance(cargo_type, dest)
            return result["message"]
        
        # USE ALTERNATIVE / SWITCH
        if any(p in action for p in ["use alternative", "switch to", "try instead", "use hamburg", "book via"]):
            for alt in ["hamburg", "antwerp"]:
                if alt in action:
                    for dest in ["boston", "new_york", "shanghai"]:
                        if dest in action:
                            result = world.book_shipment(alt, dest, cargo_type, "standard")
                            return result["message"]
                    return f"Switching to {alt.title()}. Now book from {alt} to [destination]."
            return "Specify alternative port"
        
        # DIRECT HAMBURG BOOKING SHORTCUT
        if "hamburg" in action and any(d in action for d in ["boston", "new_york", "shanghai"]):
            for dest in ["boston", "new_york", "shanghai"]:
                if dest in action:
                    result = world.book_shipment("hamburg", dest, cargo_type, "standard")
                    return result["message"]
        
        return f"Action '{action_type}' noted. Use: book from [origin] to [destination]"
    
    return tool_executor


# =============================================================================
# STATIC KNOWLEDGE BASE
# =============================================================================

def get_logistics_knowledge_base() -> Dict[str, Any]:
    """
    Get standard logistics knowledge base.
    
    This represents the "static" knowledge that would be ingested
    into a vector database during hard ingestion.
    
    Note: This does NOT include black swan rules - those must be learned!
    
    Returns:
        Dictionary with ports, carriers, and compliance info
    """
    return {
        "ports": {
            "rotterdam": {
                "name": "Rotterdam",
                "country": "Netherlands",
                "status": "operational",
                "capabilities": ["standard", "pharma", "hazmat"],
            },
            "hamburg": {
                "name": "Hamburg",
                "country": "Germany",
                "status": "operational",
                "capabilities": ["standard", "pharma", "hazmat"],
            },
            "antwerp": {
                "name": "Antwerp",
                "country": "Belgium",
                "status": "operational",
                "capabilities": ["standard", "pharma"],  # Note: no hazmat in static KB
            },
            "boston": {
                "name": "Boston",
                "country": "USA",
                "status": "operational",
                "capabilities": ["standard", "pharma"],
            },
        },
        "carriers": {
            "maersk": {
                "name": "Maersk",
                "reliability": "high",
                "coverage": "global",
            },
            "msc": {
                "name": "MSC",
                "reliability": "high",
                "coverage": "global",
            },
            "northstar": {
                "name": "NorthStar",
                "reliability": "medium",
                "coverage": "regional",
            },
        },
        "compliance": {
            "pharma": [
                "Temperature monitoring",
                "GDP certification",
            ],
            "hazmat": [
                "IMDG classification",
                "Safety sheets",
            ],
        },
        "routes": {
            "transatlantic": {
                "typical_ports": ["rotterdam", "hamburg", "antwerp"],
                "destinations": ["boston", "new_york"],
                "transit_time": "7-10 days",
            },
        },
    }


# =============================================================================
# HARD MODE BLACK SWAN WORLD (CRYPTIC ERRORS - NO HINTS)
# =============================================================================

class HardModeBlackSwanWorld:
    """
    REALISTIC Black Swan simulation with CRYPTIC error codes.
    
    This version:
    - Uses error codes instead of helpful messages (e.g., "Error R-482" not "Rotterdam closed")
    - Provides NO alternatives in error messages
    - Requires agents to DISCOVER patterns through repeated failures
    - Is designed to show PRECEPT's true learning advantage over RAG+Tools
    
    KEY DESIGN: Simple guessing strategies (like "always try Hamburg") will FAIL
    because we have MULTIPLE interacting black swans that require LEARNING:
    
    - Rotterdam: Closed Tuesdays (R-482)
    - Hamburg: Blocked for US destinations on 15th (H-903) 
    - Antwerp: Only works for non-pharma (A-117)
    - The ONLY solution: Rotterdam→Shanghai, Hamburg→Boston, Antwerp→NY (non-pharma)
    
    The LLM cannot "reason" through these - it must LEARN from experience.
    """
    
    # World state
    CURRENT_DAY = "Tuesday"
    CURRENT_DATE = 15  # Hamburg blocked for US
    
    # Error code mapping (internal - not revealed to agents)
    ERROR_CODES = {
        "R-482": "rotterdam_tuesday",      # Rotterdam closed Tuesdays
        "A-117": "antwerp_hazmat",          # Antwerp no hazmat/pharma
        "H-903": "hamburg_us_15th",         # Hamburg blocked for US on 15th
        "N-551": "northstar_blackout",      # NorthStar blackout
        "Q-739": "qdna7_required",          # QDNA-7 certificate
        "X-666": "unknown_failure",         # Unknown/random failure
    }
    
    # Tracking
    encounter_count: Dict[str, int] = {}
    learned_rules: Dict[str, str] = {}  # Only populated when agent discovers pattern
    failure_history: List[Dict] = []
    
    @classmethod
    def reset(cls):
        """Reset all state."""
        cls.encounter_count = {}
        cls.learned_rules = {}
        cls.failure_history = []
    
    @classmethod
    def reset_encounters(cls):
        """Reset encounters but keep learned rules."""
        cls.encounter_count = {}
    
    @classmethod
    def check_port(cls, port: str, cargo_type: str = "standard", destination: str = "") -> Dict:
        """Check port with CRYPTIC error codes (no hints)."""
        port = port.lower()
        dest = destination.lower() if destination else ""
        
        # BLACK SWAN 1: Rotterdam closed Tuesdays - CRYPTIC
        if port == "rotterdam" and cls.CURRENT_DAY == "Tuesday":
            cls.encounter_count["R-482"] = cls.encounter_count.get("R-482", 0) + 1
            count = cls.encounter_count["R-482"]
            
            # After 3+ failures, agent MIGHT discover the pattern
            if count >= 3:
                cls.learned_rules["R-482"] = "Error R-482 occurs at Rotterdam → try different port"
            
            cls.failure_history.append({
                "action": f"check_port({port})",
                "error": "R-482",
                "count": count,
            })
            
            # CRYPTIC error - no hint about alternative!
            return {
                "success": False,
                "message": f"BOOKING FAILED. Error code: R-482. Contact support.",
                "error_code": "R-482",
            }
        
        # BLACK SWAN 2: Antwerp rejects pharma cargo - CRYPTIC
        if port == "antwerp" and cargo_type.lower() in ["pharma", "hazmat"]:
            cls.encounter_count["A-117"] = cls.encounter_count.get("A-117", 0) + 1
            count = cls.encounter_count["A-117"]
            
            if count >= 3:
                cls.learned_rules["A-117"] = "Error A-117 occurs with special cargo at Antwerp → standard only"
            
            return {
                "success": False,
                "message": f"BOOKING FAILED. Error code: A-117. Cargo rejected.",
                "error_code": "A-117",
            }
        
        # BLACK SWAN 3: Hamburg blocked for US destinations on 15th - CRYPTIC
        # This is the KEY rule that makes simple guessing fail!
        if port == "hamburg" and cls.CURRENT_DATE == 15:
            us_dests = ["boston", "new_york", "new york", "us", "usa"]
            if any(d in dest for d in us_dests):
                cls.encounter_count["H-903"] = cls.encounter_count.get("H-903", 0) + 1
                count = cls.encounter_count["H-903"]
                
                if count >= 3:
                    cls.learned_rules["H-903"] = "Error H-903 occurs Hamburg→US on 15th → use Antwerp for US"
                
                return {
                    "success": False,
                    "message": f"BOOKING FAILED. Error code: H-903. Route unavailable.",
                    "error_code": "H-903",
                }
        
        return {"success": True, "message": f"Port {port.title()} OPERATIONAL. Ready for booking."}
    
    @classmethod
    def check_carrier(cls, carrier: str) -> Dict:
        """Check carrier with CRYPTIC error codes."""
        carrier = carrier.lower()
        
        # BLACK SWAN: NorthStar blackout - CRYPTIC
        if "northstar" in carrier and cls.CURRENT_DATE in [1, 15]:
            cls.encounter_count["N-551"] = cls.encounter_count.get("N-551", 0) + 1
            count = cls.encounter_count["N-551"]
            
            if count >= 3:
                cls.learned_rules["N-551"] = "Error N-551 occurs with NorthStar → try different carrier"
            
            return {
                "success": False,
                "message": f"CARRIER CHECK FAILED. Error code: N-551. System offline.",
                "error_code": "N-551",
            }
        
        return {"success": True, "message": f"Carrier {carrier.title()} AVAILABLE."}
    
    @classmethod
    def check_compliance(cls, cargo_type: str, destination: str) -> Dict:
        """Check compliance with CRYPTIC error codes."""
        cargo = cargo_type.lower()
        dest = destination.lower()
        
        # BLACK SWAN: QDNA-7 for pharma to US - CRYPTIC
        if "pharma" in cargo and dest in ["boston", "new_york", "us", "usa"]:
            cls.encounter_count["Q-739"] = cls.encounter_count.get("Q-739", 0) + 1
            count = cls.encounter_count["Q-739"]
            
            if count >= 3:
                cls.learned_rules["Q-739"] = "Error Q-739 occurs with pharma to US → certification issue"
            
            return {
                "success": False,
                "message": f"COMPLIANCE CHECK FAILED. Error code: Q-739. Documentation incomplete.",
                "error_code": "Q-739",
            }
        
        return {"success": True, "message": f"Compliance PASSED for {cargo} to {dest}."}
    
    @classmethod
    def book_shipment(
        cls,
        origin: str,
        destination: str,
        cargo_type: str = "standard",
        carrier: str = "standard",
    ) -> Dict:
        """Book shipment with CRYPTIC error propagation."""
        # Check port - pass destination for route-specific rules
        port_check = cls.check_port(origin, cargo_type, destination)
        if not port_check["success"]:
            return {
                "success": False,
                "message": port_check["message"],
                "error_code": port_check.get("error_code"),
            }
        
        # Check carrier
        carrier_check = cls.check_carrier(carrier)
        if not carrier_check["success"]:
            return {
                "success": False,
                "message": carrier_check["message"],
                "error_code": carrier_check.get("error_code"),
            }
        
        # Check compliance
        compliance_check = cls.check_compliance(cargo_type, destination)
        if not compliance_check["success"]:
            return {
                "success": False,
                "message": compliance_check["message"],
                "error_code": compliance_check.get("error_code"),
            }
        
        # Success!
        import random
        booking_id = f"BK-{random.randint(100000, 999999)}"
        return {
            "success": True,
            "message": f"BOOKING CONFIRMED. ID: {booking_id}. {origin.title()} → {destination.title()}. Task completed successfully.",
        }
    
    @classmethod
    def get_learned_rules(cls) -> List[str]:
        """Get learned rules as list."""
        return list(cls.learned_rules.values())
    
    @classmethod
    def get_learned_rules_dict(cls) -> Dict[str, str]:
        """Get learned rules as dict."""
        return cls.learned_rules.copy()
    
    @classmethod
    def get_failure_stats(cls) -> Dict:
        """Get statistics about failures encountered."""
        return {
            "total_failures": len(cls.failure_history),
            "unique_errors": len(cls.encounter_count),
            "errors_by_code": cls.encounter_count.copy(),
            "rules_discovered": len(cls.learned_rules),
        }


def create_hard_mode_tool_executor(world: type = HardModeBlackSwanWorld) -> Callable:
    """
    Create a HARD MODE tool executor with cryptic errors.
    
    NO auto-retry, NO hints, NO alternatives suggested.
    Agent must figure it out through learning.
    """
    
    async def tool_executor(action_type: str, action_content: str, state: Any = None) -> str:
        """Execute with cryptic error handling."""
        action = f"{action_type} {action_content}".lower()
        
        # Normalize "new york" to "new_york" for matching
        action = action.replace("new york", "new_york")
        
        # Detect cargo type
        cargo_type = "standard"
        if "pharma" in action:
            cargo_type = "pharma"
        elif "hazmat" in action:
            cargo_type = "hazmat"
        
        # BOOKING
        if "book" in action:
            origin = dest = carrier = None
            
            for port in ["rotterdam", "antwerp", "hamburg"]:
                if f"from {port}" in action or port in action.split()[:3]:
                    origin = port
                    break
            
            for port in ["boston", "new_york", "shanghai"]:
                if f"to {port}" in action or port in action:
                    dest = port
                    break
            
            for c in ["northstar", "maersk", "msc"]:
                if c in action:
                    carrier = c
            
            if origin and dest:
                result = world.book_shipment(origin, dest, cargo_type, carrier or "standard")
                return result["message"]
            
            return "Specify: book from [origin] to [destination]"
        
        # CHECK PORT
        if "check port" in action or "port status" in action:
            for port in ["rotterdam", "antwerp", "hamburg", "boston"]:
                if port in action:
                    result = world.check_port(port, cargo_type)
                    return result["message"]
            return "Specify port: rotterdam, hamburg, antwerp, boston"
        
        # CHECK CARRIER
        if "check carrier" in action or "carrier status" in action:
            for carrier in ["northstar", "maersk", "msc"]:
                if carrier in action:
                    result = world.check_carrier(carrier)
                    return result["message"]
            return "Specify carrier: northstar, maersk, msc"
        
        # CHECK COMPLIANCE
        if "compliance" in action:
            dest = "boston" if "boston" in action else "new_york" if "new_york" in action else "us"
            result = world.check_compliance(cargo_type, dest)
            return result["message"]
        
        # TRY ALTERNATIVE - agent must explicitly try alternatives
        if any(word in action for word in ["try", "use", "switch", "alternative", "instead"]):
            for alt in ["hamburg", "antwerp", "maersk", "msc"]:
                if alt in action:
                    for dest in ["boston", "new_york", "shanghai"]:
                        if dest in action:
                            result = world.book_shipment(alt, dest, cargo_type, "standard")
                            return result["message"]
                    return f"Trying {alt.title()}. Specify destination."
            return "Specify what alternative to try."
        
        # FALLBACK - unknown action
        return f"Unknown action: {action[:50]}. Use: book, check port, check carrier, check compliance"
    
    return tool_executor


# =============================================================================
# EXTENDED TEST SCENARIOS (15 LEARNING + 5 TEST)
# =============================================================================

def get_extended_learning_scenarios() -> List[tuple]:
    """
    Get EXTENDED learning scenarios (15 tasks) for cumulative learning test.
    
    COMPLEX BLACK SWAN MATRIX (on Tuesday the 15th):
    - Rotterdam: ALWAYS blocked (R-482)
    - Hamburg → US: BLOCKED on 15th (H-903)  
    - Antwerp → US: WORKS (for standard cargo)
    - Hamburg → Non-US: WORKS
    
    Simple guessing strategy (try Hamburg) will FAIL for US destinations!
    Agent must learn: Rotterdam→fail, Hamburg→US→fail, must use Antwerp for US
    
    Tasks 1-5: Discovery phase - encounter failures
    Tasks 6-10: Learning phase - PRECEPT discovers patterns
    Tasks 11-15: Mastery - PRECEPT knows the complex routing
    
    RAG+Tools with simple "try Hamburg" will fail US destinations!
    """
    return [
        # Phase A: Discovery (tasks 1-5) - Rotterdam fails, Hamburg→US also fails!
        ("Book shipment from Rotterdam to Boston", "Rotterdam blocked + Hamburg→US blocked"),
        ("Book cargo from Rotterdam to New York", "Rotterdam blocked + Hamburg→US blocked"),
        ("Book from Rotterdam to Shanghai", "Rotterdam blocked, Hamburg→Shanghai should work"),
        ("Ship goods from Rotterdam to Boston", "Rotterdam + Hamburg→US both blocked"),
        ("Book from Rotterdam to New York", "Must learn: Antwerp for US"),
        
        # Phase B: Learning (tasks 6-10) - PRECEPT learns complex routing
        ("Transport cargo from Rotterdam to Boston", "PRECEPT should try Antwerp for US"),
        ("Arrange shipment from Rotterdam to Shanghai", "Hamburg works for non-US"),
        ("Book freight from Rotterdam to New York", "Antwerp is the only option for US"),
        ("Ship container from Rotterdam to Boston", "Learning: Antwerp for US routes"),
        ("Book delivery from Rotterdam to Shanghai", "Hamburg for non-US routes"),
        
        # Phase C: Mastery (tasks 11-15) - PRECEPT knows the matrix
        ("Book express from Rotterdam to Boston", "Should use Antwerp directly"),
        ("Rush shipment from Rotterdam to New York", "Should use Antwerp directly"),
        ("Priority cargo from Rotterdam to Shanghai", "Should use Hamburg directly"),
        ("Urgent delivery from Rotterdam to Boston", "Antwerp is automatic now"),
        ("Fast shipping from Rotterdam to Shanghai", "Hamburg is automatic for non-US"),
    ]


def get_extended_test_scenarios() -> List[tuple]:
    """
    Get EXTENDED test scenarios (5 tasks) to verify learning persistence.
    
    MIX of US (need Antwerp) and non-US (Hamburg works):
    - RAG+Tools: Will fail US destinations (tries Hamburg which fails)
    - PRECEPT: Should know Antwerp for US, Hamburg for non-US
    """
    return [
        ("Book shipment from Rotterdam to Boston", "US dest: PRECEPT→Antwerp, RAG→Hamburg→FAIL"),
        ("Ship cargo from Rotterdam to Shanghai", "Non-US: Both should use Hamburg"),
        ("Send goods from Rotterdam to New York", "US dest: PRECEPT→Antwerp, RAG→Hamburg→FAIL"),
        ("Book delivery from Rotterdam to Boston", "US dest: PRECEPT→Antwerp, RAG→Hamburg→FAIL"),
        ("Transport package from Rotterdam to Shanghai", "Non-US: Both should use Hamburg"),
    ]


# =============================================================================
# TEST SCENARIOS (RIGOROUS - NO HINTS)
# =============================================================================

def get_learning_scenarios() -> List[tuple]:
    """
    Get RIGOROUS scenarios designed to trigger learning.
    
    All scenarios hit black swans with NO hints or alternatives mentioned.
    RAG+Tools will fail repeatedly. PRECEPT should learn after failures.
    
    Returns:
        List of (task, black_swan_description) tuples
    """
    return [
        # ALL Rotterdam scenarios - will hit Tuesday black swan
        # NO hints about alternatives!
        ("Book shipment from Rotterdam to Boston", "Rotterdam closed Tuesday - no hint"),
        ("Book cargo from Rotterdam to New York", "Rotterdam closed Tuesday - repeated"),
        ("Ship goods from Rotterdam to Boston", "Rotterdam closed Tuesday - 3rd attempt"),
        ("Book from Rotterdam to Shanghai", "Rotterdam closed Tuesday - 4th attempt"),
        ("Send package from Rotterdam to Boston", "Rotterdam closed Tuesday - 5th attempt"),
    ]


def get_post_learning_scenarios() -> List[tuple]:
    """
    Get POST-LEARNING scenarios - SAME black swan, no hints.
    
    Tests if learning actually happened:
    - RAG+Tools: Will fail (no memory)
    - PRECEPT: Should succeed (learned to use Hamburg)
    
    Returns:
        List of (task, expected_behavior) tuples
    """
    return [
        # SAME Rotterdam scenarios - PRECEPT should know to use Hamburg
        ("Book shipment from Rotterdam to Boston", "PRECEPT should use Hamburg automatically"),
        ("Book cargo from Rotterdam to New York", "PRECEPT should use Hamburg automatically"),
        ("Ship goods from Rotterdam to Shanghai", "PRECEPT should use Hamburg automatically"),
    ]


def get_hard_black_swan_scenarios() -> List[tuple]:
    """
    Get EXTRA HARD scenarios with multiple black swans.
    
    Returns:
        List of (task, black_swans) tuples
    """
    return [
        # Hits Rotterdam + QDNA-7 black swans
        ("Book pharma from Rotterdam to Boston", "Rotterdam closed + QDNA-7 required"),
        # Hits NorthStar blackout
        ("Book via NorthStar from Rotterdam to Boston", "Rotterdam + NorthStar blocked"),
        # Multiple constraints
        ("Book hazmat from Rotterdam to New York via NorthStar", "Rotterdam + NorthStar + hazmat"),
    ]

