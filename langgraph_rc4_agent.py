from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END

from dataset_tool import generate_dataset
from position_discovery_tool import analyze_positions
from memory_agent import MemoryAgent
from rl_agent import RLAgent
from report_agent import generate_report
from validator_node import validate_bias


# =========================================================
# STATE
# =========================================================

class RC4AgentState(TypedDict):
    sample_size: int
    attempt: int
    max_attempts: int

    dataset_path: Optional[str]

    best_result: Optional[Dict[str, Any]]

    action: Optional[str]

    memory: List[Dict[str, Any]]

    confidence: float

    validated: bool

    final_report: Optional[str]


memory_agent = MemoryAgent()
rl_agent = RLAgent()


# =========================================================
# DATA GENERATION NODE
# =========================================================

def generate_data_node(state: RC4AgentState):

    dataset_path = generate_dataset(
        num_samples=state["sample_size"],
        keystream_length=10
    )

    print("\n[DATA GENERATION NODE]")
    print("Generated dataset with:",
          state["sample_size"],
          "samples")

    return {
        **state,
        "dataset_path": dataset_path
    }


# =========================================================
# ANALYSIS NODE
# =========================================================

def analyze_node(state: RC4AgentState):

    results_df = analyze_positions(
        dataset_path=state["dataset_path"],
        max_position=10
    )

    best_result = results_df.iloc[0].to_dict()

    print("\n[POSITION DISCOVERY TOOL]")
    print("Best position:",
          int(best_result["position"]))

    print("Most common byte:",
          int(best_result["most_common_byte"]))

    print("Observed probability:",
          round(best_result["observed_probability"], 6))

    print("Expected probability:",
          round(best_result["expected_probability"], 6))

    print("Bias ratio:",
          round(best_result["bias_ratio"], 4))

    return {
        **state,
        "best_result": best_result
    }


# =========================================================
# MEMORY NODE
# =========================================================

def memory_node(state: RC4AgentState):

    best = state["best_result"]

    experiment = {
        "attempt": state["attempt"],

        "sample_size": state["sample_size"],

        "position": int(best["position"]),

        "most_common_byte":
            int(best["most_common_byte"]),

        "bias_ratio":
            float(best["bias_ratio"]),

        "observed_probability":
            float(best["observed_probability"])
    }

    memory_agent.remember(experiment)

    print("\n[MEMORY AGENT]")
    print("Stored experiment:")
    print(experiment)

    return {
        **state,
        "memory": memory_agent.get_history()
    }


# =========================================================
# RL DECISION NODE
# =========================================================

def rl_decision_node(state: RC4AgentState):

    best_bias = float(
        state["best_result"]["bias_ratio"]
    )

    confidence = state.get("confidence", 1.0)

    validated = state.get("validated", False)

    decision = rl_agent.decide(
        best_bias=best_bias,
        confidence=confidence,
        validated=validated
    )

    action = decision["action"]

    print("\n[RL DECISION AGENT]")
    print("Chosen action:", action)

    print("Reward:",
          round(decision["reward"], 4))

    print("Q-values:")
    print(decision["q_values"])

    return {
        **state,
        "action": action
    }


# =========================================================
# ROUTER NODE
# =========================================================

def router_node(state: RC4AgentState):

    action = state["action"]

    print("\n[ROUTER NODE]")
    print("Attempt:",
          state["attempt"])

    print("Action:",
          action)

    if state["attempt"] >= state["max_attempts"]:

        print("Max attempts reached")
        return "report"

    if action == "increase_samples":
        return "increase_samples"

    elif action == "validate_best":
        return "validate_best"

    elif action == "stop_and_report":
        return "report"

    return "report"


# =========================================================
# INCREASE SAMPLE NODE
# =========================================================

def increase_samples_node(state: RC4AgentState):

    new_sample_size = state["sample_size"] * 2

    print("\n[INCREASE SAMPLE NODE]")
    print("Increasing samples to:",
          new_sample_size)

    return {
        **state,

        "sample_size":
            new_sample_size,

        "attempt":
            state["attempt"] + 1
    }


# =========================================================
# VALIDATION NODE
# =========================================================

def validate_best_node(state: RC4AgentState):

    validation = validate_bias(
        state["best_result"]
    )

    confidence = validation["confidence"]

    current_samples = state["sample_size"]

    # adaptive scaling
    if confidence < 1.2:

        new_sample_size = current_samples * 4

    elif confidence < 2.0:

        new_sample_size = current_samples * 2

    else:

        new_sample_size = int(current_samples * 1.5)

    print("\n[VALIDATION NODE]")

    print("Confidence:",
          round(confidence, 4))

    print("Validated:",
          validation["validated"])

    print("New sample size:",
          new_sample_size)

    return {
        **state,

        "sample_size":
            new_sample_size,

        "attempt":
            state["attempt"] + 1,

        "confidence":
            confidence,

        "validated":
            validation["validated"]
    }


# =========================================================
# REPORT NODE
# =========================================================

def report_node(state: RC4AgentState):

    report = generate_report(
        best_result=state["best_result"],
        memory=state["memory"]
    )

    print("\n[FINAL REPORT]")
    print(report)

    return {
        **state,
        "final_report": report
    }


# =========================================================
# BUILD GRAPH
# =========================================================

graph = StateGraph(RC4AgentState)

graph.add_node(
    "generate_data",
    generate_data_node
)

graph.add_node(
    "analyze",
    analyze_node
)

graph.add_node(
    "memory",
    memory_node
)

graph.add_node(
    "rl_decision",
    rl_decision_node
)

graph.add_node(
    "increase_samples",
    increase_samples_node
)

graph.add_node(
    "validate_best",
    validate_best_node
)

graph.add_node(
    "report",
    report_node
)


# =========================================================
# ENTRY POINT
# =========================================================

graph.set_entry_point("generate_data")


# =========================================================
# EDGES
# =========================================================

graph.add_edge(
    "generate_data",
    "analyze"
)

graph.add_edge(
    "analyze",
    "memory"
)

graph.add_edge(
    "memory",
    "rl_decision"
)


# =========================================================
# CONDITIONAL ROUTING
# =========================================================

graph.add_conditional_edges(
    "rl_decision",

    router_node,

    {
        "increase_samples":
            "increase_samples",

        "validate_best":
            "validate_best",

        "report":
            "report"
    }
)


graph.add_edge(
    "increase_samples",
    "generate_data"
)

graph.add_edge(
    "validate_best",
    "generate_data"
)

graph.add_edge(
    "report",
    END
)


# =========================================================
# COMPILE GRAPH
# =========================================================

app = graph.compile()


# =========================================================
# INITIAL STATE
# =========================================================

initial_state = {

    "sample_size": 50000,

    "attempt": 1,

    "max_attempts": 4,

    "dataset_path": None,

    "best_result": None,

    "action": None,

    "memory": [],

    "confidence": 1.0,

    "validated": False,

    "final_report": None
}


# =========================================================
# RUN GRAPH
# =========================================================

result = app.invoke(initial_state)

print("\n[LANGGRAPH FINISHED]")
print(result["final_report"])