from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END

import time

from dataset_tool import generate_dataset
from position_discovery_tool import analyze_positions
from memory_agent import MemoryAgent
from rl_agent import RLAgent
from report_agent import generate_report
from validator_node import validate_bias
from experiment_logger import (initialize_csv,log_experiment)


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


# =========================================================
# GLOBAL AGENTS
# =========================================================

memory_agent = MemoryAgent()

rl_agent = RLAgent()


# =========================================================
# DATA GENERATION NODE
# =========================================================

def generate_data_node(state: RC4AgentState):

    print("\n=================================================")
    print("[DATA GENERATION NODE]")
    print("=================================================")

    sample_size = state.get("sample_size", 50000)

    start = time.time()

    dataset_path = generate_dataset(
        num_samples=sample_size,
        keystream_length=10
    )

    end = time.time()

    print("Generated dataset")
    print("Samples:", sample_size)
    print("Time taken:", round(end - start, 2), "seconds")

    return {
        **state,

        "dataset_path":
            dataset_path
    }


# =========================================================
# ANALYSIS NODE
# =========================================================

def analyze_node(state: RC4AgentState):

    print("\n=================================================")
    print("[ANALYSIS NODE]")
    print("=================================================")

    start = time.time()

    results_df = analyze_positions(
        dataset_path=state["dataset_path"],
        max_position=10
    )

    end = time.time()

    best_result = results_df.iloc[0].to_dict()

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

    print("Analysis time:",
          round(end - start, 2), "seconds")

    return {

        **state,

        "best_result":
            best_result
    }


# =========================================================
# MEMORY NODE
# =========================================================

def memory_node(state: RC4AgentState):

    print("\n=================================================")
    print("[MEMORY NODE]")
    print("=================================================")

    best = state["best_result"]

    experiment = {

        "attempt":
            state.get("attempt", 1),

        "sample_size":
            state.get("sample_size", 50000),

        "position":
            int(best["position"]),

        "most_common_byte":
            int(best["most_common_byte"]),

        "bias_ratio":
            float(best["bias_ratio"]),

        "observed_probability":
            float(best["observed_probability"])
    }

    memory_agent.remember(experiment)
    log_experiment({

    "attempt":
        state["attempt"],

    "sample_size":
        state["sample_size"],

    "position":
        experiment["position"],

    "most_common_byte":
        experiment["most_common_byte"],

    "bias_ratio":
        experiment["bias_ratio"],

    "confidence":
        state.get("confidence", 1.0),

    "validated":
        state.get("validated", False),

    "action":
        state.get("action", "none")
})

    print("Stored experiment:")
    print(experiment)

    return {

        **state,

        "memory":
            memory_agent.get_history()
    }


# =========================================================
# RL DECISION NODE
# =========================================================

def rl_decision_node(state: RC4AgentState):

    print("\n=================================================")
    print("[RL DECISION NODE]")
    print("=================================================")

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

    print("Chosen action:", action)

    print("Reward:",
          round(decision["reward"], 4))

    print("Q-values:")
    print(decision["q_values"])

    return {

        **state,

        "action":
            action
    }


# =========================================================
# ROUTER NODE
# =========================================================

def router_node(state: RC4AgentState):

    print("\n=================================================")
    print("[ROUTER NODE]")
    print("=================================================")

    attempt = state.get("attempt", 1)

    action = state.get("action", "report")

    print("Current attempt:", attempt)

    print("Selected action:", action)

    # =====================================================
    # MAX ATTEMPT STOP
    # =====================================================

    if attempt >= state.get("max_attempts", 4):

        print("Max attempts reached")
        return "report"

    # =====================================================
    # CONVERGENCE CHECK
    # =====================================================

    memory = state.get("memory", [])

    if len(memory) >= 3:

        recent_biases = [
            exp["bias_ratio"]
            for exp in memory[-3:]
        ]

        avg_bias = (
            sum(recent_biases)
            / len(recent_biases)
        )

        bias_range = (
            max(recent_biases)
            - min(recent_biases)
        )

        print("\n[CONVERGENCE CHECK]")

        print("Recent biases:",
              recent_biases)

        print("Average bias:",
              round(avg_bias, 4))

        print("Bias range:",
              round(bias_range, 4))

        if bias_range < 0.3:

            print("Bias converged")
            return "report"

    # =====================================================
    # NORMAL ROUTING
    # =====================================================

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

    print("\n=================================================")
    print("[INCREASE SAMPLE NODE]")
    print("=================================================")

    current_samples = state.get(
        "sample_size",
        50000
    )

    new_sample_size = current_samples * 2

    MAX_SAMPLES = 1000000

    if new_sample_size > MAX_SAMPLES:

        new_sample_size = MAX_SAMPLES

    print("Increasing samples to:",
          new_sample_size)

    return {

        **state,

        "sample_size":
            new_sample_size,

        "attempt":
            state.get("attempt", 1) + 1
    }


# =========================================================
# VALIDATION NODE
# =========================================================

def validate_best_node(state: RC4AgentState):

    print("\n=================================================")
    print("[VALIDATION NODE]")
    print("=================================================")

    validation = validate_bias(
        state["best_result"]
    )

    confidence = validation["confidence"]

    current_samples = state.get(
        "sample_size",
        50000
    )

    # =====================================================
    # ADAPTIVE SCALING
    # =====================================================

    if confidence < 1.2:

        new_sample_size = current_samples * 4

    elif confidence < 2.0:

        new_sample_size = current_samples * 2

    else:

        new_sample_size = int(
            current_samples * 1.5
        )

    MAX_SAMPLES = 1000000

    if new_sample_size > MAX_SAMPLES:

        new_sample_size = MAX_SAMPLES

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
            state.get("attempt", 1) + 1,

        "confidence":
            confidence,

        "validated":
            validation["validated"]
    }


# =========================================================
# REPORT NODE
# =========================================================

def report_node(state: RC4AgentState):

    print("\n=================================================")
    print("[FINAL REPORT NODE]")
    print("=================================================")

    report = generate_report(
        best_result=state["best_result"],
        memory=state["memory"]
    )

    print(report)

    return {

        **state,

        "final_report":
            report
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

graph.set_entry_point(
    "generate_data"
)


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
initialize_csv()
app = graph.compile()


# =========================================================
# INITIAL STATE
# =========================================================

initial_state = {

    "sample_size":
        50000,

    "attempt":
        1,

    "max_attempts":
        4,

    "dataset_path":
        None,

    "best_result":
        None,

    "action":
        None,

    "memory":
        [],

    "confidence":
        1.0,

    "validated":
        False,

    "final_report":
        None
}


# =========================================================
# RUN GRAPH
# =========================================================

try:

    result = app.invoke(initial_state)

    print("\n=================================================")
    print("[LANGGRAPH FINISHED]")
    print("=================================================")

    print(result["final_report"])

except Exception as e:

    print("\n=================================================")
    print("[ERROR]")
    print("=================================================")

    print(str(e))