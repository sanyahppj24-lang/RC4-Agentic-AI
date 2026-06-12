import random


class RLAgent:

    def __init__(self):

        # Q-values for actions
        self.q_values = {
            "increase_samples": 0.0,
            "validate_best": 0.0,
            "stop_and_report": 0.0
        }

        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9

    # =====================================================
    # CHOOSE BEST ACTION
    # =====================================================
    def choose_action(self):

        # Select action with highest Q-value
        best_action = max(
            self.q_values,
            key=self.q_values.get
        )

        return best_action

    # =====================================================
    # REWARD FUNCTION
    # =====================================================
    def calculate_reward(
        self,
        best_bias,
        confidence,
        validated
    ):

        reward = 0

        # Strong bias found
        if best_bias > 2.0:
            reward += 5

        elif best_bias > 1.5:
            reward += 2

        else:
            reward -= 2

        # Confidence reward
        if confidence > 2.0:
            reward += 4

        elif confidence > 1.2:
            reward += 2

        else:
            reward -= 1

        # Validation reward
        if validated:
            reward += 5

        return reward

    # =====================================================
    # UPDATE Q VALUES
    # =====================================================
    def update_q_values(
        self,
        action,
        reward
    ):

        current_q = self.q_values[action]

        new_q = current_q + self.learning_rate * (
            reward
            + self.discount_factor * max(self.q_values.values())
            - current_q
        )

        self.q_values[action] = new_q

    # =====================================================
    # MAIN RL DECISION FUNCTION
    # =====================================================
    def decide(
        self,
        best_bias,
        confidence,
        validated
    ):

        # Rule-based action selection
        if best_bias < 1.5:

            action = "increase_samples"

        elif confidence < 2.0:

            action = "validate_best"

        else:

            action = "stop_and_report"

        # Calculate reward
        reward = self.calculate_reward(
            best_bias,
            confidence,
            validated
        )

        # Update Q-table
        self.update_q_values(
            action,
            reward
        )

        print("\n[RL AGENT]")
        print("Best bias:", round(best_bias, 4))
        print("Confidence:", round(confidence, 4))
        print("Validated:", validated)
        print("Selected action:", action)
        print("Reward:", reward)

        print("\nUpdated Q-values:")

        for action_name, q_value in self.q_values.items():

            print(
                action_name,
                ":",
                round(q_value, 4)
            )

        return {
            "action": action,
            "reward": reward,
            "q_values": self.q_values
        }