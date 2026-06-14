import random


class RLAgent:

    def __init__(self):

        # =====================================================
        # Q TABLE
        # =====================================================

        self.q_values = {
            "increase_samples": 0.0,
            "validate_best": 0.0,
            "stop_and_report": 0.0
        }

        # =====================================================
        # RL PARAMETERS
        # =====================================================

        self.learning_rate = 0.15
        self.discount_factor = 0.9

        # exploration chance
        self.epsilon = 0.2

        # reward history
        self.reward_history = []

    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def choose_action(self):

        actions = list(self.q_values.keys())

        # Exploration
        if random.random() < self.epsilon:

            action = random.choice(actions)

            print("\n[RL AGENT]")
            print("Exploration selected")

            return action

        # Exploitation
        best_action = max(
            self.q_values,
            key=self.q_values.get
        )

        print("\n[RL AGENT]")
        print("Best Q-value action selected")

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

        reward = 0.0

        # =====================================================
        # BIAS REWARD
        # =====================================================

        if best_bias >= 2.0:

            reward += 5

        elif best_bias >= 1.7:

            reward += 3

        elif best_bias >= 1.5:

            reward += 1

        else:

            reward -= 3

        # =====================================================
        # CONFIDENCE REWARD
        # =====================================================

        if confidence >= 3.0:

            reward += 5

        elif confidence >= 2.0:

            reward += 3

        elif confidence >= 1.2:

            reward += 1

        else:

            reward -= 2

        # =====================================================
        # VALIDATION REWARD
        # =====================================================

        if validated:

            reward += 6

        else:

            reward -= 1

        return reward

    # =====================================================
    # Q VALUE UPDATE
    # =====================================================

    def update_q_values(
        self,
        action,
        reward
    ):

        current_q = self.q_values[action]

        max_future_q = max(
            self.q_values.values()
        )

        new_q = current_q + self.learning_rate * (
            reward
            + self.discount_factor * max_future_q
            - current_q
        )

        # Stabilization clamp
        new_q = max(min(new_q, 100), -100)

        self.q_values[action] = round(new_q, 4)

    # =====================================================
    # MAIN RL LOGIC
    # =====================================================

    def decide(
        self,
        best_bias,
        confidence,
        validated
    ):

        # =====================================================
        # INITIAL RULE STABILIZATION
        # =====================================================

        if best_bias < 1.5:

            suggested_action = "increase_samples"

        elif confidence < 2.0:

            suggested_action = "validate_best"

        else:

            suggested_action = "stop_and_report"

        # =====================================================
        # RL ACTION
        # =====================================================

        rl_action = self.choose_action()

        # Hybrid stable strategy
        if self.q_values[rl_action] < 0:

            action = suggested_action

        else:

            action = rl_action

        # =====================================================
        # REWARD
        # =====================================================

        reward = self.calculate_reward(
            best_bias,
            confidence,
            validated
        )

        self.reward_history.append(reward)

        # =====================================================
        # UPDATE Q TABLE
        # =====================================================

        self.update_q_values(
            action,
            reward
        )

        # =====================================================
        # DEBUG PRINTS
        # =====================================================

        print("\n================ RL AGENT ================")

        print("Best Bias:",
              round(best_bias, 4))

        print("Confidence:",
              round(confidence, 4))

        print("Validated:",
              validated)

        print("Chosen Action:",
              action)

        print("Reward:",
              round(reward, 4))

        print("\nQ VALUES:")

        for name, value in self.q_values.items():

            print(
                name,
                "->",
                round(value, 4)
            )

        print("==========================================")

        # =====================================================
        # RETURN
        # =====================================================

        return {
            "action": action,
            "reward": reward,
            "q_values": self.q_values
        }