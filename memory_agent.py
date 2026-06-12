class MemoryAgent:

    def __init__(self):
        self.history = []

    def remember(self, experiment):
        self.history.append(experiment)

    def get_history(self):
        return self.history

    def get_best_result(self):
        if not self.history:
            return None

        return max(
            self.history,
            key=lambda x: x["bias_strength"]
        )