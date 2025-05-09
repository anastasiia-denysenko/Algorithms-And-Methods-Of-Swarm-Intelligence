class Greedy:
    def __init__(self, Backpack):
        self.Backpack = Backpack
        self.solution = self.solve()
        self.weight_of_best, self.best_value = self.Backpack.find_sum(self.solution)

    def solve(self):
        val_to_weight = {}
        for i in range(self.Backpack.items):
            val_to_weight[self.Backpack.values[i]/self.Backpack.weights[i]] = i
        val_to_weight = dict(sorted(val_to_weight.items())).values()
        res = []
        for i in val_to_weight:
            if self.Backpack.find_sum(res+[i]) != -1:
                res.append(i)
            else:
                return res
        