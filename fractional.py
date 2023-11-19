def fractional_knapsack(weights, values, capacity):
    ratios = sorted([(v / w, w, v) for v, w in zip(values, weights)], reverse=True)
    total_value, knapsack = 0, [0] * len(weights)

    for i, (ratio, weight, value) in enumerate(ratios):
        fraction = min(capacity / weight, 1)
        knapsack[i], capacity, total_value = fraction, capacity - fraction * weight, total_value + fraction * value

    return total_value, knapsack

# Example usage:
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50

max_value, fractions = fractional_knapsack(weights, values, capacity)

print("Maximum value in Knapsack =", max_value)
print("Fractions of items in the Knapsack =", fractions)
