def knapsack(values, weights, capacity):
    n = len(values)
    dp = [0] * (capacity + 1)

    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    max_value, selected_items = dp[capacity], [i for i in range(n-1, -1, -1) if weights[i] <= capacity and (capacity == 0 or dp[capacity] == dp[capacity - weights[i]] + values[i])]

    return max_value, selected_items

# Example usage:
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
max_value, selected_items = knapsack(values, weights, capacity)

print("Maximum value:", max_value)
print("Selected items:", selected_items)
