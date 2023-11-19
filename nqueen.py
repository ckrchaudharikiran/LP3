def initialize_board(n):
    return [[0] * n for _ in range(n)]

def is_safe(board, row, col, n):
    return all(
        board[i][j] == 0 and 
        all(board[i + k][j + l] == 0 for k, l in ((0, 0), (1, 1), (-1, 1)))
        for i, j in zip(range(row), range(col, -1, -1))
    )

def solve_n_queens(board, col, n):
    if col == n:
        return True

    for i in range(n):
        if is_safe(board, i, col, n):
            board[i][col] = 1
            if solve_n_queens(board, col + 1, n):
                return True
            board[i][col] = 0

    return False

n = int(input("Enter the value of n: "))
chessboard = initialize_board(n)
chessboard[0][0] = 1

if solve_n_queens(chessboard, 1, n):
    for row in chessboard:
        print(row)
else:
    print("No solution exists.")
