import heapq

class Node:
    def __init__(self, c, f, l=None, r=None):
        self.c, self.f, self.l, self.r = c, f, l, r

    def __lt__(self, o):
        return self.f < o.f

def calculate_frequencies(t):
    return {c: t.count(c) for c in set(t)}

def build_huffman_tree(t):
    q, f = [], calculate_frequencies(t)
    [heapq.heappush(q, Node(ch, fr)) for ch, fr in f.items()]

    while len(q) > 1:
        l, r = heapq.heappop(q), heapq.heappop(q)
        heapq.heappush(q, Node(None, l.f + r.f, l, r))

    return heapq.heappop(q)

def get_encoding(n, c, e):
    if n:
        if n.l or n.r:
            get_encoding(n.l, c + "0", e)
            get_encoding(n.r, c + "1", e)
        else:
            e[n.c] = c

def huffman_encoding(t):
    r, e = build_huffman_tree(t), {}
    get_encoding(r, "", e)
    return e

def test_huffman_encoding(t):
    print(f"Text: {t}")
    print(f"Huffman Encoding: {huffman_encoding(t)}")
    print("\n")

# Test the program with sample input
text = input("Enter the text: ")
test_huffman_encoding(text)
