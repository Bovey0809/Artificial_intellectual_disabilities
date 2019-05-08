# Path search.
graph = {
    1: [2, 3],
    2: [1, 10, 5],
    3: [1, 4, 5, 7],
    4: [3, 9],
    5: [2, 3, 6, 10],
    6: [5, 13],
    7: [3],
    9: [4],
    10: [2, 5, 11],
    11: [10, 12, 13, 14],
    12: [11],
    13: [11, 6],
    14: [11]
}


def bfs_search(start, target, graph):
    paths = [[start, ]]
    while paths:
        print(paths)
        path = paths.pop()
        now = path[-1]
        for next_city in graph[now]:
            if next_city == target:
                return path + [next_city]
            if next_city in path:
                continue
            paths.append(path + [next_city])


print(bfs_search(1, 12, graph))
