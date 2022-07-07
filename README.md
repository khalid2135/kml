#Implement Simple HCS

# (NEIGHBOUR, HEURISTIC)

graph = {
    "A": [('B', 6), ('C', 5), ('D', 4)],
    "B": [('A', 7), ('E', 2)],
    "C": [('A', 7), ('E', 2), ('F', 3)],
    "D": [('A', 7), ('F', 3)],
    "E": [('B', 6), ('C', 5), ('G', 1)],
    "F": [('C', 5), ('D', 4), ('G', 1)],
    "G": [('E', 2), ('F', 3)]
}

START_NODE = ('A', 7) # Start node
LAST_NODE = -2
HEURISTIC = -1

def hill_climbing(start, my_graph):
    explored = list()
    current_node = start
    
    while True:
        explored.append(current_node[LAST_NODE])
        
        current_heuristic = current_node[HEURISTIC]
        next_node = None
        
        for neighbour in my_graph[current_node[LAST_NODE]]:
            if neighbour[LAST_NODE] in explored:
                continue
            if current_heuristic > neighbour[HEURISTIC]:
                next_node = neighbour
                current_heuristic = neighbour[HEURISTIC]
        if next_node is not None:
            current_node = next_node
        else:
            break
    
    return explored

# Main
result = hill_climbing(START_NODE, graph)
print("Path:", result)
