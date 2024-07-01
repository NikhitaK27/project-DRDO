import heapq

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic: Estimated cost from current node to goal
        self.f = 0  # Total cost: g + h

    def __eq__(self, other):
        return self.position == other.position

def astar(maze, start, end):
    # Initialize start and end nodes
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize lists for open and closed nodes
    open_list = []
    closed_list = []

    # Heapify the open list and add the start node
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # Create a set for faster lookup
    visited = set()

    # Define movement directions (up, down, left, right, diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Loop until open list is empty
    while len(open_list) > 0:
        # Pop the node with the smallest f value from heap
        current_node = heapq.heappop(open_list)

        # Add current node to closed list
        closed_list.append(current_node)

        # Check if goal is reached
        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        # Generate children nodes
        children = []
        for direction in directions:
            # Get node position
            node_position = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

            # Check if within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            # Check if obstacle
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append child node to children list
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in open list and f cost is lower
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add child to open list
            heapq.heappush(open_list, child)

    # No path found
    return None

