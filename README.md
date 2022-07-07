#Implement Simple HCS
"""
A --> MM Alam Road
B --> Johar Town
C --> Shahdara
D --> DHA Phase 6
E --> Wapda Town
F --> Allama Iqbal Town
"""

# (LOCATION, DISTANCE, HEURISTIC)

graph = {
    "A": [('B', 30, 8), ('C', 20, 20)],
    "B": [('A', 30, 25), ('C', 17, 20), ('D', 22, 6)],
    "C": [('A', 20, 25), ('B', 17, 8), ('E', 40, 12)],
    "D": [('B', 22, 8), ('E', 38, 12), ('F', 42, 0)],
    "E": [('C', 40, 20), ('D', 38, 6), ('F', 50, 0)],
    "F": [('D', 42, 6), ('E', 50, 12)]
}

START_NODE = ('A', 0, 25) # Goal node & Start node
LAST_NODE = -3
DISTANCE = -2
HEURISTIC = -1

def hill_climbing(start, my_graph):
    explored = list()
    total_distance = 0
    current_node = start
    
    while True:
        explored.append(current_node[LAST_NODE])
        total_distance += current_node[DISTANCE]
        
        current_heuristic = 99999999
        next_node = None
        goal = None
        
        for neighbour in my_graph[current_node[LAST_NODE]]:
            if neighbour[LAST_NODE] == START_NODE[LAST_NODE]:
                goal = neighbour
                continue
            if neighbour[LAST_NODE] in explored:
                continue
            if current_heuristic > neighbour[HEURISTIC]:
                next_node = neighbour
                current_heuristic = neighbour[HEURISTIC]
        if next_node is not None:
            current_node = next_node
        elif goal is not None:
            explored.append(goal[LAST_NODE])
            total_distance += goal[DISTANCE]
            break
        else:
            break
    
    return explored, total_distance

# Main
result = hill_climbing(START_NODE, graph)
print("Path:", result[0])
print("Total distance:", result[1])











----------Hill Climb 2---------




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










-------------MinMax 1-----------------

#Implement Minimax Algorithm only for the winning value.
# D for Depth
# I for Index
# T for Turn
# S for Score
# T_D for Target_Depth
import math
def MINIMAX_ALGO(D,I,T,S,T_D):
    if (D == T_D):  
        return S[I]
    if (T):
        return max(MINIMAX_ALGO(D + 1,I*2,False,S,T_D),MINIMAX_ALGO(D + 1,I*2+1,False,S,T_D))
    else: 
        return min(MINIMAX_ALGO(D + 1,I*2,True,S,T_D),MINIMAX_ALGO(D + 1,I*2+1,True,S,T_D))
# Driver code 
Utility_Values = [-1,4,2,6,-3,-5,0,7]
Tree_Depth = int(math.log(len(Utility_Values), 2))
print("Node A Value is : ", end = "") 
print(MINIMAX_ALGO(0,0,True,Utility_Values,Tree_Depth)) 




------------MinMax2-----------------

#Implement Minimax Algorithm that print paths to wining/optimal value.
# D for Depth
# I for Index
# T for Turn
# S for Score
# T_D for Target_Depth
import math
def MINIMAX_ALGO(D,I,T,S,T_D):
    if (D == T_D):  
        return S[I]
    if (T):
        return max(MINIMAX_ALGO(D + 1,I*2,False,S,T_D),MINIMAX_ALGO(D + 1,I*2+1,False,S,T_D))
    else: 
        return min(MINIMAX_ALGO(D + 1,I*2,True,S,T_D),MINIMAX_ALGO(D + 1,I*2+1,True,S,T_D))
# Driver code 
Utility_Values = [-1,3,5,1,-6,-4,0,9]
Tree_Depth = int(math.log(len(Utility_Values), 2))
print("The Optimal Value is : ", end = "") 
print(MINIMAX_ALGO(0,0,True,Utility_Values,Tree_Depth)) 











--------------Alpha Beta1-----------------

#Implement Alpha_Beta Pruning that print paths to wining/optimal value.
# D for Depth
# I for Index
# T for Turn
# S for Score
# A for Aplha
# B for Beta 
def Alpha_Beta(D,I,Turn,S,A,B):    
    if D == 3:  
        return S[I]
    if Turn:  
        Best = Beta 
        # Recur for left and right children  
        for i in range(0, 2):  
            V = Alpha_Beta(D+1,I*2+i,False,S,A,B)  
            Best = max(Best, V)  
            A = max(A,Best)  
            # Alpha Beta Pruning  
            if B <= A:  
                break
        return Best  
    else: 
        Best = Alpha 
        # Recur for left and right children  
        for i in range(0, 2):  
            V = Alpha_Beta(D+1,I*2+i,True,S,A,B)  
            Best = min(Best,V)  
            B = min(B,Best)  
            # Alpha Beta Pruning  
            if B <= A:  
                break 
        return Best
# Drive Code
# Initial values of Aplha and Beta  
Alpha, Beta = 1000, -1000 
Scores = [3,5,6,9,1,2,0,-1]
print("The optimal value is :", Alpha_Beta(0,0,True,Scores,Beta,Alpha))






------------Alpha Beta2------------------


#Implement Alpha_Beta Pruning that print paths to wining/optimal value.
# D for Depth
# I for Index
# T for Turn
# S for Score
# A for Aplha
# B for Beta 
def Alpha_Beta(D,I,Turn,S,A,B):    
    if D == 3:  
        return S[I]
    if Turn:  
        Best = Beta 
        # Recur for left and right children  
        for i in range(0, 2):  
            V = Alpha_Beta(D+1,I*2+i,False,S,A,B)  
            Best = max(Best, V)  
            A = max(A,Best)  
            # Alpha Beta Pruning  
            if B <= A:  
                break
        return Best  
    else: 
        Best = Alpha 
        # Recur for left and right children  
        for i in range(0, 2):  
            V = Alpha_Beta(D+1,I*2+i,True,S,A,B)  
            Best = min(Best,V)  
            B = min(B,Best)  
            # Alpha Beta Pruning  
            if B <= A:  
                break 
        return Best
# Drive Code
# Initial values of Aplha and Beta  
Alpha, Beta = 1000, -1000 
Scores = [8,5,6,-4,3,8,4,-6,1,-3,5,2,-3,1,-2,5]
print("The optimal value is :", Alpha_Beta(0,0,True,Scores,Beta,Alpha))







---------K mean----------------


# Machine Learning Algorithm --> Unsupervised --> K means clustering
# A1(2, 10), A2(2, 5), A3(8, 4), A4(5, 8), A5(7, 5), A6(6, 4), A7(1, 2), A8(4, 9)
import numpy as np
import math

def eucledian_distance(point, centeroid):
    return math.ceil(math.sqrt(((centeroid[0] - point[0]) ** 2) + ((centeroid[1] - point[1]) ** 2)))

length_of_dataset = 8
no_of_centeroids = 3

# Storing dataset in 2D array
points = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]], dtype=np.int8)

# Selecting 3 random points from datasets as initial centroids
centeroids_no = np.random.choice(points.shape[0], size=no_of_centeroids, replace=False)
centeroids = np.zeros(shape=(no_of_centeroids, 2), dtype=np.int8)

for i in range(0, no_of_centeroids):
    centeroids[i] = points[centeroids_no[i]]

point_lies = np.zeros(shape=length_of_dataset, dtype=np.int8)

while True:
    # Calculating eucledian distance of points from centroids
    for i in range(0, length_of_dataset):
        mini_dis = np.inf
        for j in range(0, no_of_centeroids):
            dis = eucledian_distance(point=points[i], centeroid=centeroids[j])
            if mini_dis > dis:
                mini_dis = dis
                point_lies[i] = j

    # Updating centeroids
    exit_loop = 0
    for i in range(0, no_of_centeroids):
        sumx = 0
        sumy = 0
        count = 0
        
        for j in range(0, length_of_dataset):
            if point_lies[j] == i:
                sumx += points[j][0]
                sumy += points[j][1]
                count += 1
        
        new_centeroid = (math.ceil(sumx/count), math.ceil(sumy/count))
        
        if new_centeroid[0] == centeroids[i][0] and new_centeroid[1] == centeroids[i][1]:
            exit_loop += 1
        else:
            centeroids[i] = new_centeroid

    if exit_loop == no_of_centeroids:   # If no centeroid is updated
        break

for i in range(0, length_of_dataset):
    print(f"Point {points[i]} lies in cluster {point_lies[i]} with centeroid {centeroids[point_lies[i]]}")
    
    
    
    
    
    
    ------------k mean 2--------------
    
    import random
import math

NUM_CLUSTERS = 3
TOTAL_DATA = 8
LOWEST_SAMPLE_POINT = 6
HIGHEST_SAMPLE_POINT = 3
MIDDLE_SAMPLE_POINT = 5
BIG_NUMBER = math.pow(10, 10)

SAMPLES = [[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]]

data = []
centroids = []

class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def set_x(self, x):
        self.x = x
    
    def get_x(self):
        return self.x
    
    def set_y(self, y):
        self.y = y
    
    def get_y(self):
        return self.y
    
    def set_cluster(self, clusterNumber):
        self.clusterNumber = clusterNumber
    
    def get_cluster(self):
        return self.clusterNumber

class Centroid:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def set_x(self, x):
        self.x = x
    
    def get_x(self):
        return self.x
    
    def set_y(self, y):
        self.y = y
    
    def get_y(self):
        return self.y

def initialize_centroids():
    centroids.append(Centroid(SAMPLES[LOWEST_SAMPLE_POINT][0], SAMPLES[LOWEST_SAMPLE_POINT][1]))
    centroids.append(Centroid(SAMPLES[HIGHEST_SAMPLE_POINT][0], SAMPLES[HIGHEST_SAMPLE_POINT][1]))
    centroids.append(Centroid(SAMPLES[MIDDLE_SAMPLE_POINT][0], SAMPLES[MIDDLE_SAMPLE_POINT][1]))
    
    print("Centroids initialized at:")
    print("(", centroids[0].get_x(), ", ", centroids[0].get_y(), ")")
    print("(", centroids[1].get_x(), ", ", centroids[1].get_y(), ")")
    print("(", centroids[2].get_x(), ", ", centroids[2].get_y(), ")")
    print()
    return

def initialize_datapoints():
    for i in range(TOTAL_DATA):
        newPoint = DataPoint(SAMPLES[i][0], SAMPLES[i][1])
        
        if(i == LOWEST_SAMPLE_POINT):
            newPoint.set_cluster(0)
        elif(i == HIGHEST_SAMPLE_POINT):
            newPoint.set_cluster(1)
        else:
            newPoint.set_cluster(2)
            
        data.append(newPoint)
    
    return

def get_distance(dataPointX, dataPointY, centroidX, centroidY):
    # Calculate Euclidean distance.
    return math.sqrt(math.pow((centroidY - dataPointY), 2) + math.pow((centroidX - dataPointX), 2))

def recalculate_centroids():
    totalX = 0
    totalY = 0
    totalZ = 0
    totalInCluster = 0
    
    for j in range(NUM_CLUSTERS):
        for k in range(len(data)):
            if(data[k].get_cluster() == j):
                totalX += data[k].get_x()
                totalY += data[k].get_y()
                totalInCluster += 1
        
        if(totalInCluster > 0):
            centroids[j].set_x(totalX / totalInCluster)
            centroids[j].set_y(totalY / totalInCluster)
    
    return

def update_clusters():
    isStillMoving = 0
    
    for i in range(TOTAL_DATA):
        bestMinimum = BIG_NUMBER
        currentCluster = 1
        
        for j in range(NUM_CLUSTERS):
            distance = get_distance(data[i].get_x(), data[i].get_y(), centroids[j].get_x(), centroids[j].get_y())
            if(distance < bestMinimum):
                bestMinimum = distance
                currentCluster = j
        
        data[i].set_cluster(currentCluster)
        
        if(data[i].get_cluster() is None or data[i].get_cluster() != currentCluster):
            data[i].set_cluster(currentCluster)
            isStillMoving = 1
    
    return isStillMoving

def perform_kmeans():
    isStillMoving = 1
    
    initialize_centroids()
    
    initialize_datapoints()
    
    while(isStillMoving):
        recalculate_centroids()
        isStillMoving = update_clusters()
    
    return

def print_results():
    for i in range(NUM_CLUSTERS):
        print("Cluster ", i, " includes:")
        for j in range(TOTAL_DATA):
            if(data[j].get_cluster() == i):
                print("(", data[j].get_x(), ", ", data[j].get_y(), ")")
        print()
    
    return

perform_kmeans()
print_results()







----------------Genetic------------

import random
def pick_rand(num):
    
    list2=['0','0','0','0','0']
    list= bin(num).replace("0b", "")
    
    x=4
    w=len(list)-1
    for i in range(len(list)):
        list2[i]=list[i]
        w-=1
        x-=1
    return list2

def fitness_of_num(num):
    w=round((pow(-num,2)/10))
    w=w*-1
    w=w+3*num
    return w

def cross_over(list):
    w=random.randint(1,4)
    x=random.randint(1,4)
    
    if w>x:
        temp=w
        w=x
        x=temp
    for i in range(0,len(list),2):
        for j in range(w,x):
            temp=list[i][j]
            list[i][j]=list[i+1][j]
            list[i+1][j]=temp
        
            
    return list
def convert(list):
    w=str(list)
    x=''
    for i in w:
        if i=='0'or i=='1':
            x+=i
    return x

def binaryToDecimal(binary): 

    binary1 = binary 
    decimal, i, n = 0, 0, 0
    while(binary != 0): 
        dec = binary % 10
        decimal = decimal + dec * pow(2, i) 
        binary = binary//10
        i += 1
    return decimal
def check_fitness(fitness):
    for i in range(len(fitness)):
        if fitness[i]>=23:
            return i
    return -1
def mutate(list):
    w=random.randint(0,4)
    x=random.randint(0,4)
    for i in range(len(list)):
        temp=list[i][w]
        list[i][w]=list[i][x]
        list[i][x]=temp
    return list
def askfitness(number):
    fitness=[]
    for i in range(len(number)):
        fitness.append(fitness_of_num(number[i]))
    return fitness
    
def convertonumbers(list):
    number=[]
    for i in range(len(list)):
        w=convert(list[i])
        number.append(binaryToDecimal(int(w)))

    return number
    
    
def main_function():
    list=[]
    number=[]
    fitness=[]
    
    for x in range(10):
        num=random.randint(0,31)
        number.append(num)
        list.append(pick_rand(num))
        fitness.append(fitness_of_num(num))
    
    w=check_fitness(fitness)   
    if w!=-1:
        print(list[w],"   ",fitness[w],"   ",number[w])
    else:
        num=1
        while True:
            list=cross_over(list)
            number=convertonumbers(list)
            fitness=askfitness(number)
            w=check_fitness(fitness)
            if w!=-1:
                print(list[w], "The fitness is", fitness[w], "and the number is", number[w])
                break
            if num%3==0:
                list=mutate(list)
            num+=1

main_function()
    
    
    







