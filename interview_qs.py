from collections import Counter
import random
# 1: ARRAYS AND STRINGS

#1.1
def isUnique(s):
    return len(set(s)) == len(s)

#1.2
def isPermute(s1,s2):
    return Counter(s1) == Counter(s2)

#1.3
def URLify(s):
    return s.replace(' ','%20')

#1.4
def palPermute(s):
    s = s.replace(' ','')
    c = Counter(s)
    if len(s) % 2 == 0:
        for i in c:
            if c[i] % 2 == 1:
                return False
    if len(s) % 2 == 1:
        count = 0
        for i in c:
            if c[i] % 2 == 1:
                count += 1
            if count > 1:
                return False
        if count > 1:
            return False
    return True

#1.5
def oneAway(s1,s2):
    count = 0
    if len(s1) > len(s2):
        for i in set(s1):
            if i not in set(s2):
                count += 1
            if count > 1:
                return False
        if count > 1:
            return False
    if len(s2) >= len(s1):
        for i in set(s2):
            if i not in set(s1):
                count += 1
            if count > 1:
                return False
        if count > 1:
            return False
    return True


#1.6
def strComp(s):
    count = 0
    new_str = []
    for i in range(len(s)):
        if s[i] == s[i-1] and i > 0:
            count += 1
            if i == len(s)-1:
                count += 1
                new_str.append(s[i])
                new_str.append(str(count))
                break
        if s[i] != s[i-1] and i > 0:
            count += 1
            new_str.append(s[i-1])
            new_str.append(str(count))
            count = 0
        if i == len(s)-1 and s[i] != s[i-1]:
            new_str.append(s[i])
            new_str.append(str(1))
            break
    if len(new_str) >= len(s):
        return s
    return ''.join(new_str)

#1.7 ***REVIEW***
def rotate90(mat):
    n = len(mat)
    for i in range(int(n/2)):
        bot = n - 1 - i
        for j in range(i,bot):
            offset = j - i
            top = mat[i][j] #top layer
            mat[i][j] = mat[bot-offset][i] #left to top
            mat[bot-offset][i] = mat[bot][bot-offset] #bottom to left
            mat[bot][bot-offset] = mat[j][bot] #right to bottom
            mat[j][bot] = top #top to right
    return mat

#1.8
# O(MN) Time but O(1) Space
def rowcol0(mat):
    c = len(mat[0])
    r = len(mat)
    is_col = False
    for i in range(r):
        if mat[i][0] == 0:
            is_col = True
        for j in range(1,c):
            if mat[i][j] == 0:
                mat[i][0] = 0
                mat[0][j] = 0

    for i in range(1,r):
        for j in range(1,c):
            if not mat[i][0] or not mat[0][j]:
                mat[i][j] = 0

    if mat[0][0] == 0:
        for j in range(c):
            mat[0][j] = 0

    if is_col:
        for i in range(r):
            mat[i][0] = 0

    return mat

#1.9
#first need to make a function to check if a string is a substring of another
def isSubstring(s1,s2):
    return s2 in s1
#Now the actual answer
def stringRot(s1,s2):
    s1 = s1 + s1
    return isSubstring(s1,s2)

# 2: NODES AND LINKED LISTS

#2.1
# Definition for singly-linked list.
class Node:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next

class linkedList:
    def __init__(self):
        self.head = None

    def printList(self):
        curr = self.head
        while curr:
            print(curr.val)
            curr = curr.next

def delDup(head: Node) -> Node:
    if head == None:
        return head
    prev = head
    current = head.next
    while current != None:
        if current.val == prev.val:
            prev.next = current.next
            current = current.next
        else:
            current = current.next
            prev = prev.next
    return head

#2.2
def ktoLast(k, head: Node) -> Node:
    fast = head
    slow = head
    for i in range(k):
        if fast == None:
            return None
        fast = fast.next

    while fast:
        fast = fast.next
        slow = slow.next

    return slow

#2.3
def delMidNode(node: Node) -> Node:
    if node.next != None:
        node.val = node.next.val
        node.next = node.next.next

#2.4
def partitionList(node: Node, x: int) -> Node:
    if node == None:
        return None

    head = node
    head_start = None
    tail_start = None
    tail = node

    while node:
        next_n = node.next
        if node.val < x and head_start == None:
            head_start = node
            head = node
            head.next = node
        if node.val < x and head_start != None:
            head.next = node
            head = node
        elif node.val >= x and tail_start == None:
            tail_start = node
            tail = node
            tail.next = node
        elif node.val >= x and tail_start != None:
            tail.next = node
            tail = node
        node = next_n
    tail.next = None
    head.next = tail_start

    return head_start

#2.5
def sumLists(rev1: Node, rev2: Node, carry: int) -> Node:
    if rev1 == None and rev2 == None and carry == 0:
        return None
    newNode = Node(-1)
    value = carry
    if rev1 != None:
        value += rev1.val
    if rev2 != None:
        value += rev2.val

    newNode.val = value % 10

    if value >= 10:
        carry = 1
    else:
        carry = 0
    if rev1 != None or rev2 != None:
        if rev1 == None:
            newNode.next = sumLists(rev1,rev2.next,carry)
        elif rev2 == None:
            newNode.next = sumLists(rev1.next, rev2, carry)
        else:
            newNode.next = sumLists(rev1.next, rev2.next, carry)

    return newNode

#2.6
def tail_and_length(head: Node) -> int:
    length = 0
    tail = None
    current = head
    while current:
        length += 1
        if current.next == None:
            tail = current
        current = current.next
    return tail, length

def isPalindrome(head: Node) -> bool:
    isPal = []
    current = head
    while current:
        isPal.append(current.val)
        current = current.next

    if isPal[::-1] == isPal:
        return True
    else:
        return False

#2.7
def intersect(head1: Node, head2: Node):
    curr1 = head1
    curr2 = head2

    tail1, length1 = tail_and_length(head1)
    tail2, length2 = tail_and_length(head2)

    diff = abs(length2 - length1)

    if tail1 != tail2:
        return False, None

    if length1 > length2:
        for i in range(diff):
            curr1 = curr1.next

    if length2 > length1:
        for i in range(diff):
            curr2 = curr2.next

    while curr1 != curr2:
        curr1 = curr1.next
        curr2 = curr2.next

    return True, curr1

#2.8
def loopNode(head: Node) -> Node:
    if head == None:
        return None

    fast = head
    slow = head

    while fast != None and fast.next != None:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break

    if fast == None or fast.next == None:
        return None

    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow

# 3: STACKS AND QUEUES

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        if self.items:
            return False
        else:
            return True

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        if self.items:
            return False
        else:
            return True

    def add(self, item):
        return self.items.insert(0,item)

    def remove(self):
        if len(self.items) > 0:
            return self.items.pop(0)

    def peek(self):
        return self.items[0]

#3.1
def split3(arr):
    length = len(arr)
    stack1 = Stack()
    stack2 = Stack()
    stack3 = Stack()
    for i in range(length):
        if i < length/3:
            stack1.push(arr[i])
        if i >= length/3 and i < 2*length/3:
            stack2.push(arr[i])
        if i >= 2*length/3:
            stack3.push(arr[i])
    return stack1,stack2,stack3

#3.2
def minStack(stack):
    return min(stack.items)

#3.3
class SetOfStacks:
    def __init__(self, limit):

        self.stack = Stack()
        self.stacks = [self.stack]

        self.limit = limit

        self.last_stack = self.stacks[len(self.stacks)-1]


    def push(self, item):
        length = len(self.last_stack.items)
        if self.last_stack != None and length != self.limit:
            self.last_stack.push(item)
        else:
            new_stack = Stack()
            new_stack.push(item)
            self.stacks.append(new_stack)
            self.last_stack = self.stacks[len(self.stacks)-1]
    def pop(self):
        length = len(self.last_stack.items)
        if length == None or length == 0:
            self.stacks.pop()
        else:
            self.last_stack.items.pop()
            length = len(self.last_stack.items)
            if length == None or length == 0:
                self.stacks.pop()

#3.4
class MyQueue:

    def __init__(self, stack):

        self.stack = stack
        self.queue = Stack()

    def push(self,item):
        self.stack.push(item)
        for i in range(len(self.stack.items)):
            self.queue.push(self.stack.pop())

    def shift(self):
        for i in range(len(self.stack.items)):
            self.queue.push(self.stack.peek())
            self.stack.pop()
        return self.queue

    def peek(self):
        self.queue = self.shift()
        return self.queue.peek()

    def remove(self):
        self.queue = self.shift()
        self.queue.pop()

#3.5
def sortStack(stack: Stack) -> Stack:
    stack.items = sorted(stack.items, reverse = True)
    return stack

#3.6
class AnimalShelter:
    def __init__(self, pref, dogs, cats):
        self.pref = pref
        self.dogs = dogs
        self.dogs.items = sorted(self.dogs.items)
        self.cats = cats
        self.cats.items = sorted(self.cats.items)

    def add(self, age):
        animal = self.pref
        if animal == 'dog':
            self.dogs.push(age)
            self.dogs.items = sorted(self.dogs.items)
        if animal == 'cat':
            self.cats.push(age)
            self.cats.items = sorted(self.cats.items)

    def remove(self):
        if self.pref == None:
            if max(self.dogs.items) >= max(self.cats.items):
                self.dogs.pop()
            if max(self.dogs.items) < max(self.cats.items):
                self.cats.pop()
        elif self.pref == 'cat':
            self.cats.pop()
        else:
            self.dogs.pop()


# 4: TREES AND GRAPHS

class TreeNode:
    def __init__(self, root, left = None, right = None):
        self.root = root
        self.left = left
        self.right = right



class Graph:
    def __init__(self, graph = None):
        if graph == None:
            self.graph = {}
        else:
            self.graph = graph

    def add_vertex(self,vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, edge):
        v1, v2 = edge
        if v1 in self.graph:
            self.graph[v1].append(v2)
        else:
            self.graph[v1] = [v2]

#4.1
def route_between(graph, start, end):
    if start == end:
        return True

    visited = [False]*(len(graph.graph))

    queue = Queue()
    queue.add(start)
    visited[start] = True

    while queue:
        start = queue.remove()
        if start == None:
            return False
        for i in graph.graph[start]:
            if i == end:
                return True
            if visited[i] == False:
                queue.add(i)
                visited[i] = True

    return False

#4.2
def minTree(arr):
    if arr == []:
        return None

    mid = int(len(arr)/2)

    rootNode = TreeNode(arr[mid])
    rootNode.left = minTree(arr[:mid])
    rootNode.right = minTree(arr[mid+1:])

    return rootNode

#4.3
def depthList(root: TreeNode, arr, level):
    if root == None:
        return

    if len(arr) == level:
        newList = Node(root)
        arr.append(newList)
    else:
        head = arr[level]
        while head != None:
            if head.next == None:
                head.next = Node(root)
                break
            head = head.next

    depthList(root.left,arr,level+1)
    depthList(root.right,arr,level+1)

#4.4
def treeHeight(root):
    if not root:
        return 0
    return max(treeHeight(root.left), treeHeight(root.right)) + 1

def isBalanced(root: TreeNode) -> bool:
    if not root:
        return True

    height = abs(treeHeight(root.left) - treeHeight(root.right))

    if height > 1:
        return False
    else:
        return True

#4.5 ***
def isSearch(tree):
    if not tree:
        return True
    if not tree.left and not tree.right:
        return True
    elif tree.right and not tree.left:
        if tree.root >= tree.right.root:
            return False
    elif tree.left and not tree.right:
        if tree.left.root > tree.root:
            return False
    else:
        if tree.root >= tree.right.root:
            return False
        if tree.left.root > tree.root:
            return False
    if tree.left and tree.right:
        return isSearch(tree.left) and isSearch(tree.right)
    elif tree.left and not tree.right:
        return isSearch(tree.left)
    elif tree.right and not tree.left:
        return isSearch(tree.right)

#4.10
def isSubTree(t1, t2):
    if t1 == None:
        return False
    elif t1.root == t2.root and matchTree(t1,t2):
        return True

    return isSubTree(t1.left,t2) or isSubTree(t1.right, t2)

def matchTree(t1,t2):
    if t1 == None and t2 == None:
        return True
    elif t1 == None or t2 == None:
        return False
    elif t1.root != t2.root:
        return False
    else:
        return matchTree(t1.left,t2.left) and matchTree(t1.right,t2.right)

#4.11
class newTreeNode:
    def __init__(self, root = None, left = None, right = None, size = None):
        self.root = root
        if size:
            self.size = size
        else:
            self.size = 1
        self.left = left
        self.right = right

def getRandomNode(tree):
        if not tree.left:
            return tree
        leftSize = tree.left.size
        index = random.randint(tree.left.size-1,tree.left.size+1)
        if index < leftSize:
            return getRandomNode(tree.left)
        elif index == leftSize:
            return tree
        else:
            return getRandomNode(tree.right)

def countPathsSum(tree, prev_sum, tot_path, target):

    if not tree or prev_sum > target:
        return 0

    prev_sum += tree.root

    if prev_sum == target:
        tot_path += 1

    tot_path += countPathsSum(tree.left, prev_sum, tot_path, target)
    tot_path += countPathsSum(tree.right, prev_sum, tot_path, target)

    return tot_path

# Chapter 5: Bit Manipulation

def getBit(num, i):
    return (num & (1 << i) != 0)

def setBit(num, i, value):
    mask = ~(1 << i)
    return (num & mask) | (value << i)

#5.1
# N and M are 32 bit numbers
# i and j are bit positions i < j
def insertion(N, M, i, j):
    ones = ~0b0
    left = ones << j+1
    right = ((0b1 << i) - 0b1)
    mask = left | right
    n_cleared = N & mask
    m_shifted = M << i
    return n_cleared | m_shifted

#5.2
def floatToBit(fl):
    frac = 0.5
    bin_num = ['0','.']
    while fl > 0:
        if len(bin_num) > 12:
            break
        if fl >= frac:
            bin_num.append('1')
            fl = fl - frac
        else:
            bin_num.append('0')
        frac = frac/2

    bin_num = ''.join(bin_num)
    return bin_num

#5.3
def longestSeqOnes(num: int) -> int:
    if num+1 & num == 0 and num != 0:
        return len(bin(num)) - 2
    curr_len = 0
    prev_len = 0
    max_len = 1
    while num > 0:
        if (num & 1) == 1:
            curr_len += 1
        elif (num & 1) == 0:
            prev_len = 0 if ((num & 2) == 0) else curr_len
            curr_len = 0
        max_len = max(prev_len + curr_len + 1, max_len)
        num = num >> 1

    return max_len

#5.4
def getNextNum(num: int):
    temp = num
    c0 = 0
    c1 = 0
    while (temp & 1) == 0:
        c0 += 1
        temp >>= 1

    while (temp & 1) == 1:
        c1 += 1
        temp >>= 1

    return num + (1 << c0) + (1 << (c1 - 1)) - 1

def getPrevNum(num: int):
    temp = num
    c0 = 0
    c1 = 0
    while (temp & 1) == 0:
        c0 += 1
        temp >>= 1

    while (temp & 1) == 1:
        c1 += 1
        temp >>= 1

    return num - (1 << c1) - (1 << (c1-1)) + 1

#5.6
#Assumptions: A and B have the same length
def flipsAtoB(A, B):

    if A & B == 0 and A != 0 and B != 0:
        return len(bin(A)) - 2

    C = A ^ B
    count = 0

    while C != 0:
        count += 1
        C = C & (C-1)

    return count

#5.7
def swapOddEven(x):
    return ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1)

