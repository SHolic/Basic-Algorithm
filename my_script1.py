# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:35:31 2017

@author: 罗骏
"""
#import math


class Node(object):
    def __init__(self, value=-1, lchild=None,rchild=None):
        self.value = value
        self.lchild = lchild
        self.rchild = rchild
class Tree(object):
    def __init__(self):
        self.root = None
        self.queue = []
    def add_node(self, value):
        node = Node(value)
        if self.root == None:
            self.root = node
            self.queue += [self.root]
        elif self.queue[0].lchild == None:
            self.queue[0].lchild = node
            self.queue += [node]
        elif self.queue[0].rchild == None:
            self.queue[0].rchild = node
            self.queue += [node]
            self.queue.pop(0)
    def front_digui(self, node):
        if node == None:
            return
        else:
            print(node.value)
            self.front_digui(node.lchild)
            self.front_digui(node.rchild)
    def front_stack(self, root):
        if root == None:
            return False
        stack = []
        node = root
        while node or stack:
            while node:
                stack += [node]
                print(node.value)
                node = node.lchild
            node = stack.pop()
            node = node.rchild
    def middle_stack(self, root):
        if root == None:
            return False
        node = root
        stack = []
        while node or stack:
            while node:
                stack += [node]
                node = node.lchild
            node = stack.pop()
            print(node.value)
            node = node.rchild            
    def rear_stack(self, root):
        if root == None:
            return False
        stack1 = [root]
        stack2 = []
        while stack1:
            node = stack1.pop()
            if node.lchild:
                stack1 += [node.lchild]
            if node.rchild:
                stack1 += [node.rchild]
            stack2 += [node]
        while stack2:
            print(stack2.pop().value)
    
    def level_stack(self):
        myQueue = [self.root]
        while myQueue:
            node = myQueue.pop(0)
            if node.lchild:
                myQueue += [node.lchild]
            if node.rchild:
                myQueue += [node.rchild]
            print(node.value)

    def nostack(self, kind):
        #不使用栈前中后序遍历二叉树
        if kind == "post":
            dummy = Node(-1, self.root, None)
            cur = dummy
        else:
            cur = self.root
        pre = None
        while cur:
            if cur.lchild == None:
                if kind == "pre" or kind == "in":
                    print(cur.value)
                cur = cur.rchild
            else:
                pre = cur.lchild 
                while pre.rchild and pre.rchild != cur:
                    pre = pre.rchild
                if pre.rchild == None:
                    pre.rchild = cur
                    if kind == "pre":
                        print(cur.value)
                    cur = cur.lchild
                else:
                    pre.rchild = None
                    if kind == "in":
                        print(cur.value)
                    elif kind == "post":
                        self.reverse(cur.lchild, pre)
                    cur = cur.rchild
    def reverse(self, begin, end):
        if begin == end:
            print(begin.value)
            return
        self.reverse(begin.rchild, end)
        print(begin.value)
    
    def node_count(self, node):
        if node == None:
            return 0
        else:
            return self.node_count(node.lchild)+self.node_count(node.rchild)+1
    def leaves_count(self, node):
        if node == None:
            return 0
        if node.lchild == None and node.rchild == None:
            return 1
        else:
            return self.leaves_count(node.lchild)+self.leaves_count(node.rchild)
    def depth_get(self, node):
        if node == None:
            return 0
        else:
            return max(self.depth_get(node.lchild),self.depth_get(node.rchild))+1
    def klevel_get(self, node, k):
        if node == None:
            return
        elif k == 1:
            print(node.value)
            return 1
        else:
            return (self.klevel_get(node.lchild, k-1) + self.klevel_get(node.rchild, k-1))
    def find_LCA(self, node, target1, target2):
        #最低公共祖先
        if node == None:
            return None
        if node.value == target1 or node.value == target2:
            return node.value
        left = self.find_LCA(node.lchild, target1, target2)
        right = self.find_LCA(node.rchild, target1, target2)
        if left and right:
            return node
        return left if left else right
        
    def distance_nodes(self, node, target1, target2):
        #求两结点距离
        parents = self.find_LCA(self.root, target1, target2)
        node1 = self.find_level(parents, target1)
        node2 = self.find_level(parents, target2)
        print(node1 + node2)
    def find_level(self, node, target):
        #从LCA向下找节点
        if node == None:
            return -1
        if node.value == target:
            return 0
        level = self.find_level(node.lchild, target)
        if level == -1:
            level = self.find_level(node.rchild, target)
        if level != -1:
            return level + 1
        return -1
        
    def find_all_ancestors(self, node, target):
        #查找一个节点的所有祖先节点
        if node == None:
            return False
        if node.value == target:
            return True
        if (self.find_all_ancestors(node.lchild, target) \
            or self.find_all_ancestors(node.rchild, target)):
            print(node.value)
            return True
        else:
            return False
    
    pre_order_arry = [1, 2, 4, 7, 3, 5, 8, 9, 6]
    in_order_arry = [4, 7, 2, 1, 8, 5, 9, 3, 6]
    def print_post_order(self, pos1, pos2, n):
        #二叉树前序中序推后序
        if n == 1:
            print(self.pre_order_arry[pos1])            #
            return 
        elif n == 0:
            return
        else:
            i = 0
            while self.pre_order_arry[pos1] != self.in_order_arry[pos2+i]:
                i += 1
            self.print_post_order(pos1+1, pos2, i)
            self.print_post_order(pos1+1+i, pos2+1+i, n-i-1)    #
            print(self.pre_order_arry[pos1])            #
    def is_CBT(self):
        #检查是否是完全二叉树
        myqueue = [self.root]
        flag = False
        while myqueue:
            node = myqueue.pop(0)
            if flag and (node.lchild or node.rchild):
                return False
            else:
                if node.lchild and node.rchild:
                    myqueue.append(node.lchild)
                    myqueue.append(node.rchild)
                elif node.rchild :
                    return False
                elif node.lchild:
                    flag = True
                    myqueue.append(node.lchild)
                else:
                    flag = True              #
        return True
    def is_BST(self, node, min_number, max_number):
        #检查是否是二叉查找树
        if node == None:
            return True
        if node.value >= max_number or node.value <= min_number:
            return False
        return(self.is_BST(node.lchild, min_number, node.value) \
               and self.is_BST(node.rchild, node.value, max_number))
        
    def is_post_BST(self, array, begin, end):
        #判断是否是二叉查找树的后序遍历
        if array[end] <= array[begin]:
            return True
        i = begin
        while array[i] < array[end]:
            i += 1
        for j in range(i,end):
            if array[j] < array[end]:
                return False
        return self.is_post_BST(array, begin, i-1) and self.is_post_BST(array, i, end-1)

    def sortedlistobst(self, link_node):
        if link_node == None:
            return 
        elif link_node.next_node == None:
            return Tree_Node(link_node.value)
        slow = link_node
        pre_slow = link_node
        fast = link_node
        while fast.next_node and fast:
            pre_slow = slow
            slow = slow.next_node
            fast = fast.next_node.next_node
        mid = Tree_node(slow.value)
        if pre_slow:
            pre_slow.next_node = None
            mid.lchild = self.sortedlistobst(link_node)
        mid.rchild = self.sortedlistobst(slow.next_node)
        return mid
       
def select_sort(lists):
    # 选择排序
    count = len(lists)
    for i in range(0, count):
        min = i
        for j in range(i + 1, count):
            if lists[min] > lists[j]:
                min = j
        lists[min], lists[i] = lists[i], lists[min]
    return lists
def sift(array, left, end):
    """交换成大顶堆"""
    i = left
    j = 2 * i + 1
    key = array[i]
    while j <= end:                     #
        if j < end and array[j] < array[j+1]:
            j = j + 1
        if array[j] > key:
            array[i] = array[j]
            i = j
            j = 2 * i + 1
        else: break            
    array[i] = key
def heap_sort(lists):
    """堆排序"""
    if len(lists) <= 0:
        return False
    count = len(lists)
    local_list = []
    local_list += lists
    for i in range(count//2-1, -1, -1):     #
        sift(local_list, i, count-1)
    for i in range(count-1, 0, -1):
        local_list[0], local_list[i] = local_list[i], local_list[0]
        sift(local_list, 0, i-1) 
    print(local_list)
    
def insert_sort(lists):
    # 插入排序
    count = len(lists)
    for i in range(1, count):
        key = lists[i]
        j = i - 1
        while j >= 0:
            if lists[j] > key:
                lists[j + 1] = lists[j]
                lists[j] = key
            j -= 1
    return lists
    
def shell_sort(lists):
    """希尔排序"""
    if len(lists) <= 0:
        return False
    local_list = []
    local_list += lists
    count = len(local_list)
    group = len(local_list) // 2
    while group > 0:
        for i in range(group, count):
            key = local_list[i]
            j = i - group
            while j >= 0 and local_list[j] > key:               #
                local_list[j + group] = local_list[j]
                j -= group
            local_list[j + group] = key
        group //= 2
    print(local_list)

def bubble_sort(lists):
    # 冒泡排序
    count = len(lists)
    for i in range(count):
        for j in range(i + 1, count):
            if lists[i] > lists[j]:
                lists[i], lists[j] = lists[j], lists[i]
    return lists
def partition(lists, left, right):
    """分堆"""
    key = lists[left]
    while left < right:
        while left < right and lists[right] >= key:
            right -= 1
        lists[left] = lists[right]
        while left < right and lists[left] <= key:
            left += 1
        lists[right] = lists[left]
    lists[left] = key
    return left
def quick_sort(lists, left, right):
    """快排"""
    if right > left:
        mid = partition(lists, left, right)
        quick_sort(lists, left, mid-1)
        quick_sort(lists, mid+1, right) 
def top_k_quick_sort(array, k, left, right):
    if left == right:
        return array[right]
    mid = partition(array, left, right)
    if mid - left + 1>k:
        return top_k_quick_sort(array, k, left, mid-1)
    elif mid - left + 1 == k:
        return array[mid]
    else:
        return top_k_quick_sort(array, k - mid + left - 1, mid+1, right) #
    
def merge(array, low, mid, high):
    """单次归并"""
    tmp = []
    i = low
    j = mid + 1
    while i <= mid and j <= high:
        if array[i] <= array[j]:
            tmp += [array[i]]
            i += 1
        elif array[j] < array[i]:
            tmp += [array[j]]
            j += 1
    if i <= mid:
        tmp += array[i:mid+1]
    elif j <= high:
        tmp += array[j:high+1]
    array[low:high+1] = tmp    
def merge_sort(array, low, high):
    """归并排序"""
    if low < high:
        mid = (low+high)//2
        merge_sort(array, low, mid)
        merge_sort(array, mid+1, high)
        merge(array, low, mid, high)

tree = Tree()
for i in range(20):
    tree.add_node(i)
print("front_digui")
tree.front_digui(tree.root)
print("front_stack")
tree.front_stack(tree.root)
print("middle_stack")
tree.middle_stack(tree.root)
print("rear_stack")
tree.rear_stack(tree.root)
print("level_stack")
tree.level_stack()
print("node_count")
print(tree.node_count(tree.root))
print("count_leaves")
print(tree.leaves_count(tree.root))
print("depth_get")
print(tree.depth_get(tree.root))
print("klevel_get")
print(tree.klevel_get(tree.root, 3))
print("find_LCA")
print(tree.find_LCA(tree.root, 15, 18).value)
print("distance_nodes")
tree.distance_nodes(tree.root, 15, 18)
print("find_all_ancestors")
tree.find_all_ancestors(tree.root, 15)
print("pre and in find postorder")
tree.print_post_order(0, 0, 9)
print("is BST")
print(tree.is_BST(tree.root, 0, 20))
print("is CBT")
print(tree.is_CBT())

a = [2,3,46,21,76,8,6,9,14,23,39,51,0,17,20,28,34,52,52,14,3]
array1 = "ABCDABD"
array2 = "BBCABCDABABCDABCDABDE"
print("heap_sort")
heap_sort(a)
print("shell_sort")
shell_sort(a)
print("quick_sort")
quick_sort(a, 0, len(a)-1)
print(a)
a = [2,3,46,21,76,8,6,9,14,23,39,51,0,17,20,28,34,52,52,14,3]
print("merge_sort")
merge_sort(a, 0, len(a)-1)
print(a)

class Link_Node(object):
    def __init__(self, value = -1, next_node = None):
        self.value = value
        self.next_node = next_node
class Link_List(object):
    def __init__(self):
        self.root = Link_Node()
        self.temp = Link_Node()
    def add_node(self, value):
        #加node
        node = Link_Node(value)
        if self.root.value == -1:
            self.root = node
            self.temp = self.root
        else:
            self.temp.next_node = node
            self.temp = self.temp.next_node
    def bianli(self, node):
        #就是遍历
        while node:
            print(node.value)
            node = node.next_node
            
    def mid_node(self, node):
        #找到中间的node并返回
        fast = node
        slow = node
        while fast.next_node and fast.next_node.next_node:
            fast = fast.next_node.next_node
            slow = slow.next_node
        print(slow.value)
        return slow
    
    def del_node(self, root, node):
        #删除任意一个node，这个用上面返回的mid
        if node.next_node:
            node.next_node.value, node.value = node.value, node.next_node.value
            node.next_node = node.next_node.next_node
        else:
            while node.next_node.next_node:
                node = node.next_node
            node.next_node = None
        print(node.value, node.next_node.value)
        return
        
    def con_loop(self, node):
        #建立一个单环链表
        loop_node = self.root
        while loop_node.next_node:
            loop_node = loop_node.next_node
        loop_node.next_node = node
    def dect_loop(self, root):
        #检测是否带环并返回环的节点
        fast = root.next_node.next_node
        slow = root.next_node
        while fast != slow and fast != None:
            fast = fast.next_node.next_node
            slow = slow.next_node
        print(fast.value) if fast == slow else print("no loop")
        if fast == slow:
            start = root
            cont = slow
            while start != cont:
                start = start.next_node
                cont = cont.next_node
            print(start.value)
        
    def kth_last(self, k):
        #返回倒数第k个值
        tail_node = self.root
        while k>=1:
            tail_node = tail_node.next_node
            k -= 1
        front_node = self.root
        while tail_node.next_node:
            tail_node = tail_node.next_node
            front_node = front_node.next_node
        print(front_node.value)
        
    def desc_add(self, value):
        #反向加入节点
        node = Link_Node(value)
        tmp = self.root
        self.root = node
        if tmp.value == -1:
            node.next_node = None
        else:
            node.next_node = tmp

    def reverse(self):
        #将已有的链表翻转
        node = self.root
        if node == None or node.next_node == None:
            return
        tmp_node1 = node.next_node
        tmp_node2 = node.next_node.next_node
        node.next_node = None
        while tmp_node2:
            tmp_node1.next_node = node
            node = tmp_node1
            tmp_node1 = tmp_node2
            tmp_node2 = tmp_node2.next_node
        tmp_node1.next_node = node
        self.root = tmp_node1
    
    def merge_link(self, root1, root2):
        #两个链表合成一个，去除重复项
        self.merge_root = None
        point1 = root1
        point2 = root2
        while point1 != None and point2 != None:
            if self.merge_root == None:
                if root1.value <= root2.value:
                    self.merge_root = root1
                    point1 = root1.next_node
                elif root1.value > root2.value:
                    self.merge_root = root2
                    point2 = root2.next_node
                cur = self.merge_root
            if point1.value <= point2.value and point1.value != cur.value:
                cur, cur.next_node, point1 = cur.next_node, point1, point1.next_node
            elif point1.value > point2.value and point2.value != cur.value:
                cur, cur.next_node, point2 = cur.next_node, point2, point2.next_node
            elif point1 != None and point1.value == cur.value:
                point1 = point1.next_node
            elif point2 != None and point2.value == cur.value:
                point2 = point2.next_node
        while point1 != None:
            cur, cur.next_node, point1 = cur.next_node, point1, point1.next_node
        while point2 != None:
            cur, cur.next_node, point2 = cur.next_node, point2, point2.next_node 
        return self.merge_root
    
    def return_end(self):
        node = self.root
        while node.next_node:
            node = node.next_node
        return node
    def asc_sort(self, start, end):
        #按从大到小重排链表
        if start == end or start.next_node == end:
            return
        i = start
        j = start.next_node
        while j != end:
            if j.value >= start.value:
                i = i.next_node
                i.value, j.value = j.value, i.value
            j = j.next_node
        i.value, start.value = start.value, i.value    #
        self.asc_sort(start, i)
        self.asc_sort(i.next_node, end)
                
link_list = Link_List()
for i in range(20):
    link_list.add_node(i)
link_list.bianli(link_list.root)
#for i in range(20):
#    link_list.desc_add(i)
#link_list.bianli(link_list.root)
print("reverse")
link_list.reverse()
link_list.bianli(link_list.root)
link_list.reverse()
print("mid_node")
node_to_del = link_list.mid_node(link_list.root)
#link_list.del_node(link_list.root, node_to_del)
#link_list.con_loop(node_to_del)
#print("dect_loop")
#link_list.dect_loop(link_list.root)
print("kth_last")
link_list.kth_last(3)
end = link_list.return_end()
print("sort")
link_list.asc_sort(link_list.root, None)
link_list.bianli(link_list.root)
print('merge_two_link')
link_list2 = Link_List()
for i in range(20,40):
    link_list2.add_node(0.5*i)
merge_root = link_list1.merge_link(link_list1.root, link_list2.root)
link_list1.bianli(merge_root)


class RB_Node(object):
    def __init__(self, value = -1, parents = None, lchild = None, rchild = None, color = "red"):
        self.value = value
        self.parents = parents
        self.lchild = lchild
        self.rchild = rchild
        self.color = color
        
class RB_Tree(object):
    def __init__(self):
        node = RB_Node()
        self.root = node
    def lrotate(self, x):
        w = x.rchild
        w.parents = x.parents
        if x.parents == None:
            self.root = w
        elif x == x.parents.lchild:
            x.parents.lchild = w
        else:
            x.parents.rchild = w
        x.parents = w
        x.rchild = w.lchild
        if w.lchild:
            w.lchild.parents = x
        w.lchild = x
    def rrotate(self, x):
        w = x.lchild
        w.parents = x.parents
        if x.parents == None:
            self.root = w
        elif x == x.parents.lchild:
            x.parents.lchild = w
        else :
            x.parents.rchild = w
        x.parents = w
        if w.rchild:
            w.rchild.parents = x
        x.rchild = w.rchild
        w.rchild = x
    def insert_reblance(self, x):
        while x.parents.color == "red" :
            if x.parents == x.parents.parents.lchild:
                y = x.parents.parents.rchild
                if y.color == "red":
                    x.parents.color = "black"
                    y.color = "black"
                    x.parents.parents.color = "red"
                    x = x.parents.parents
                elif x == x.parent.rchild:
                    x = x.parents
                    self.lrotate(x)
                x.parents.color = "black"
                x.parents.parents.color = "red"
                self.rrotate(x.parents.parents)
                break
        self.root.color = "black"
    def eraser_reblance(self, x):
        if x != self.root and x.color == "black":
            #只处理x在左子树的情况
            if x == x.parents.lchild:
                w = x.parents.rchild
                if w.color == "red":                #case1
                    w.color = "black"
                    x.parents.color = "red"
                    self.lrotate(x.parents)
                if (w.lchild.color == "black" or w.lchild == None) \
                and (w.rchild == None or w.rchild.color == "black"):#case2
                    w.color = "red"
                    x = x.parents
                    x.parents = x.parents.parents
                else:                               #case3
                    if w.rchild.color == "black" or w.rchild == None:
                        w.color = "red"
                        w.parents.color = "black"
                        self.rrotate(w)
                        w = w.parents
                    w.color = x.parents.color      #case4
                    if w.rchild:
                        w.rchild.color = "black"
                    x.parents.color = "black"
                    self.lrotate(x.parents)
        x.color = "black"