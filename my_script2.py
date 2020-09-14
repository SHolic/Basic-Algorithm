# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:35:31 2017
@author: 罗骏
"""
a = [[1,2],[3,3],[0,4],[2,1]]
a.sort(key = lambda a:a[0])
import random, math

def yanghuitringe(n):
    queue = [1]
    print(queue)
    for i in range(1,n+1):
        queue.append(1)
        if i >= 2:
            for j in range(i-1,0,-1):
                queue[j] = queue[j]+queue[j-1]
        print(queue)

def move(n,A,B,C):
    #汉诺塔
    if n==1:
        print(A,'-->',C)
    else:
        move(n-1,A,C,B)#将a上前n-1个盘子从a移动到b上
        move(1,A,B,C)#将a最底下的最后一个盘子从a移动到c上
        move(n-1,B,A,C)#将b上的n-1个盘子移动到c上  
       
def ReverseString(array, left, right):
    "将整个array前后对调"
    while left < right:
        array[left], array[right] = array[right], array[left]
        left += 1
        right -= 1        
def LeftRotateString(array, k):
    "将k个字符移到末尾"
    if k > len(array):
        k %= len(array)
    ReverseString(array, 0, k-1)
    ReverseString(array, k, len(array)-1)
    ReverseString(array, 0, len(array)-1)
    return 

def string_contain(array1, array2):
    "检查array2的所有字符是否在array1中"
    array1 = array1.upper()
    array2 = array2.upper()
    hashtable = 0
    for i in array1:
        hashtable |= 1<<(ord(i)-ord('A'))
    for j in array2:
        if hashtable & 1<<(ord(j)-ord('A')) == 0:
            return False
    return True
    
def partially_matched(array):
    matched = [0 for i in range(len(array))]
    for i in range(1, len(array)+1):
        for j in range(1, i):                
            if array[0:j] == array[i-j:i]:
                matched[i-1] = j                #
    return matched
def KMP(array1, array2): #array1 < array2
    match = partially_matched(array1)
    i = 0
    j = 0
    while 1:
        if j > len(array2):
            return False
        elif array1[i] == array2[j]:
            i += 1
            j += 1
            if i >= len(array1):
                print(j - i + 1)                #
                return True
        else:
            j -= match[i] - 1                   #
            i = 0

def is_palindrome(str1):
    "检查是否是回文数"
    i = 0
    j = len(str1)-1
    while i < j:
        if str1[i] == str1[j]:
            i += 1
            j -= 1
            continue
        else:
            return False
    return True
    
def countSubstrings(array):
    #输入S中有多少回文子串
    return sum((v)//2 for v in manacher(array))
def manacher(array):
    "求一个字符串中的最长回文子串"
    inited_array = '$#' + '#'.join(array) + '#@'
    count = len(inited_array)
    max_len = -1
    farest = 0
    tmp_max_point = -1
    ass_list = [0 for i in range(count)]
    for i in range(count):
        ass_list[i] = min(ass_list[2 * tmp_max_point - i], farest - i) if i < farest else 1
        while (i+ass_list[i] < count-1) and inited_array[i-ass_list[i]] == inited_array[i+ass_list[i]]:
            ass_list[i] += 1
        if farest < i + ass_list[i]:
            tmp_max_point = i
            farest = i + ass_list[i]
        max_len = max(max_len, ass_list[i]-1)
    print(max_len)
    return ass_list

def calcallpermutation(array):
    "全排列 非递归方法"
    my_list = []
    array = array.upper()
    for i in array:
        my_list += [ord(i)-ord('A')+1]
    my_list.sort()
    print(my_list)
    while 1:
        point1 = 0
        for i in range(len(my_list)-2, -1, -1):
            if my_list[i] < my_list[i+1]:
                point1 = i
                break
            elif my_list[i] >= my_list[i+1] and i == 0:
                return
        point2 = point1+1
        for i in range(len(my_list)-1, point1, -1):
            if my_list[i] > my_list[point1] and my_list[i] < my_list[point1+1]:
                point2 = i
                break
        my_list[point1], my_list[point2] = my_list[point2], my_list[point1]
        i, j = point1+1, len(my_list)-1
        while i <= j:
            my_list[i], my_list[j] = my_list[j], my_list[i]
            i += 1
            j -= 1
        print(my_list)
    
def not_equal(array, left, right):
    for i in range(left, right):
        if array[i] == array[right]:                        #
            return False
    return True
def fullpermutation(array, left, right):
    "全排列，递归方法"
    if left == right:
        print(array)
        return
    for i in range(left, right+1):
        if not_equal(array, left, i):
            array[left], array[i] = array[i], array[left]
            fullpermutation(array, left+1, right)           #
            array[left], array[i] = array[i], array[left]
            
total = 0
def not_confilt(col, n):
    for i in range(n):
        if col[i] == col[n] or col[i]-col[n] == i-n or col[i]-col[n] == n-i:
            return False
    return True
def eight_queen(col, n):
    "八皇后问题"
    global total
    if n == len(col):
        total += 1
        print(col)
        return
    else:
        for i in range(len(col)):
            col[n] = i
            if not_confilt(col, n):
                eight_queen(col, n+1)

def fourSum(array, target):
    "找和为定值的N个数"
    array.sort()
    results = []
    findNsum(array, target, 4, [], results)
    return results
def findNsum(array, target, N, result, results):
    if len(array) < N or N < 2: return
    # solve 2-sum
    if N == 2:
        l,r = 0,len(array)-1
        while l < r:
            if array[l] + array[r] == target:
                results.append(result + [array[l], array[r]])
                l += 1
                r -= 1
                while l < r and array[l] == array[l - 1]:
                    l += 1
                while r > l and array[r] == array[r + 1]:
                    r -= 1
            elif array[l] + array[r] < target:
                l += 1
            else:
                r -= 1
    else:
        for i in range(0, len(array)-N+1):           # careful about range
            if target < array[i]*N or target > array[-1]*N:  # take advantages of sorted list
                break
            if i == 0 or (i > 0 and array[i-1] != array[i]):  # recursively reduce N
                findNsum(array[i+1:], target-array[i], N-1, result+[array[i]], results)
    return

def oi_bag(n, contain, value, weight):
    #这里是求最大值，如果是求最小值float('inf')，bag=min(...)+1
    bag = [float("-inf") for i in range(contain+1)] #for full bag
    bag[0] = 0
    #bag = [0 for i in range(contain+1)] #for bag not required full
    for i in range(n):
        #for j in range(weight[i], contain+1, 1): # total bag
        for j in range(contain, weight[i]-1, -1): # 0-1 bag
            bag[j] = max(bag[j-weight[i]]+value[i], bag[j])
    print(bag)    
def two_dimen_bag(n, contain, num_request, value, weight):
    # contain capacity
    # m number restraint
    '有个变体，如果找一组数中是否有其中几个相加正好是完整数组和的一半，就把contain改为sum/2'
    bag = [[float("-inf") for i in range(num_request+1)] for j in range(contain+1)]
    for i in range(num_request+1):
        bag[0][i] = 0
    for i in range(n):
        for j in range(contain, weight[i]-1, -1):
            for k in range(num_request, 0, -1):
                bag[j][k] = max(bag[j][k], bag[j-weight[i]][k-1] + value[i])
    '''
    bag = [False for i in range(sum/2+1)]
    bag[0] = True
    for i in array:#一组数
        for j in (sum/2, i, -1):
            bag[j] = bag[j] if bag[j] else bag[j-i]
    return bag[sum/2]
    '''
    print(bag)            
            
def findMax(array):
    #数组中乘积最大值
    tmpmax = 1
    tmpmin = 1
    out_put = 1
    for i in array:
        tmpmax = max(tmpmax*i, tmpmin*i, i)
        tmpmin = min(tmpmax*i, tmpmin*i, i)
        out_put = max(out_put, tmpmax)
    print(out_put)

def mac(array):
    #连续最大子数组的和
    n = len(array)
    start = array[n-1]
    call = array[n-1]
    for i in range(n-2, -1, -1):
        start = max(start+array[i], array[i])
        call = max(call, start)
    print(call)
    
def max_profit(array):
    "一次买进卖出的股票最大收益"
    cur_max = 0
    output_max = 0
    for i in range(1,len(array)):
        cur_max = max(0, cur_max + array[i] - array[i-1])
        output_max = max(cur_max, output_max)
    print(output_max)
def max_profit2(array):
    "两次买进卖出的股票最大收益"
    if len(array) < 2 :
        return
    buy1, buy2, profit1, profit2 = array[0], -array[0], 0, 0  #
    for i in array:
        buy1 = min(buy1, i)
        profit1 = max(profit1, i-buy1)
        buy2 = max(buy2, profit1-i)
        profit2 = max(profit2, buy2+i)
    print(profit2)
def max_profit_withcool(array):
    "买进卖出要有一天冷却时间的最大收益"
    if len(array) < 2:
        return
    buy1, buy2, profit1, profit2 = -array[0], -array[0], 0, 0  #
    for i in array:
        buy1 = buy2
        buy2 = max(buy2, profit1-i)
        profit1 = profit2        
        profit2 = max(profit2, buy1+i)  #
    print(profit2)

def House_Robber1(array):
    #不能连续偷两家相邻的
    rob = 0
    not_rob = 0
    for value in array:
        tmp = rob
        rob = value + not_rob
        not_rob = max(tmp, not_rob)
    return max(rob, not_rob)
def House_Robber2(array):
    #小区为环形
    return max(House_Robber1(array[:-1]), House_Robber1(array[1:]))

def jumpfloor(n):
    "跳台阶，可以转化成斐波那契数列问题"
    a1 = 1
    a2 = 2
    tmp = 0
    i = 2
    while i != n:
        tmp = a2+a1
        a1 = a2
        a2 = tmp
        i += 1
    print(a2)
    
def Dutchflag(array):
    "荷兰国旗问题（最少的交换使同数字在一起的方法）"
    if len(array) < 3:
        return
    cur = 0
    pre = 0
    end = len(array)-1
    while cur <= end:
        if array[cur] == 0:
            array[cur], array[pre] = array[pre], array[cur]
            cur += 1
            pre += 1
        elif array[cur] == 1:
            cur += 1
        elif array[cur] == 2:
            array[cur], array[end] = array[end], array[cur]
            end -= 1
    print(array)

class my_queue(object):
    #两个栈实现一个堆
    stack1 = []
    stack2 = []
    def add(self,value):
        self.stack1.append(value)
    def out(self):
        if self.stack2:
            return self.stack2.pop()
        else:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()

def longest_ins_sub(a):
    "最长递增子序列"
    tail = [0 for i in range(len(a))]
    #这个表示从一系列长为i的递增子序列中找到递增的最小的最后一个值tail[i]
    longest = 0
    for cur_num in a:
        i, j = 0, longest
        while i != j:
            m = (i + j) // 2
            #于是有if tails[i-1] < x <= tails[i], update tails[i]
            if tail[m] < cur_num:
                i = m + 1
            else:
                j = m
        tail[i] = cur_num
        #if x is larger than all tails, append it, increase the size by 1
        longest = max(i + 1, longest)
    return longest

def LCS(str1, str2):
    "最长公共子串"
    tmp = [0 for i in range(len(str2))]
    longest = 0
    for i in range(len(str1)):
        for j in range(len(str2)-1, -1, -1):
            if str1[i] == str2[j]:
                if j > 0:
                    tmp[j] = tmp[j-1] + 1
                elif j == 0:
                    tmp[j] = 1
            else:
                tmp[j] = 0
            longest = max(longest, tmp[j])
    print(tmp, longest)
  
def unique_paths_with_obstacles(a):
    "二维矩阵中有障碍时独立路径数"
    if a[0][0] or a[-1][-1]:
        return False
    uniquepath = [0 for i in range(len(a[0]))]
    uniquepath[0] = 1
    for i in range(len(a)):
        for j in range(1, len(a[0])):
            if a[i][j]:
                uniquepath[j] = 0 
            else:
                uniquepath[j] = uniquepath[j-1] + uniquepath[j]
    print(uniquepath)
  
def LCSS(str1, str2):
    "最长公共子序列"
    tmp = [[0 for i in range(len(str2)+1)]for j in range(len(str1)+1)]
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            match = tmp[i-1][j-1]+1 if str1[i-1] == str2[j-1] else tmp[i-1][j-1]
            tmp[i][j] = max(tmp[i-1][j], tmp[i][j-1], match)
    print(tmp[-1][-1])
    i = len(str1)
    j = len(str2)
    lcss = []
    while i >= 0 and j >= 0:
        if str1[i-1] == str2[j-1]:
            lcss += str1[i-1]
            i -= 1
            j -= 1
        elif tmp[i][j] == tmp[i-1][j]:
            i -= 1
        elif tmp[i][j] == tmp[i][j-1]:
            j -= 1
    print(lcss)
            
def SED(str1, str2):
    "字符串编辑距离"
    tmp = [[0 for i in range(len(str2)+1)]for j in range(len(str1)+1)]
    for i in range(1, len(str2)+1):
        tmp[0][i] = i
    for j in range(1, len(str1)+1):
        tmp[j][0] = j
    for i in range(1, len(str1)+1):                                         #
        for j in range(1, len(str2)+1):                                     #
            sub = tmp[i-1][j-1] if str1[i-1] == str2[j-1] else tmp[i-1][j-1]+1
            tmp[i][j] = min(tmp[i-1][j]+1, tmp[i][j-1]+1, sub)
    print(tmp[-1][-1])
    
def right_rotate(array, startpoint, midpoint, endpoint):
    ReverseString(array, startpoint, midpoint-1)
    ReverseString(array, midpoint, endpoint-1)
    ReverseString(array, startpoint, endpoint-1)
def cycle_leader(array, cyclestart, begin, mod):
    i = cyclestart + 2 * (begin + 1) - 1
    tmp = array[cyclestart + begin]
    while i != (cyclestart + begin):
        tmp, array[i] = array[i], tmp
        i = (i + 1 - cyclestart) * 2 % mod + cyclestart -1
    array[i] = tmp
def perfect_shuffle(array, cyclestart, n2):
    #完美洗牌算法
    k = math.floor(math.log(n2 + 1, 3))
    if k == 0:
        return
    m2 = 3 ** k - 1
    right_rotate(array, cyclestart + m2//2, cyclestart + n2//2, cyclestart + (m2+n2)//2)
    for i in range(k):
        cycle_leader(array, cyclestart, 3**i-1, m2+1)                       #
    perfect_shuffle(array, cyclestart + m2, n2-m2)
    return array        
def shuffle(array):
    #洗牌的初始化
    array[1:-1] = perfect_shuffle(array[1:-1], 0, len(array[1:-1]))
    return array

def isreasonable(step, x1, x2, n):
    y1 = step - x1
    y2 = step - x2
    return x1 >= 0 and x2 >= 0 and x1 < n and x2 < n\
        and y1 >= 0 and y1 < n and y2 >= 0 and y2 < n
def get_value(pick, step, x1, x2, n):
    return pick[step][x1][x2] if isreasonable(step, x1, x2, n) else float("-inf")
def pick_num2(matrix):
    #二维数组中找两条路径使经过值和最大
    n = len(matrix)
    pick = [[[float("-inf") for i in range(n)] for j in range(n)] for k in range(2*n-1)]  #
    total_step = 2*n-2
    pick[0][0][0] = matrix[0][0]
    for step in range(1, total_step+1):                             #
        for i in range(n):
            for j in range(i, n):
                if not isreasonable(step, i, j, n):
                    continue
                pick[step][i][j] = max(
                    get_value(pick, step-1, i-1, j-1, n),
                    get_value(pick, step-1, i-1, j, n),
                    get_value(pick, step-1, i, j-1, n),
                    get_value(pick, step-1, i, j, n))
                if i != j:
                    pick[step][i][j] += matrix[i][step - i] + matrix[j][step - j]
                else:
                    pick[step][i][j] += matrix[i][step - i]
    print(pick[total_step][-1][-1])
            
def reason(matrix,memory,node):
    if node[0] < 0 or node[0] >= len(matrix) or node[1] < 0 or node[1] >= len(matrix):
        #防止越界
        return False
    elif matrix[node[0]][node[1]] or memory[node[0]][node[1]][2]:
        #检查是否走过回头路，或者这个位置在原始矩阵中是否能走通
        return False
    return True
def test(matrix,memory,outnode):
    node = []
    node += outnode
    valid = []
    for plus in ([1,0],[-2,0],[1,1],[0,-2]):                        #
        #向四个方向各迈一步，检查合理的记下来
        node[0] += plus[0]
        node[1] += plus[1]
        if reason(matrix,memory,node):
            valid += [[node[0], node[1]]]
    return valid
def BFS(matrix):
    '深度优先搜索用得栈,广度优先搜索用得堆'
    start = [0,0]
    n = len(matrix)
    end = [n-1,n-1]
    memory = [[[n+1,n+1,0] for i in range(n)] for j in range(n)]   #
    #memory可以分成好几个数组，这里方便起见把每个二维数组的元素分三个部分
    #头两个值表示这个数组位置是由什么别的位置推理过来的，最后一个表示这个元素
    #已经被/正在运算中，不会重复寻找
    stack = [start]
    memory[start[0]][start[1]][2] = 1
    while stack:
        node = stack.pop(0)
        if node == end:
            break
            #任何时候找到结尾都优先停下来
        valid = test(matrix, memory, node)
        for nextnode in valid:
            #根据valid里提供的合理值，在记忆数组中把关键内容记录下来
            memory[nextnode[0]][nextnode[1]] = [node[0], node[1], 1]
        stack += valid
    prenode = end
    while prenode != start:
        #反向寻找合理路径
        [prenode[0], prenode[1], t] = memory[prenode[0]][prenode[1]]

def Dijkstra(matrix):
    "单源最短路径 也是广度优先算法"
    count = len(matrix)
    cand = [chr(i) for i in range(ord("A"), ord("A")+count)]
    dist = {}
    for name in cand:
        dist[name] = 10000
    dist["A"] = 0
    key_have_done = "A"
    while len(cand) != 1:
        cand = list(filter(lambda x: x!= key_have_done, cand))
        row = ord(key_have_done) - ord("A")
        for i in range(count):
            if matrix[row][i] == -1:                    #
                continue
            elif matrix[row][i] + dist[key_have_done] < dist[chr(ord("A")+i)]:
                dist[chr(ord("A")+i)] = matrix[row][i] + dist[key_have_done]
        min_value = 10000
        min_character = " "
        for name in cand:
            if dist[name] < min_value:
                min_value = dist[name]
                min_character = name
        key_have_done = min_character
    print(dist)
    
def strtofloat(str1):
    "把一个可能带正负号小数点的字符型数字变成float形式"
    high_pos = 0
    low_pos = 0
    flag = 0
    if len(str1) <= 0:
        return
    for i in str1:
        if i == "+" or i == "-":
            continue
        if i == ".":
            flag = -1
        elif flag == 0:
            high_pos = high_pos * 10 + (ord(i) - ord("0"))
        elif flag < 0:
            low_pos = low_pos + (ord(i) - ord("0")) * pow(10,flag)
            flag -= 1
    num = high_pos + low_pos
    if str1[0] == "-":
        num = -num
    print(num)

def count_one(N):
    #目前的数之前出现过多少个1
    high_pos = 0
    low_pos = 0
    count = 0
    figure = 1
    while N:
        high_pos = N//10
        current = N - high_pos * 10
        if current == 0:
            count += high_pos * figure
        elif current == 1:
            count += high_pos * figure + low_pos + 1
        else:
            count += (high_pos+1) * figure
        low_pos += current * figure
        N //= 10
        figure *= 10
    print(count)
        
def zero_one(mod):
    #最小的能被mod整除的全1和0组成的数
    queue = [1]
    while queue:
        node = queue.pop(0)
        if node % mod == 0:
            print(node)
            break
        queue += [node * 10, node * 10 + 1]

def gcd(num1, num2):
    if num1 < num2:
        return gcd(num2, num1)
    if num2 == 0:
        return num1
    else:
        if (not(num1 & 1) and not(num2 & 1)):
            return (gcd(num1 >> 1, num2 >> 1) << 1)
        elif (not(num1 & 1) and (num2 & 1)):
            return gcd(num1 >> 1, num2)
        elif ((num1 & 1) and not(num2 & 1)):
            return gcd(num1, num2 >> 1)
        elif ((num1 & 1) and (num2 & 1)):
            return gcd(num2, num1-num2)

array2 = ["a1","a2","a3","a4","a5","a6",
          "b1","b2","b3","b4","b5","b6"]
print(shuffle(array2))
array1 = ["a","b","c","d","e","f"]
LeftRotateString(array1, 2)
print(array1)
array1 = "ABCDEFGTY"
array2 = "BCDEX"
print(string_contain(array1, array2))
array1 = "ABCDABD"
array2 = "BBCABCDABABCDABCDABDE"
print("KMP")
KMP(array1, array2)
strtofloat("-1236.384")
manacher("abbahopxpo")
array1 = "DCBA"
print("calcallpermutation")
calcallpermutation(array1)
print('fullpermutation')
fullpermutation([1,1,3,4],0,3)
print(fourSum([1, 0, -1, 0, -2, 2],0))
c = 10
v = [6,3,5,4,6]
w = [2,2,6,5,4]
oi_bag(len(v), c, v, w)
m = 3
two_dimen_bag(len(v), c, m, v, w)
max_profit([7, 1, 5, 3, 6, 4])
max_profit2([2,3,46,21,76,8,6,9,14,23,39,51,0,17,20,28,34,52,52,14,3])
max_profit_withcool([2,3,46,21,76,8,6,9,14,23,39,51,0,17,20,28,34,52,52,14,3])
print("jumpfloor")
jumpfloor(5)
Dutchflag([1,2,2,2,1,1,0,0,0,1,2])
findMax([-2.5, 4, 0, 3, 0.5, 8, -1])
col = [0 for i in range(8)]
#eight_queen(col, 0)
#print(total)
ary = [1,9,3,8,11,4,5,6,4,1,9,7,1,7]
print("longgest increasing subsequence")
longest_ins_sub(ary)
#matrix = [[random.randint(0,10) for i in range(8)] for j in range(8)]
#matrix = [[2,0,8,0,2],[0,0,0,0,0],[0,3,2,0,0],[0,0,0,0,0],[2,0,8,0,2]]
#print(matrix)
#pick_num2(matrix)
grid = [[0,0,0,0],[0,1,0,0],[0,0,0,0]]
print("total unique path")
unique_paths_with_obstacles(grid)
str1="21232523311324"
str2="312123223445"
print("LCS")
LCS(str1, str2)
str1="GCCCTAGCG"
str2="CCGCAATC"
print("LCSS")
LCSS(str1, str2)
print("SED")
SED(str1, str2)
matrix = [[0,1,0,0,0],
          [0,0,1,1,1],
          [1,0,0,0,1],
          [1,0,1,0,1],
          [0,0,1,0,0]]
BFS(matrix)
matrix = [[-1,6,3,-1,-1,-1],
          [6,-1,2,5,-1,-1],
          [3,2,-1,3,4,-1],
          [-1,5,3,-1,2,3],
          [-1,-1,4,2,-1,5],
          [-1,-1,-1,3,5,-1]]
Dijkstra(matrix)
count_one(123)
zero_one(7)
print(gcd(132,44))
array1 = [1,-2,3,5,-3,2]
mac(array1)

def trans(word, i):
    figure = 1
    out = 0
    j = i
    while j < len(word) and ord(word[j]) >= ord('0') and ord(word[j]) <= ord('9')\
    and figure > 0:                                                                  #
        out = out * 10 + (ord(word[j])-ord('0'))
        j += 1
    if j < len(word) and word[j] == '.':
        figure = -1
    while j < len(word) and ord(word[j]) >= ord('0') and ord(word[j]) <= ord('9')\
    and figure < 0:                                                                  #
        out += (ord(word[j])-ord('0')) * pow(10, figure)
        figure -= 1
        j += 1
    return out, j    #
def getlevel(op):
    if op in ['+', '-']:
        return 1
    elif op in ['*', '/']:
        return 2
    elif op == '(':
        return 0
    elif op == '#':
        return -1
def operate(opnd, optr):
    a2 = opnd.pop()
    a1 = opnd.pop()
    op = optr.pop()
    if op == "+":
        opnd.append(a1 + a2)
    elif op == "-":
        opnd.append(a1 - a2)
    elif op == "*":
        opnd.append(a1 * a2)
    elif op == "/":
        opnd.append(a1 / a2)
def compute(word):
    optr = []   #符号
    opnd = []   #数字
    optr.append("#")
    count = len(word)
    is_minus = True
    i = 0
    while i < count:                        #
        if word[i] == "-" and is_minus:
            opnd.append(0)
            optr.append("-")
            i += 1                          #
        elif word[i] == ")":                #
            is_minus = False
            while optr[-1] != "(":
                operate(opnd, optr)
            optr.pop()                      #
            i += 1
        elif ord(word[i])>=ord("0") and ord(word[i])<=ord("9"):
            is_minus = False
            num, j = trans(word, i)
            opnd.append(num)
            i = j
        elif word[i] == "(":
            is_minus = True
            optr.append("(")
            i += 1
        elif word[i] in ['+', '-', '*', '/']:
            while getlevel(word[i]) <= getlevel(optr[-1]):
                operate(opnd, optr)
            optr.append(word[i])
            i += 1
        else:
            return False
    while optr[-1] != "#":
        operate(opnd, optr)
    return opnd.pop(0)
word = "(-6*3+10)/2+2"           
print(compute(word))
def Binary1(array, value):
    left = 0
    right = len(array)-1
    while left <= right:
        mid = left + (left + right)>>1
        if array[mid] > value:
            right = mid - 1
        elif array[mid] < value:
            left = mid + 1
        else:
            return mid
    return -1

def Binary2(array, value, left, right):
    if left > right:
        return -1
    mid = left + (left + right)>>1
    if array[mid] > value:
        return Binary2(array, value, left, mid-1)
    elif array[mid] < value:
        return Binary2(array, value, mid+1, right)
    else:
        return mid
        
def BinarySearch(array, target):
    #轮转后的有序数组二分查找
    low, high = 0, len(array)-1
    while low <= high:
        mid = (low+high)//2
        if target < array[mid]: #先研究比中间值小的，这样只要右侧有序就只用研究左边
            if array[mid] < array[high]: #中间向右有序
                high = mid - 1 #目标只能出现在左边
            else: #中间向左有序
                if target < array[low]: #目标如果小于最左的数，表明实际位置在被轮转的右侧
                    low = mid + 1
                else:
                    high = mid - 1
        elif array[mid] < target: #从这开始完全对称
            if array[low] < array[mid]: #中间向左有序
                low = mid + 1 #目标只能出现在右边
            else: #中间向右有序
                if array[high] < target: #目标如果大于最右侧的数，表明实际位置在被轮转的左侧
                    high = mid - 1
                else:
                    low = mid + 1
        else: # if array[mid] == target
            return mid
    return -1            
array = [5,7,8,11,12,13,15,1,3]
print(BinarySearch(array, 11))

def searchRange(array, target):
    #二分查找target在array中的范围
    return search(array, 0, len(array)-1, target)
def search(array, low, high, target):
    if array[low] == target == array[high]:
        return [low, high]
    if array[low] <= target <= array[high]:
        mid = (low + high) // 2
        l, r = search(array, low, mid, target), search(array, mid+1, high, target)
        return max(l, r) if -1 in l+r else [l[0], r[1]]
    return [-1, -1]
