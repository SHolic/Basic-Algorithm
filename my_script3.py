# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:14:07 2018

@author: 罗骏
"""
import collections
class Trie(object):
    #Trie树实现
    def __init__(self):
        self.root = {}
    def insert(self, word):
        #Inserts a word into the trie.
        cur = self.root
        for l in word:
            if l not in cur:
                cur[l] = {}
            cur = cur[l]
        cur['END'] = 'END'
    def search(self, word):
        #Returns if the word is in the trie.
        cur = self.root
        for l in word:
            if l not in cur:
                return False
            cur = cur[l]
        return 'END' in cur
    def startsWith(self, prefix):
        #Returns if there is any word in the trie that starts with the given prefix.
        cur = self.root
        for l in prefix:
            if l not in cur:
                return False
            cur = cur[l]
        return len(cur) != 0
    
def decodeString(s):
    #"3[a2[c]]"=> "accaccacc"
    stack = []; curNum = 0; curString = ''
    for c in s:
        if c == '[':
            stack.append(curString)
            stack.append(curNum)
            curString = ''
            curNum = 0
        elif c == ']':
            num = stack.pop()
            preString = stack.pop()
            curString = preString + num*curString
        elif c.isdigit():
            curNum = curNum*10 + int(c)
        else:
            curString += c
    return curString

def countBits(num):
    #快速返回直到num的每个数二进制中有多少1
    res = [0 for i in range(num+1)]
    offset = 1
    for index in range(1,num+1):
        if offset*2 == index:
            offset *= 2
        res[index] = res[index - offset] + 1
    return res
    
def maxArea(height):
    #一系列围栏高度中最多存多少水
    i, j = 0, len(height) - 1
    water = 0
    while i < j:
        water = max(water, (j - i) * min(height[i], height[j]))
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return water
    
import math
def numSquares(n):
    #输入的n最少可以表示为几个完全平方数之和
    if n < 2:
        return n
    lst = [i*i for i in range(1,math.floor(math.sqrt(n))+1)]  #
    cnt = 0
    toCheck = [n]
    while toCheck:
        cnt += 1
        temp = []
        for x in toCheck:
            for y in lst:
                if x == y: return cnt
                if x < y: break
                temp.append(x-y)
        toCheck = temp
    return cnt
        
def word_break(s, words):
    #s能否由words里的词组成
 	d = [False for i in range(len(s))]
 	for i in range(len(s)):
 		for w in words:
 			if w == s[i-len(w)+1:i+1] and (d[i-len(w)] or i-len(w) == -1):  #
 				d[i] = True
 	return d[-1]

def maximalSquare(matrix):
    #找输入矩阵中最大的由1组成的正方形
    if not matrix: return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0 for i in range(n)]for j in range(m)]
    ans = 0
    for i in range(m):
        for j in range(n):
            if i == 0 or j == 0:
                dp[i][j] = matrix[i][j]
            elif matrix[i][j] == 1: 
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
            else: dp[i][j] = 0   
            ans = max(ans, dp[i][j])
    return ans ** 2
                
def letterCombinations(digits):
    #输入的电话号码有多少字母组合
    dicts = {"1":"", "2":"abc", "3":"def", "4":"ghi", "5":"jkl", 
             "6":"mno", "7":"pqrs","8":"tuv","9":"wxyz","10":" "}
    result = [""]
    for num in digits:
        new_result = []
        strs = dicts[num]
        for pre_str in result:
            for cur_str in strs:
                new_result += [pre_str + cur_str]
        result = new_result
    return result

def subsetsWithDup(nums):
    #找array的全部子集,并去重
    res = [[]]
    for item, freq in collections.Counter(nums).items():
        res_len = len(res)
        for pre in res[:res_len]:
            for i in range(1, freq+1):
                res.append(pre + [item] * i)
    return res
print(subsetsWithDup('CGCAAT'))
    
def subarraySum(nums, k):
    #输入nums序列有多少组连续相加得k
    count, cur, res = {0: 1}, 0, 0
    for x in nums:
        cur += x
        res += count.get(cur - k, 0) #(object, default if not get)
        count[cur] = count.get(cur, 0) + 1  
    return res
        
def findTargetSumWays(array, S):
    # candidates = [2,3,6,7], target = 7
    #从一列数中修改正负号使相加得S(就是或加或减)，输出有几组
    count = {0: 1}
    for x in array:
        count2 = {}
        for tmpSum in count:
            count2[tmpSum + x] = count2.get(tmpSum + x, 0) + count[tmpSum]
            count2[tmpSum - x] = count2.get(tmpSum - x, 0) + count[tmpSum]
        count = count2
    return count.get(S, 0)
print(findTargetSumWays([1,2,3,4,5], 7))
   
def canJump(array):
    #给一个array序列，每一个值都是在此位置能跳的最远距离，开始在0，问能不能调到最后一个
    i, reach = 0, 0
    while i < len(array) and i <= reach:
        reach = max(i + array[i], reach)
        i += 1
    if i == len(array): return True
    else: return False
print(canJump([3,2,1,0,4]))

def leastInterval(tasks, N):
    #task:"asdfuinreadaf"
    #给几种不同类型的任务task，每种相同任务间隔要求至少N，问全执行完tasks要多久
    taskcount = collections.Counter(tasks).values()
    frequent = max(taskcount)
    frequent_count = 0
    for i in taskcount:
        if taskcount[i] == frequent:
            frequent_count += 1
    return max(len(tasks), (frequent-1)*(N+1)+frequent_count)

def reconstructQueue(people):
    # 一列人站队但是给出的数据打乱，
    #给出[7,4]前面是身高，后面是在他之前有几个比他高的，输出正确站位
    res = []
    for p in sorted(people, key=lambda x: (-x[0], x[1])):
        res.insert(p[1], p)  #(index, obj)
    return res

def findOrder(numCourses, prerequisites):
    # 2, [[1,0],[0,1]] return False
    #有一定数量的课程，每个课程有前置课程要求学习，list里前面是课程，后面是前置课程，问是否能学完
    g = {i:[] for i in range(numCourses)}
    g_inv = {i:[] for i in range(numCourses)}
    for i, j in prerequisites:
        g[i].append(j)
        g_inv[j].append(i)
    q = []
    for course in g:
        if not g[course]:
            q.append(course)    
    lesson_sort = []
    while q:
        pre_learn = q.pop(0)
        lesson_sort.append(pre_learn)
        for course in g_inv[pre_learn]:
            g[course].remove(pre_learn)
            if not g[course]:
                q.append(course)
    return lesson_sort if len(lesson_sort) == numCourses else []
    
def productExceptSelf(array):
    #输出array中每一个值除了自己以外其他值的乘积，不能用除法
    p = 1
    n = len(array)
    output = []
    for i in range(0,n):
        output.append(p)
        p *= array[i]
    p = 1
    for i in range(n-1,-1,-1):
        output[i] *= p
        p = p * array[i]
    return output        
    
def getSum(a, b):
    # 不用加号计算a+b，这个是标准写法
    # 32 bits integer max
    MAX = 0x7FFFFFFF
    # mask to get last 32 bits
    mask = 0xFFFFFFFF
    while b != 0:
        # ^ get different bits and & gets double 1s, << moves carry
        a, b = (a ^ b) & mask, ((a & b) << 1) & mask
    # if a is negative, get a's 32 bits complement positive first
    # then get 32-bit positive's Python complement negative
    return a if a <= MAX else ~(a ^ mask)

def groupAnagrams(array):
    '''["eat", "tea", "tan", "ate", "nat", "bat"]=>
    ["ate","eat","tea"],
    ["nat","tan"],
    ["bat"]
    '''
    dist = {}
    for i in array:
        key = tuple(sorted(i))
        dist[key] = dist.get(key, []) + [i]
    return dist.values()

def removeInvalidParentheses(s):
    "输入一个字符串，有左右括号，返回去除其中不对称的括号"
    result = []
    remove(s, result, 0, 0, ('(', ')'))
    return result
def remove(s, result, last_i, last_j, par):
    count = 0
    for i in range(last_i, len(s)):
        count += (s[i] == par[0]) - (s[i] == par[1])
        if count >= 0:
            continue
        for j in range(last_j, i + 1):
            if s[j] == par[1] and (j == last_j or s[j - 1] != par[1]):
                #s[j - 1] != par[1]是为了防止连续两个')'时删哪个都一样还删了两回
                remove(s[:j] + s[j + 1:], result, i, j, par)
        return
    reversed_s = s[::-1]
    if par[0] == '(':
        remove(reversed_s, result, 0, 0, (')', '('))
    else:
        result.append(reversed_s)
print(removeInvalidParentheses('(()(())'))

def longestValidParentheses(s):
    #最长的对称子序列
    stack, res, s = [0], 0, ')'+s                       #
    for i in range(1, len(s)):                          #
        if s[i] == ')' and s[stack[-1]] == '(':
            stack.pop()
            #如果完全对称，stack会一直pop到第一个(处，这样res就会给出最早的一个对称位置
            res = max(res, i - stack[-1])
        else:
            stack.append(i)
    return res
s = '()((())()()))()()'
print(longestValidParentheses(s))

from collections import deque
def maxSlidingWindow(nums, k):
    '给定一系列数字，和一个窗口大小k，输出在这个窗口从前到后过nums时每次窗口中最大的数'
    if not nums: return []
    res = []
    dq = deque()  # store index
    #这个和普通list很像，但是多两个操作：.appendleft()从左侧加入deque
    #.popleft()从左侧弹出元素
    for i in range(len(nums)):
        if dq and dq[0]<i-k+1:  # out of the window
            dq.popleft()
        while dq and nums[dq[-1]]<nums[i]:  # remove impossible candidate
            dq.pop()
        dq.append(i)
        if i>k-2:
            res.append(nums[dq[0]])
    return res

def flatten(self, root):
    '''1. flatten left subtree
    2. find left subtree's tail
    3. set root's left to None, root's right to root'left, tail's right to root.right
    4. flatten the original right subtree'''
    # escape condition
    if not root:
        return
    right = root.right
    if root.left:
        # flatten left subtree
        self.flatten(root.left)
        # find the tail of left subtree
        tail = root.left
        while tail.right:
            tail = tail.right
        # left <-- None, right <-- left, tail's right <- right
        root.left, root.right, tail.right = None, root.left, right
    # flatten right subtree
    self.flatten(right)
