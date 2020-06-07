import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
from enum import Enum,unique
import heapq

#############################
## 一些与数值误差相关的常数的定义
INF = 1e16 ## infinity
MAX = 1e20 ## 默认最大值，比infinity稍大
TOL = 1e-6 ## 数值误差tolerance
#############################

#############################
## 用于记录状态的枚举类型
@unique
class SolveStatus(Enum):
    ## 求解状态
    ONGOING = 0 ## 求解进行中
    OPT = 1 ## 最优解
    INFEAS = 2 ## 不可行
    ERR = -1 ## error
#############################

class Node:
    '''
    B&B树中的结点，对应一个bound tightened的子问题。主要负责管理以下信息：
        作为树的一个元素，需要存储指向父节点和子节点的指针以及自身的idx、深度。
        作为原问题的一个子问题，需要存储相比原问题的modification。
        作为一个MIP问题，需存储解的相关信息：obj、sol等。
        为了B&B整体流程的效果评估，需要存储到目前为止子树的目标值上下界。
    '''
    
    def __init__(self,node_idx=0,parent=None,
                 var_idx=-1,var_sol=0,var_bnd=0,
                 var_lb=0,var_ub=0,init_var_lb=0,init_var_ub=0):
        ## 在B&B树中的位置及状态
        self.idx = node_idx
        self.parent,self.childs = parent,[]
        self.level = 0
        if parent is not None:
            self.level = parent.level + 1
        self.is_processed,self.is_active = False,False

        ## 定义子问题所需的边界信息
        self.var_idx = var_idx
        self.var_bnd = var_bnd
        self.var_delta = var_bnd - var_sol
        if parent is not None:
            self.local_vars_lb_dict,self.local_vars_ub_dict = \
                parent.local_vars_lb_dict.copy(),parent.local_vars_ub_dict.copy()
            if self.var_delta > 0:
                self.local_vars_lb_dict[(var_idx,init_var_lb)] = var_lb ## new lb
            else:
                self.local_vars_ub_dict[(var_idx,init_var_ub)] = var_ub ## new ub
        else:
            self.local_vars_lb_dict,self.local_vars_ub_dict = dict(),dict()

        ## 其他求解相关信息
        self.LB = self.obj = MAX
        if parent is not None:
            self.LB = parent.obj ## 用父结点的求解目标值作为子结点的目标值下界估计
        self.branch_var_idx = -1
        self.branch_var_sol = 0

    def mark_processed(self):
        self.is_processed = True
        
    def activate(self,solver):
        self.mark_processed()
        self.is_active = True
        self.sol,self.sol_feas = None,False

    def update(self,sol,obj,solver):
        ## 更新子问题的相关信息
        self.sol,self.obj,self.LB = sol,obj,obj
        sol_bool_infeas = (np.abs(sol - np.round(sol)) > TOL) & solver.int_vars_bool
        self.sol_feas = (~np.any(sol_bool_infeas))

    def deactivate(self):
        ## 删除中间变量，释放内存
        self.is_active = False
        del self.sol,self.sol_feas
        del self.local_vars_lb_dict,self.local_vars_ub_dict
        
    def clear(self):
        if self.is_active:
            self.deactivate()
        del self.parent,self.childs

class Brancher:
    def __init__(self):
        self.branch = self.branch_inf_most

    def branch_inf_most(self,solver,node):
        ## 选出距离整数最远的变量
        score = np.abs(node.sol - np.round(node.sol))
        var_idx = solver.int_vars_idx[np.argmax(score[solver.int_vars_idx])]
        return var_idx

class Selector:
    def __init__(self):
        self.select = self.select_bfs

    def select_bfs(self,solver,curr_node):
        ## 选出下界
        node_idx = heapq.heappop(solver.unprocessed_node_bounds)[1]
        while (len(solver.unprocessed_node_bounds) > 0 and node_idx not in solver.unprocessed_node_idxs):
            node_idx = heapq.heappop(solver.unprocessed_node_bounds)[1]
        if node_idx not in solver.unprocessed_node_idxs:
            node_idx = -1
        return node_idx

class BBSolver:
    '''
    B&B求解流程
    '''
    def __init__(self,problem):
        '''
        B&B流程初始化
        
        输入
        problem: CyClpSimplex对象，定义了初始MIP问题
        '''
        ## 获取MIP问题的相关信息（通过CyLP的API）
        self.problem = problem
        self.int_vars_bool = problem.integerInformation.copy() > 0 ## 决策变量的整数信息
        self.int_vars_idx = np.where(self.int_vars_bool)[0] ## 整数决策变量的下标
        self.vars_ub = problem.variablesUpper ## 对问题的变量上界的reference
        self.vars_lb = problem.variablesLower ## 对问题的变量下界的reference
        self.num_vars = len(self.vars_ub) ## 问题的变量个数
        self.init_vars_ub = self.vars_ub.copy() ## 问题的原始变量上界
        self.init_vars_lb = self.vars_lb.copy() ## 问题的原始变量上界
        self.primal_sol = self.problem.primalVariableSolution ## 对问题的LP relaxation的最优primal解的reference
        self.dual_sol = self.problem.dualVariableSolution ## 对问题的LP relaxation的最优dual解的reference
        
        ## 初始化各执行对象
        self.brancher = Brancher() ## 负责branching的对象
        self.selector = Selector() ## 负责node selection的对象
        
        ## 初始化B&B树的相关信息
        self.nodes = [] ## 生成结点队列
        self.num_nodes,self.next_node_idx = 0,0 ## 当前结点数和下一个结点下标
        self.unprocessed_node_idxs = set([]) ## 未处理结点下标
        self.unprocessed_node_bounds = [] ## 未处理结点的下界信息
        heapq.heapify(self.unprocessed_node_bounds)
        
        self.global_LB = MAX ## MIP问题目标值的全局下界
        self.global_obj,self.global_sol = MAX,None ## MIP问题的当前最优解和目标值
        
    def add_node(self,node):
        self.nodes.append(node)
        ## 更新未处理结点的信息
        self.unprocessed_node_idxs.add(node.idx)
        heapq.heappush(self.unprocessed_node_bounds,(node.LB,node.idx))
        self.num_nodes += 1
        self.next_node_idx += 1
        
    def modify_problem_bounds(self,node):
        '''
        根据Node中存储的子问题信息，对问题接口中的变量上下界进行修改
        '''
        for (var_idx,init_var_bnd),var_bnd in node.local_vars_lb_dict.items():
            self.vars_lb[var_idx] = var_bnd
        for (var_idx,init_var_bnd),var_bnd in node.local_vars_ub_dict.items():
            self.vars_ub[var_idx] = var_bnd
    
    def recover_problem_bounds(self,node):
        '''
        根据Node中存储的子问题信息，对问题接口中的变量上下界进行恢复
        '''
        for (var_idx,init_var_bnd),var_bnd in node.local_vars_lb_dict.items():
            self.vars_lb[var_idx] = init_var_bnd
        for (var_idx,init_var_bnd),var_bnd in node.local_vars_ub_dict.items():
            self.vars_ub[var_idx] = init_var_bnd
    
    def node_solve(self,node):
        '''
        Node中的子问题求解
        '''
        ## dual simplex求解LP relaxation，并获取相应解和目标值
        self.problem.dual() 
        sol = self.primal_sol.copy()
        obj = self.problem.objectiveValue if self.problem.getStatusCode() == 0 else MAX
        if node.level == 0:
            self.root_sol = sol.copy()
        
        ## 根据求解结果对node的信息进行更新
        node.update(sol,obj,self)
        
        ## 判断是否产生了更好的整数解
        if node.sol_feas and node.obj < self.global_obj:
            ## 更新全局可行解
            self.global_obj,self.global_sol = node.obj,node.sol.copy()        
            
    def create_node(self,parent_node,var_idx,var_lb,var_ub,var_sol,var_bnd):
        '''
        创建子结点
        '''
        new_node = Node(node_idx=self.next_node_idx,parent=parent_node,
                        var_idx=var_idx,var_lb=var_lb,var_ub=var_ub,var_sol=var_sol,var_bnd=var_bnd,
                        init_var_lb=self.init_vars_lb[var_idx],init_var_ub=self.init_vars_ub[var_idx])
        parent_node.childs += [new_node]
        self.add_node(new_node)
    
    def branch(self,node):
        '''
        branching
        '''
        ## 调用brancher的branch方法找到branch var idx
        var_idx = self.brancher.branch(self,node) 
        node.branch_var_idx,node.branch_var_sol = var_idx,node.sol[var_idx]
        
        ## 创建子结点
        x_j = node.sol[var_idx]
        u_j,l_j = self.vars_ub[var_idx],self.vars_lb[var_idx]
        u_j_new,l_j_new = math.floor(x_j),math.ceil(x_j)
        for (var_lb,var_ub,var_sol,var_bnd) in \
            [(l_j,u_j_new,x_j,u_j_new),(l_j_new,u_j,x_j,l_j_new)]:
            self.create_node(node,var_idx,var_lb,var_ub,var_sol,var_bnd)
    
    def node_process(self,node_idx):
        '''
        对选出的node/子问题进行处理
        '''
        if node_idx not in self.unprocessed_node_idxs:
            print(self.unprocessed_node_idxs,node_idx)
            
        node = self.nodes[node_idx]
        self.unprocessed_node_idxs.remove(node_idx)

        if node.LB >= self.global_obj:
            ## 可以直接剪枝
            node.mark_processed()
            prun_status = 0
            return prun_status

        node.activate(self) ## 激活node，生成局部信息
        self.modify_problem_bounds(node) ## 更新问题接口中的变量上下界
        self.node_solve(node) ## 求解子问题
        
        ## 更新BB树的下界信息
        node_child = node
        update_lb_flag = True
        while (node_child.level > 0) and update_lb_flag:
            node_parent = node_child.parent
            curr_LB = MAX
            for child in node_parent.childs:
                if child.LB < curr_LB:
                    curr_LB = child.LB
                if not child.is_processed:
                    ## no need to further update LB
                    update_lb_flag = False
                    break
            node_parent.LB = curr_LB
            node_child = node_parent  
        self.global_LB = self.root_node.LB
        
        prun_status = 0
        if not node.sol_feas and node.LB < self.global_obj:
            ## 子问题整数不可行且子问题最优解小于当前最优整数解，继续branch生成子问题
            self.branch(node)
            prun_status = 1
        
        self.recover_problem_bounds(node) ## 恢复问题接口中的变量上下界
        node.deactivate() ## 释放内存
        return prun_status
        
    def node_select(self,node_idx):
        '''
        node selection
        '''
        if len(self.unprocessed_node_idxs) == 0:
            ## 没有未处理的子问题
            return -1
        node = self.nodes[node_idx]
        ## 调用selector的select方法找到下一个node idx
        select_node_idx = self.selector.select(self,node)
        return select_node_idx

    def solve_root_node(self):
        ## 生成根结点
        root_node = Node()
        self.root_node = root_node
        self.add_node(root_node)
        ## 执行简化版的process
        self.unprocessed_node_idxs.remove(root_node.idx)
        root_node.activate(self) ## 激活node，生成局部信息
        self.node_solve(root_node) ## 求解子问题
        self.root_sol = root_node.sol ## 根结点（初始问题）的LP relaxation最优解
        self.global_LB = root_node.LB ## 更新BB树的上下界信息
        status = SolveStatus.ONGOING
        if not root_node.sol_feas and self.global_LB < self.global_obj:
            self.branch(root_node)
        else:
            status = SolveStatus.OPT
        root_node.deactivate() ## 释放内存
        return status
    
    def solve(self,time_limit=3600,iter_limit=1000000,if_disp=True,disp_iters=100000):
        '''
        主求解流程
        '''
        tt = time.time()
        status = self.solve_root_node()
        if (status == SolveStatus.ONGOING):
            node_idx,self.iter_ = self.root_node.idx,0
            while (self.iter_ < iter_limit) and (time.time()-tt < time_limit):
                self.iter_ += 1
                node_idx = self.node_select(node_idx)
                if node_idx < 0:
                    ## 没有选出下一个子问题，报错退出
                    status = SolveStatus.ERR
                    break

                self.node_process(node_idx)
                if (self.global_obj - self.global_LB) < TOL:
                    ## 上下界差距小于tolerance，可认为找到最优解
                    status = SolveStatus.OPT
                    break

                if if_disp:
                    if self.iter_ % disp_iters == 0:
                        print('iter {} time {:.2f} LB {:.3e} UB {:.3e}'.format(self.iter_,time.time()-tt,self.global_LB,self.global_obj))
        LB,obj,dt = self.global_LB,self.global_obj,time.time()-tt
        return (dt,status,LB,obj)