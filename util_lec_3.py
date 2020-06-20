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
class BranchType(Enum):
    ## branch方法类型
    MOST_INF = 1 ## 选取距离整数最远的变量
    LEAST_INF = 2 ## 选取距离整数最近的变量
    PSEUDO = 3 ## 通过pseudo cost估计LP目标值，进而选取变量
    FULL_STRONG = 4 ## 通过LP求解结果选取变量
    STRONG = 5 ## 通过LP求解结果选取变量
    RELIABLE = 6 ## 结合strong branching和pseudo cost branching

@unique
class SelectType(Enum):
    ## select方法类型
    DFS = 1 ## deep first search，选取最大深度的子问题
    BFS = 2 ## best first search，选取目标值下界最小的子问题
    BE = 3 ## best estimate，选取estimate最小的子问题
    BFS_PLUNGE = 4 ## 结合bfs和dfs
    BE_PLUNGE = 5 ## 结合be和dfs
    BFS_DIVE = 6 ## 结合bfs和dfs，先通过dfs找可行解
    BE_DIVE = 7 ## 结合be和dfs，先通过dfs找可行解

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
    
    def __init__(self,node_idx=0,parent=None,cost_manager=None,
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

        ## 进一步初始化estimate
        self.estimate = MAX
        if parent is not None:
            self.estimate = parent.estimate
            if cost_manager is not None:
                ## 根据pseudo cost对estimate做调整
                ## Note：要求parent node保留了中间变量delta_plus和delta_minus
                var_delta = self.var_delta
                cost_plus,cost_minus = cost_manager.get_cost()
                if var_delta > 0:
                    self.estimate = parent.estimate + var_delta * cost_plus[var_idx] ## 向上调整
                else:
                    self.estimate = parent.estimate - var_delta * cost_minus[var_idx] ## 向下调整
                ## 去除变量var_idx原来的贡献值
                self.estimate -= min(parent.delta_plus[var_idx]*cost_plus[var_idx],
                                     parent.delta_minus[var_idx]*cost_minus[var_idx])

    def mark_processed(self):
        self.is_processed = True
        
    def activate(self,solver):
        self.mark_processed()
        self.is_active = True
        self.sol,self.sol_feas = None,False
        ## 增加中间变量
        self.sol_bool_infeas,self.sol_int_infeas,self.sol_bool_feas = None,None,None ## 记录每个变量是否满足整数约束
        self.delta_minus,self.delta_plus = None,None ## 每个变量为满足整数可行性约束所应变化的大小（-、+）

    def update(self,sol,obj,solver):
        '''
        更新子问题的相关信息
        '''
        ## 保存LP求解结果
        self.sol,self.obj,self.LB = sol,obj,obj
        ## 增加中间变量的计算
        self.sol_bool_infeas = (np.abs(sol - np.round(sol)) > TOL) & solver.int_vars_bool
        self.sol_int_infeas = np.where(self.sol_bool_infeas)[0]
        self.sol_feas = (~np.any(self.sol_bool_infeas))
        self.sol_bool_feas = ~self.sol_bool_infeas
        sol_elem_infeas = sol[self.sol_bool_infeas]
        self.delta_minus,self.delta_plus = np.zeros((len(sol),)),np.zeros((len(sol),))
        self.delta_minus[self.sol_bool_infeas] = sol_elem_infeas - np.floor(sol_elem_infeas)
        self.delta_plus[self.sol_bool_infeas] = np.ceil(sol_elem_infeas) - sol_elem_infeas
        ## 根据LP求解结果更新estimate
        if obj < MAX and sol is not None:
            cost_plus,cost_minus = solver.cost_manager.get_cost()
            score = np.minimum(self.delta_plus * cost_plus,self.delta_minus * cost_minus)
            score_sum = np.sum(score)
            self.estimate = self.obj + score_sum
        else:
            self.estimate = self.obj

    def deactivate(self):
        ## 删除中间变量，释放内存
        self.is_active = False
        del self.sol,self.sol_feas
        del self.local_vars_lb_dict,self.local_vars_ub_dict
        ## 增加中间变量，与activate对应
        del self.sol_bool_infeas,self.sol_int_infeas,self.sol_bool_feas ## 记录每个变量是否满足整数约束
        del self.delta_minus,self.delta_plus ## 每个变量为满足整数可行性约束所应变化的大小（-、+）
        
    def clear(self):
        if self.is_active:
            self.deactivate()
        del self.parent,self.childs

class PseudoCostManager:
    def __init__(self,problem):
        ## 初始化
        self.num_vars = problem.nCols
        self.reset()

    def reset(self):
        self.pseudo_cost_info = (np.zeros((self.num_vars,),dtype=int),np.zeros((self.num_vars,),dtype=int),
                np.zeros((self.num_vars,),dtype=float),np.zeros((self.num_vars,),dtype=float),
                np.ones((self.num_vars,),dtype=float),np.ones((self.num_vars,),dtype=float))
        self.global_pseudo_cost_info = (0,0,0,0,1,1)

    def update(self,var_idx,var_delta,obj_delta):
        '''
        pseudo cost = (\Delta obj / \Delta x_j)的均值；在这里，每步更新逻辑通过两个累加器实现
        '''
        count_plus,count_minus,value_plus,value_minus,cost_plus,cost_minus = self.pseudo_cost_info
        count_plus_global,count_minus_global,value_plus_global,\
            value_minus_global,cost_plus_global,cost_minus_global = self.global_pseudo_cost_info

        if var_delta > 0:
            ## 更新变量维度pseudo cost
            count_plus[var_idx] += 1
            value_plus[var_idx] += obj_delta / var_delta
            cost_plus[var_idx] = value_plus[var_idx] / count_plus[var_idx]
            ## 更新全局pseudo cost
            count_plus_global += 1
            value_plus_global += obj_delta / var_delta
            cost_plus_global = value_plus_global / count_plus_global
        if var_delta < 0:
            ## 更新变量维度pseudo cost
            count_minus[var_idx] += 1
            value_minus[var_idx] += -obj_delta / var_delta
            cost_minus[var_idx] = value_minus[var_idx] / count_minus[var_idx]
            ## 更新全局pseudo cost
            count_minus_global += 1
            value_minus_global += -obj_delta / var_delta
            cost_minus_global = value_minus_global / count_minus_global

        ## 对于pseudo cost信息缺失的变量，用全局pseudo cost代替
        cost_plus[count_plus == 0] = cost_plus_global
        cost_minus[count_minus == 0] = cost_minus_global

        self.pseudo_cost_info = (count_plus,count_minus,value_plus,value_minus,cost_plus,cost_minus)
        self.global_pseudo_cost_info = (count_plus_global,count_minus_global,value_plus_global,
                                        value_minus_global,cost_plus_global,cost_minus_global)
    
    def get_cost(self):
        '''
        获取各变量的pseudo cost信息
        '''
        _,_,_,_,cost_plus,cost_minus = self.pseudo_cost_info
        return cost_plus,cost_minus

class Brancher:
    def __init__(self,problem,branch_type=BranchType.RELIABLE):
        ## 确定branch方法类型
        self.branch_type = branch_type
        self.branch = self.branch_reliable
        if self.branch_type == BranchType.MOST_INF:
            self.branch = self.branch_inf_most
        elif self.branch_type == BranchType.LEAST_INF:
            self.branch = self.branch_inf_least
        elif self.branch_type == BranchType.FULL_STRONG:
            self.branch = self.branch_full_strong
        elif self.branch_type == BranchType.STRONG:
            self.branch = self.branch_strong
        elif self.branch_type == BranchType.PSEUDO:
            self.branch = self.branch_pseudo
        elif self.branch_type == BranchType.RELIABLE:
            self.branch = self.branch_reliable
    
        ## 初始化辅助变量
        self.max_iter_strong = 100 ## strong branching中用于控制dual simplex最大迭代次数
        self.max_var_strong = 100 ## strong branching中用于控制做strong branch的变量最大个数
        self.max_look_ahead = 10 ## strong branching中，如果连续多个变量对最优解没有改进，则停止搜索
        self.max_level_strong = 10 ## strong branching中用于在strong和pseudo之间切换的阈值
        self.reliable_threshold = 5 ## reliable branching中用于在strong和pseudo之间切换的阈值

    def score_func(self,score_plus,score_minus):
        '''
        打分函数，根据上下侧得分生成综合分
        '''
        score = np.maximum(score_plus,TOL) * np.maximum(score_minus,TOL)
        return score

    def check_dual_inf(self,solver,problem):
        '''
        检查当前解的对偶可行性，用于确认所得目标值是否可用
        '''
        var_status = problem.getBasisStatus()[0]
        dsol = solver.dual_sol
        dual_inf_num = np.sum(dsol[var_status == 2] >= problem.dualTolerance) + \
                        np.sum(dsol[var_status == 3] <= -problem.dualTolerance)
        if dual_inf_num > 0:
            # print(problem.objectiveValue,dual_inf_num,
            #      np.sum(np.maximum(dsol[var_status == 2],0)),
            #      np.sum(np.minimum(dsol[var_status == 3],0))) ## for debug
            return True
        return False
    
    def score_var_strong(self,solver,node,var_idx,max_iter=None):
        '''
        对列/变量j，通过调整上下界的方式（strong branching）获取其分值
        参数max_iter用于控制每个子问题的dual simplex迭代次数
        '''
        ## 获取问题基本信息
        problem = solver.problem
        basis_status = problem.getBasisStatus() ## 取出最优基
        l,u = solver.vars_lb,solver.vars_ub
        l_j,u_j,x_j = l[var_idx],u[var_idx],node.sol[var_idx]
        u_j_new,l_j_new = math.floor(x_j),math.ceil(x_j)
        obj = node.obj
        
        ## 调整上下界求解
        objs_new = []
        for (var_lb,var_ub,var_sol,var_bnd) in [(l_j,u_j_new,x_j,u_j_new),(l_j_new,u_j,x_j,l_j_new)]:
            l[var_idx],u[var_idx] = var_lb,var_ub
            if max_iter is not None:
                problem.maxNumIteration = max_iter
            problem.dual()
            is_dual_inf = self.check_dual_inf(solver,problem) ## CyLP缺少相应接口，手动实现，效率低
            obj_new = problem.objectiveValue if problem.getStatusCode() in (0,3) else MAX
            problem.setBasisStatus(*basis_status) ## 恢复最优基
            l[var_idx],u[var_idx] = l_j,u_j ## 恢复上下界
            if max_iter is not None:
                problem.maxNumIteration = 2 ** 31 - 1
            # problem.dual() ## only for debug
            if is_dual_inf:
                ## 如果对偶不可行，则求解信息不可用，废弃
                return 0
            objs_new += [obj_new]
            ## 用obj结果更新psc信息
            ##（因为dual simplex iter的限制，这里给出的psc信息与node solve中的不完全一致，需要做check）
            if obj_new >= obj and obj_new < MAX:
                solver.cost_manager.update(var_idx,var_bnd - var_sol,obj_new - obj)
        
        ## 整理结果
        score = self.score_func(objs_new[0]-obj,objs_new[1]-obj)        
        return score

    def score_pseudo_cost(self,solver,node):
        cost_plus,cost_minus = solver.cost_manager.get_cost()
        score = self.score_func(cost_plus*node.delta_plus,cost_minus*node.delta_minus)
        score[node.sol_bool_feas] = 0
        return score

    def branch_inf_most(self,solver,node):
        '''
        选取距离整数最远的变量
        '''
        score = np.minimum(node.delta_minus,node.delta_plus)
        var_idx = node.sol_int_infeas[np.argmax(score[node.sol_int_infeas])]
        return var_idx

    def branch_inf_least(self,solver,node):
        '''
        选取距离整数最近的变量
        '''
        score = np.minimum(node.delta_minus,node.delta_plus)
        var_idx = node.sol_int_infeas[np.argmin(score[node.sol_int_infeas])]
        return var_idx
    
    def branch_pseudo(self,solver,node):
        '''
        psc branching
        '''
        ## 计算psc score
        score = self.score_pseudo_cost(solver,node)
        ## 选取score最大的变量
        var_idx = solver.int_vars_idx[np.argmax(score[solver.int_vars_idx])]
        ## err check
        if node.sol_bool_feas[var_idx]:
            print('branch err',var_idx)
            var_idx = node.sol_int_infeas[np.argmax(score[node.sol_int_infeas])]
        return var_idx

    def branch_full_strong(self,solver,node):
        '''
        full strong branching
        '''
        strong_var_idxs = node.sol_int_infeas
        best_score_strong,best_var_idx_strong = -1,strong_var_idxs[0]
        for score_idx,var_idx in enumerate(strong_var_idxs):
            ## 对每个变量调用full strong branching规则获取得分
            score_strong_tmp = self.score_var_strong(solver,node,var_idx)
            if score_strong_tmp > best_score_strong:
                best_score_strong,best_var_idx_strong = score_strong_tmp,var_idx
        var_idx = best_var_idx_strong
        return var_idx

    def branch_strong(self,solver,node):
        '''
        strong branching
        '''
        if node.level >= self.max_level_strong:
            return self.branch_pseudo(solver,node)
        ## 计算psc score
        score = self.score_pseudo_cost(solver,node)
        
        ## 选取top k个变量做strong branching
        strong_var_idxs = node.sol_int_infeas
        if len(strong_var_idxs) > self.max_var_strong:
            strong_var_idxs = strong_var_idxs[np.argpartition(score[strong_var_idxs], 
                                                              -self.max_var_strong)[-self.max_var_strong:]
                                             ]
        strong_var_idxs = strong_var_idxs[np.argsort(score[strong_var_idxs])[::-1]]
        
        var_idx = strong_var_idxs[0]
        best_score_strong,best_var_idx_strong,look_ahead_cnt = score[var_idx],var_idx,0
        for score_idx,var_idx in enumerate(strong_var_idxs):
            ## 对每个变量调用branching规则获取得分
            score_strong_tmp = self.score_var_strong(solver,node,var_idx,self.max_iter_strong)
            if score_strong_tmp > best_score_strong:
                best_score_strong,best_var_idx_strong,look_ahead_cnt = \
                    score_strong_tmp,var_idx,0
            else:
                look_ahead_cnt += 1
            if look_ahead_cnt > self.max_look_ahead:
                break
        var_idx = best_var_idx_strong
        
        ## err check
        if node.sol_bool_feas[var_idx]:
            print('branch err',var_idx)
            var_idx = node.sol_int_infeas[np.argmax(score[node.sol_int_infeas])]
        return var_idx

    def branch_reliable(self,solver,node):
        '''
        reliable branching
        '''
        ## 计算psc score
        score = self.score_pseudo_cost(solver,node)
        ## 选取score最大的变量
        var_idx = solver.int_vars_idx[np.argmax(score[solver.int_vars_idx])]
        
        ## 根据psc信息的置信度，选取top k个变量做strong branching
        count_plus,count_minus,_,_,_,_ = solver.cost_manager.pseudo_cost_info
        bool_strong = (node.sol_bool_infeas) & ((count_plus <= self.reliable_threshold) | \
                                        (count_minus <= self.reliable_threshold))
        if np.any(bool_strong):
            strong_var_idxs = np.where(bool_strong)[0]
            if len(strong_var_idxs) > self.max_var_strong:
                strong_var_idxs = strong_var_idxs[np.argpartition(score[strong_var_idxs], 
                                                                  -self.max_var_strong)[-self.max_var_strong:]
                                                 ]
            strong_var_idxs = strong_var_idxs[np.argsort(score[strong_var_idxs])[::-1]]

            ## 选取score最大的变量
            best_score_strong,best_var_idx_strong,look_ahead_cnt = score[var_idx],var_idx,0
            for score_idx,var_idx in enumerate(strong_var_idxs):
                ## 对每个变量调用branching规则获取得分
                score_strong_tmp = self.score_var_strong(solver,node,var_idx,self.max_iter_strong)
                if score_strong_tmp > best_score_strong:
                    best_score_strong,best_var_idx_strong,look_ahead_cnt = \
                        score_strong_tmp,var_idx,0
                else:
                    look_ahead_cnt += 1
                if look_ahead_cnt > self.max_look_ahead:
                    break
            var_idx = best_var_idx_strong
        
        ## err check
        if node.sol_bool_feas[var_idx]:
            print('branch err',var_idx)
            var_idx = node.sol_int_infeas[np.argmax(score[node.sol_int_infeas])]
        return var_idx

class Selector:
    def __init__(self,problem,select_type=SelectType.BE_PLUNGE):
        ## 确定select方法类型
        self.select_type = select_type
        self.select = self.select_bfs_plunge
        if self.select_type == SelectType.DFS:
            self.select = self.select_dfs
        elif self.select_type == SelectType.BFS:
            self.select = self.select_bfs
        elif self.select_type == SelectType.BE:
            self.select = self.select_be
        elif self.select_type == SelectType.BFS_PLUNGE:
            self.select = self.select_bfs_plunge
        elif self.select_type == SelectType.BE_PLUNGE:
            self.select = self.select_be_plunge
        elif self.select_type == SelectType.BFS_DIVE:
            self.select = self.select_bfs_dive
        elif self.select_type == SelectType.BE_DIVE:
            self.select = self.select_be_dive

        ## 初始化辅助变量
        self.plunge_count = 0 ## plunging连续触发次数计数器
        self.plunge_count_max = 1 ## plunging最大连续触发次数
        self.plunge_rate = 0.3 ## 用于调整plunge_count_max的具体值

    def select_dfs(self,solver,curr_node):
        '''
        deep first search
        '''
        is_root = False ## 判断是否为根结点
        while not is_root:
            ## 检查子结点
            if len(curr_node.childs) > 0:
                childs = curr_node.childs
                if len(childs) > 1 and curr_node.branch_var_idx >= 0:
                    ## 根据当前结点的branch方向，优化遍历顺序；
                    ## TODO: 这部分计算可以前置到curr_node的子结点生成中
                    x = curr_node.branch_var_sol
                    x0 = solver.root_sol[curr_node.branch_var_idx]
                    if x > x0 and childs[0].idx < childs[-1].idx:
                        childs = childs[::-1]
                for node in childs:
                    if not node.is_processed:
                        return node.idx
            ## 如果没有未处理的子结点，沿B&B树向上一级
            curr_node = curr_node.parent
            if curr_node is None:
                is_root = True
                return -1
        return -1
        
    def plunging(self,solver,curr_node):
        '''
        plunging可以看成是简化版的dfs
        '''
        ## 更新plunge_count_max的取值
        self.plunge_count_max = max(self.plunge_count_max,int(self.plunge_rate * curr_node.level),1)
        if self.plunge_count < self.plunge_count_max:
            ## 检查子结点，与dfs相同
            if len(curr_node.childs) > 0:
                childs = curr_node.childs
                if len(childs) > 1 and curr_node.branch_var_idx >= 0:
                    ## 根据当前结点的branch方向，优化遍历顺序
                    x = curr_node.branch_var_sol
                    x0 = solver.root_sol[curr_node.branch_var_idx]
                    if x > x0 and childs[0].idx < childs[-1].idx:
                        childs = childs[::-1]
                for node in childs:
                    if not node.is_processed:
                        self.plunge_count += 1
                        return node.idx

            ## 检查sibling结点
            if curr_node.parent is not None:
                siblings = curr_node.parent.childs
                for node in siblings:
                    if (not node.is_processed) and (node.idx != curr_node.idx):
                        self.plunge_count += 1
                        return node.idx

        self.plunge_count = 0
        return -1

    def select_bfs(self,solver,curr_node):
        '''
        best first search
        '''
        ## 从队列中取出下界最优的未处理子结点
        node_idx = heapq.heappop(solver.unprocessed_node_bounds)[1]
        while (len(solver.unprocessed_node_bounds) > 0 and node_idx not in solver.unprocessed_node_idxs):
            node_idx = heapq.heappop(solver.unprocessed_node_bounds)[1]
        if node_idx not in solver.unprocessed_node_idxs:
            node_idx = -1
        return node_idx

    def select_be(self,solver,curr_node):
        '''
        best estimate search
        '''
        ## 从队列中取出estimate最优的未处理子结点
        node_idx = heapq.heappop(solver.unprocessed_node_estimates)[1]
        while (len(solver.unprocessed_node_estimates) > 0 and node_idx not in solver.unprocessed_node_idxs):
            node_idx = heapq.heappop(solver.unprocessed_node_estimates)[1]
        if node_idx not in solver.unprocessed_node_idxs:
            node_idx = -1
        return node_idx

    def select_bfs_plunge(self,solver,curr_node):
        '''
        bfs + plunging：先尝试plunge，如果不可行再做bfs
        '''
        idx = self.plunging(solver,curr_node)
        if idx < 0:
            idx = self.select_bfs(solver,curr_node)
        return idx

    def select_be_plunge(self,solver,curr_node):
        '''
        be + plunging：先尝试plunge，如果不可行再做be
        '''
        idx = self.plunging(solver,curr_node)
        if idx < 0:
            idx = self.select_be(solver,curr_node)
        return idx
    
    def select_bfs_dive(self,solver,curr_node):
        '''
        bfs + plunging + dive：先尝试dfs寻找可行解，如果不可行再做bfs_plunge
        '''
        if solver.global_obj >= MAX and solver.iter_ <= 10000:
            idx = self.select_dfs(solver,curr_node)
        else:
            idx = self.select_bfs_plunge(solver,curr_node)
        return idx
    
    def select_be_dive(self,solver,curr_node):
        '''
        be + plunging + dive：先尝试dfs寻找可行解，如果不可行再做be_plunge
        '''
        if solver.global_obj >= MAX and solver.iter_ <= 10000:
            idx = self.select_dfs(solver,curr_node)
        else:
            idx = self.select_be_plunge(solver,curr_node)
        return idx

class BBSolver:
    '''
    B&B求解流程
    '''
    def __init__(self,problem,branch_type=None,select_type=None):
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
        self.reset_branch_type(branch_type)
        self.reset_select_type(select_type)
        self.cost_manager = PseudoCostManager(self.problem)
        
        ## 初始化B&B树的相关信息
        self.nodes = [] ## 生成结点队列
        self.num_nodes,self.next_node_idx = 0,0 ## 当前结点数和下一个结点下标
        self.unprocessed_node_idxs = set([]) ## 未处理结点下标
        self.unprocessed_node_bounds = [] ## 未处理结点的下界信息
        heapq.heapify(self.unprocessed_node_bounds)
        self.unprocessed_node_estimates = []
        heapq.heapify(self.unprocessed_node_estimates)
        
        self.global_LB = MAX ## MIP问题目标值的全局下界
        self.global_obj,self.global_sol = MAX,None ## MIP问题的当前最优解和目标值        

    def reset_branch_type(self,branch_type):
        ## 设置branch方法
        self.brancher = Brancher(self.problem,branch_type)
        
    def reset_select_type(self,select_type):
        ## 设置select方法
        self.selector = Selector(self.problem,select_type)
        
    def add_node(self,node):
        self.nodes.append(node)
        ## 更新未处理结点的信息
        self.unprocessed_node_idxs.add(node.idx)
        heapq.heappush(self.unprocessed_node_bounds,(node.LB,node.idx))
        heapq.heappush(self.unprocessed_node_estimates,(node.estimate,node.idx))
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

        ## 根据LP求解结果更新cost manager
        if node.obj < MAX and node.parent is not None:
            obj_delta = node.obj - node.parent.obj
            var_delta = node.var_delta
            var_idx = node.var_idx
            self.cost_manager.update(var_idx,var_delta,obj_delta)     
            
    def create_node(self,parent_node,var_idx,var_lb,var_ub,var_sol,var_bnd):
        '''
        创建子结点
        '''
        new_node = Node(node_idx=self.next_node_idx,parent=parent_node,cost_manager=self.cost_manager,
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