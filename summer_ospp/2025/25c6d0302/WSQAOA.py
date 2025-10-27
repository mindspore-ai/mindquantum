import numpy as np


def construct_graph(edges,j,k=-1):
    """从一个完整的图数据中提取出围绕一个或两个核心节点的子图。

    该函数不仅提取出相关的边和节点，还会对子图中的节点进行重新标记，
    使其从0开始连续编号，方便后续的局部计算。

    :param edges: 完整图的边列表，一个形状为 (m, 3) 的数组，
                  每一行格式为 [顶点1, 顶点2, 权重]。
    :type edges: np.ndarray
    :param j: 第一个核心节点的原始标签。
    :type j: int
    :param k: (可选) 第二个核心节点的原始标签。如果为-1，则只提取节点j的邻域。
              默认为 -1。
    :type k: int, optional

    :return: 根据k的值，返回包含子图信息的元组。
             - **如果k=-1 (单节点模式)**:
               返回 (edges_new, j_new, nodes_j, edges_j)
               - ``edges_new`` (np.ndarray): 节点被重新标记后的子图边列表。
               - ``j_new`` (int): 核心节点j在子图中的新标签。
               - ``nodes_j`` (np.ndarray): 子图中节点的原始标签列表，其索引对应新标签。
               - ``edges_j`` (np.ndarray): 未经重新标记的原始子图边列表。
             - **如果k!=-1 (双节点模式)**:
               返回 (edges_new, j_new, k_new, nodes_jk, edges_jk)
               - ``edges_new`` (np.ndarray): 节点被重新标记后的子图边列表。
               - ``j_new`` (int): 核心节点j在子图中的新标签。
               - ``k_new`` (int): 核心节点k在子图中的新标签。
               - ``nodes_jk`` (np.ndarray): 子图中节点的原始标签列表。
               - ``edges_jk`` (np.ndarray): 未经重新标记的原始子图边列表。
    :rtype: tuple
    """ 
    if k==-1:
        edges_j=[]
        for v1,v2,w in edges:
            if j==v1 or j==v2:
                edges_j.append([v1,v2,w])
        
        edges_j=np.array(edges_j)
        nodes_j=np.unique(edges_j[:,:2])
            
        edges_new=np.array(edges_j)
        
        n_new=len(nodes_j)
    
        for i in range(n_new):
            edges_new[np.where(edges_new[:,:2]==nodes_j[i])]=i
            
        j_new=np.where(nodes_j==j)[0][0]

        return edges_new,j_new ,nodes_j,edges_j

    else:
        edges_jk=[]
        for v1,v2,w in edges:
            if j==v1 or j==v2 or k==v1 or k==v2:
                edges_jk.append([v1,v2,w])
        
        edges_jk=np.array(edges_jk)
        nodes_jk=np.unique(edges_jk[:,:2])
        
        edges_new=np.array(edges_jk)

        
        n_new=len(nodes_jk)
    
        for i in range(n_new):
            edges_new[np.where(edges_new[:,:2]==nodes_jk[i])]=i
            
        j_new=np.where(nodes_jk==j)[0][0]
        k_new=np.where(nodes_jk==k)[0][0]
        
        return edges_new,j_new,k_new,nodes_jk,edges_jk
    
def node_connection(j,k,graph_info):
    """获取一个或两个核心节点的邻居节点，并按特定规则排序。

    该函数用于为后续的能量计算准备结构化的节点列表。

    :param j: 第一个核心节点的原始标签。
    :type j: int
    :param k: (可选) 第二个核心节点的原始标签。如果为-1，则只处理节点j。
    :type k: int
    :param graph_info: 完整图的边列表。
    :type graph_info: np.ndarray

    :return: 一个包含结构化节点列表的NumPy数组。
             - **如果k=-1 (单节点模式)**:
               返回一个一维数组，格式为 ``[j, j的邻居1, j的邻居2, ...]``。
             - **如果k!=-1 (双节点模式)**:
               返回一个二维数组，形状为(2, N)。
               - 第一行是与j相关的节点列表，格式为 ``[j, j和k的共同邻居..., j的独立邻居...]``。
               - 第二行是与k相关的节点列表，格式为 ``[k, j和k的共同邻居..., k的独立邻居...]``。
               共同邻居在两行中的顺序是一致的。
    :rtype: np.ndarray
    """
    if k!=-1:  
  
        
        aj,bj,j_nodes,cj=construct_graph(graph_info,j,-1)
        
        ak,bk,k_nodes,ck=construct_graph(graph_info,k,-1)

        jc=np.setdiff1d(j_nodes,[j,k])
        kc=np.setdiff1d(k_nodes,[j,k])
        
        s=1
    
        jc2=-1*np.ones(len(jc)+1)
        kc2=-1*np.ones(len(kc)+1)
        
        jc2[0]=j
        kc2[0]=k
        
        
        for i1 in range(len(jc)):
            
            for i2 in range(len(kc)):
                
                if jc[i1]==kc[i2]:
                    
                    jc2[s]=jc[i1]
                    kc2[s]=kc[i2]
                    s+=1
                
        jc2[s:]=np.setdiff1d(jc,jc2[:s])
        kc2[s:]=np.setdiff1d(kc,kc2[:s])
    
  
        nodes=np.array([jc2,kc2],dtype=np.int64)
    else:
        aj,bj,j_nodes,cj=construct_graph(graph_info,j,-1)
        jc=np.setdiff1d(j_nodes,[j]).astype(np.int64)
        nodes=np.concatenate(([j],jc),axis=0)
    
    return nodes 



def coeff_ZYX(alpha,beta1,j):
    
    CZj=1-2*np.cos(alpha[j])**2*np.sin(beta1)**2
    CYj=np.cos(alpha[j])*np.sin(2*beta1)    
    CXj=-np.sin(2*alpha[j])*np.sin(beta1)**2
    
    c=np.array([CZj,CYj,CXj])
    
    return c


def EZZ(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]

    e=np.cos(delta[j])*np.cos(delta[k])*h[j]*h[k]
    
    return e

def EZY(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]
    
    
    e=np.cos(gamma1)**2*np.sin(gamma1)*np.cos(delta[j])*np.sin(delta[k])*np.cos(delta[k2])*h[j]*h[k]*h[k2]

    e+=np.cos(gamma1)**2*np.sin(gamma1)*np.cos(delta[j])*np.sin(delta[k])*np.cos(delta[k1])*h[j]*h[k]*h[k1]

    e+=np.cos(gamma1)**2*np.sin(gamma1)*np.sin(delta[k])*h[k]

    e-=np.sin(gamma1)**3*np.sin(delta[k])*np.cos(delta[k1])*np.cos(delta[k2])*h[k]*h[k1]*h[k2]

    return e

def EZX(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]
    
    e=np.cos(delta[j])*np.sin(delta[k])*np.cos(gamma1)**3*h[j]*h[k]
    
    e-=np.cos(delta[j])*np.sin(delta[k])*np.cos(delta[k1])*np.cos(delta[k2])*np.cos(gamma1)*np.sin(gamma1)**2*h[j]*h[k]*h[k1]*h[k2]
    
    e-=np.sin(delta[k])*np.cos(delta[k1])*np.cos(gamma1)*np.sin(gamma1)**2*h[k]*h[k1]
    
    e-=np.sin(delta[k])*np.cos(delta[k2])*np.cos(gamma1)*np.sin(gamma1)**2*h[k]*h[k2]
    
    return e
    
def EYY(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]
    
    e=np.cos(delta[j2])*np.cos(delta[k1])*h[j2]*h[k1]

    if j2==k2:
    
        e=e+1
    else:
        e=e+np.cos(delta[j2])*np.cos(delta[k2])*h[j2]*h[k2]

    e=e+np.cos(delta[j1])*np.cos(delta[k2])*h[j1]*h[k2]

    if j1==k1:
    
        e=e+1
    else:
        e=e+np.cos(delta[j1])*np.cos(delta[k1])*h[j1]*h[k1]
    
    e=np.cos(gamma1)**2*np.sin(gamma1)**2*np.sin(delta[j])*np.sin(delta[k])*h[j]*h[k]*e
    

    return e


def EXX(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]
    
    
    e=np.cos(gamma1)**4

    e=e-np.cos(gamma1)**2*np.sin(gamma1)**2*np.cos(delta[k1])*np.cos(delta[k2])*h[k1]*h[k2]
    
    e=e-np.cos(gamma1)**2*np.sin(gamma1)**2*np.cos(delta[j1])*np.cos(delta[j2])*h[j1]*h[j2]
    
    if j1==k1 and j2!=k2:
        e=e+np.sin(gamma1)**4*np.cos(delta[j2])*np.cos(delta[k2])*h[j2]*h[k2]
    
    elif j1==k1 and j2==k2:
        
        e=e+np.sin(gamma1)**4
    else:
        
        e=e+np.sin(gamma1)**4*np.cos(delta[j1])*np.cos(delta[j2])*np.cos(delta[k1])*np.cos(delta[k2])*h[j1]*h[j2]*h[k1]*h[k2]
        
    e=e*np.sin(delta[j])*np.sin(delta[k])*h[j]*h[k]
    
    return e

def EYX(delta,gamma1,nodes,h):
    
    j,k=nodes[:,0]
    
    j1,j2=nodes[0][1:]
    
    k1,k2=nodes[1][1:]

    e=np.cos(gamma1)**2*np.cos(delta[j2])*h[j2]
    e+=np.cos(gamma1)**2*np.cos(delta[j1])*h[j1]
    
    if j1==k1:
        e=e-np.sin(gamma1)**2*np.cos(delta[k2])*h[k2]
    else:
        e=e-np.sin(gamma1)**2*np.cos(delta[j1])*np.cos(delta[k1])*np.cos(delta[k2])*h[j1]*h[k1]*h[k2]
    
    if j2==k2:
        e=e-np.sin(gamma1)**2*np.cos(delta[k1])*h[k1]
    else:
        e=e-np.sin(gamma1)**2*np.cos(delta[j2])*np.cos(delta[k1])*np.cos(delta[k2])*h[j2]*h[k1]*h[k2]
    
   
    e=e*np.cos(gamma1)*np.sin(gamma1)*np.sin(delta[j])*np.sin(delta[k])*h[j]*h[k]
    
    return e


def EZ(delta,gamma1,nodes,h):
    
    j=nodes[0]
    
    e=np.cos(delta[j])*h[j]
    
    return e

def EY(delta,gamma1,nodes,h):
    
    j,k,j1,j2=nodes
    
    e=np.cos(gamma1)**2*np.cos(delta[j2])*h[j2]
    
    e+=np.cos(gamma1)**2*np.cos(delta[j1])*h[j1]

    e+=np.cos(gamma1)**2*np.cos(delta[k])*h[k]
    
    e=e-np.sin(gamma1)**2*np.cos(delta[k])*np.cos(delta[j1])*np.cos(delta[j2])*h[k]*h[j1]*h[j2]

    e=e*np.sin(gamma1)*np.sin(delta[j])*h[j]
    
    return e

def EX(delta,gamma1,nodes,h):
    
    j,k,j1,j2=nodes

    e=np.cos(gamma1)**2
    
    e=e-np.sin(gamma1)**2*np.cos(delta[j1])*np.cos(delta[j2])*h[j1]*h[j2]
    
    e=e-np.sin(gamma1)**2*np.cos(delta[j1])*np.cos(delta[k])*h[j1]*h[k]
    
    e=e-np.sin(gamma1)**2*np.cos(delta[k])*np.cos(delta[j2])*h[k]*h[j2]
    
    e=e*np.cos(gamma1)*np.sin(delta[j])*h[j]
    
    return e


def energy_jk(nodes,alpha,delta,h,para):
    
    gamma1=para[0]
    beta1=para[1]
    
    j,k=nodes[:,0]
    
    nodesR=np.array([nodes[1],nodes[0]])
    
    cj=coeff_ZYX(alpha,beta1,j)
    ck=coeff_ZYX(alpha,beta1,k)
    
    
    ezz=EZZ(delta,gamma1,nodes,h)
    eyy=EYY(delta,gamma1,nodes,h)
    exx=EXX(delta,gamma1,nodes,h)
    
    ezy=EZY(delta,gamma1,nodes,h)
    eyz=EZY(delta,gamma1,nodesR,h)
    
    ezx=EZX(delta,gamma1,nodes,h)
    exz=EZX(delta,gamma1,nodesR,h)
    
    eyx=EYX(delta,gamma1,nodes,h)
    exy=EYX(delta,gamma1,nodesR,h)
    

    e=cj[0]*ck[0]*ezz+cj[1]*ck[1]*eyy+cj[2]*ck[2]*exx
    
    e=e+cj[0]*ck[1]*ezy+cj[1]*ck[0]*eyz

    e=e+cj[0]*ck[2]*ezx+cj[2]*ck[0]*exz

    e=e+cj[1]*ck[2]*eyx+cj[2]*ck[1]*exy

    return e


def energy_j(nodes,alpha,delta,h,para):
    
    gamma1=para[0]
    beta1=para[1]
    
    j=nodes[0]
    
    
    cj=coeff_ZYX(alpha,beta1,j)

    ez=EZ(delta,gamma1,nodes,h)

    ey=EY(delta,gamma1,nodes,h)
    ex=EX(delta,gamma1,nodes,h)

    
    e=cj[0]*ez+cj[1]*ey+cj[2]*ex
    

    return e


class QAOA:
    """QAOA类，用于封装和模拟多种QAOA变体的核心计算逻辑。
    这个类本身不直接执行优化循环，而是扮演一个“模拟量子处理器”的角色。
    它接收QAOA参数，并利用解析公式高效地计算出能量期望值。
    此外，它还包含了更新自适应偏置场和评估偏置解能量的功能。
    """
    def __init__(self,n,qaoa_variant,pro_solve):
        """QAOA类的构造函数，用于初始化一个QAOA模拟器实例。
        :param n: 图的顶点数（或量子比特数）。
        :type n: int
        :param qaoa_variant: 一个包含QAOA变体具体配置的字典。
        :type qaoa_variant: dict
        :param pro_solve: 图的边列表，一个形状为 (m, 3) 的数组，
                          每一行格式为 [顶点1, 顶点2, 权重]。
        :type pro_solve: np.ndarray
        """
        self.n=n
        self.bias=qaoa_variant['solution_WS'] #初始的偏置场编码经典算法的解 
        self.edges=pro_solve #MaxCut问题的图结构
        self.variant=qaoa_variant
        self.h=np.ones(self.n)       
        self.A=np.zeros((self.n,self.n)) 
        #构建图的邻接矩阵      
        for i,j,w in self.edges:
             self.A[int(i)][int(j)]=w
             self.A[int(j)][int(i)]=w
    def get_expectation(self,para):
        """计算给定QAOA参数下的代价函数期望值（能量）。
        这是模拟的核心，它根据不同的QAOA变体类型准备初始态，并计算
        最终的能量期望值，返回其负值作为优化器的损失函数。
        :param para: QAOA的角度参数，通常是一个包含 [gamma, beta] 的一维数组。
        :type para: np.ndarray
        :return: 代价函数期望值的负值 (-E)，用于梯度下降优化。
        :rtype: float
        """
        if self.variant['type']=='ab':
            self.alpha=np.arcsin(self.bias/np.sqrt(1+self.bias**2)) # bias = +1,-1 ----> alpha = pi/2, -\pi/2
            self.delta=self.alpha-np.pi/2 #依据经典解，要在第一层做bias = +1,-1 ----> delta = pi/2 - pi/2, -pi/2-pi/2
        elif self.variant['type']=='warm-start':
            epsilon_ws=self.variant['epsilon_ws']
            self.alpha=np.zeros(self.n)
            for i in range(self.n):
                if (1+self.bias[i])/2<=epsilon_ws:
                    self.alpha[i]=2*np.arcsin(np.sqrt(epsilon_ws))-np.pi/2
                elif (1+self.bias[i])/2>=1-epsilon_ws:
                    self.alpha[i]=2*np.arcsin(np.sqrt(1-epsilon_ws))-np.pi/2
                else:
                    self.alpha[i]=2*np.arcsin(np.sqrt((1+self.bias[i])/2))-np.pi/2
            self.alpha=np.pi-self.alpha
            self.delta=np.pi/2-self.alpha
        elif self.variant['type']=='warmest':
            self.alpha=self.bias.copy()
            self.delta=self.alpha-np.pi/2
        elif self.variant['type']=='standard':
            self.alpha=np.zeros(self.n)
            self.delta=self.alpha-np.pi/2
        else:
            0
        E=0
        for j,k,w in self.edges:
            nodes=node_connection(int(j),int(k), self.edges)
            E+=(1-energy_jk(nodes,self.alpha,self.delta,self.h,para))/2
        return -E
    
    def update_bias(self,mh,vh,alphah,para):
        """使用Adam规则更新自适应偏置场。此方法仅用于 'ab' 变体。
        :param mh: 偏置场的一阶矩估计（动量）向量。
        :type mh: np.ndarray
        :param vh: 偏置场的二阶矩估计向量。
        :type vh: np.ndarray
        :param alphah: 用于更新偏置场的学习率。
        :type alphah: float
        :param para: 当前的QAOA角度参数 [gamma, beta]。
        :type para: np.ndarray
        :return: 更新后的一阶矩和二阶矩向量 (mh, vh)。
        :rtype: tuple
        """
        beta1=0.9
        beta2=0.999
        epsilon=10**(-8)   
        bias_change=np.zeros(self.n)
        for j in range(self.n):
            nodes=node_connection(int(j),-1, self.edges)
            ezj=energy_j(nodes,self.alpha,self.delta,self.h,para)
            bias_change[j]=self.bias[j]+ezj/np.sqrt(3)
        mh=beta1*mh+(1-beta1)*bias_change
        vh=beta2*vh+(1-beta2)*bias_change**2
        self.bias=self.bias-alphah*mh/(np.sqrt(vh)+epsilon)
        return mh,vh 



    def get_bias_energy(self):
        """计算当前偏置场所代表的经典解的能量（割边权重）。

        此方法将浮点数的偏置向量转换为一个 +/-1 的经典解，并计算
        其在MaxCut问题下的得分。

        :return: 经典解的割边权重。
        :rtype: float
        """
        e=np.ones(self.n)
        L=np.diag(np.dot(self.A,e))-self.A
        E=np.dot(np.sign(self.bias),np.dot(L,np.sign(self.bias)))/4
        return E


