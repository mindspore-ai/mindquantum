import numpy as np
import cvxpy as cvx



def maxcut_interior_point(edges,iterations=4000,digits=6):
    
    n=int(np.max(edges[:,:2]))+1
    
    e=np.ones(n)
    
    A=np.zeros((n,n))
    
    for i,j,w in edges:
        A[int(i)][int(j)]=w
        A[int(j)][int(i)]=w
    
    
    L=np.diag(np.dot(A,e))-A
    
    
    b=e/4
    
    X=np.zeros((iterations,n,n))
    Z=np.zeros((iterations,n,n))
    y=np.zeros((iterations,n))
    mu=np.zeros(iterations)
    dual=np.zeros(iterations)
    primal=np.zeros(iterations)
    
    
    X[0]=np.diag(b)
    
    y[0]=1.1*np.dot(np.abs(L),e)
    
    Z[0]=np.diag(y[0])-L
    
    dual[0]=np.dot(b,y[0])
    primal[0]=np.trace(np.dot(L,X[0]))
    mu[0]=np.trace(np.dot(Z[0],X[0]))/2/n
    
    
    for i in range(1,iterations):
        
        Z_inv=np.linalg.inv(Z[i-1])
        Z_inv=(Z_inv+Z_inv.T)/2
        
        dy=np.dot(np.linalg.inv(Z_inv*X[i-1]),mu[i-1]*np.diag(Z_inv)-b)
        
        dZ=np.diag(dy)
        
        dX=mu[i-1]*Z_inv-X[i-1]-np.dot(np.dot(Z_inv,dZ),X[i-1])
        dX=(dX+dX.T)/2
        
        
        alphap=1
        
        eX=np.linalg.eigvalsh(X[i-1]+alphap*dX)

        flag_X=(np.round(eX[0],8)>=0)
    
        while flag_X==False:
            alphap=alphap*0.8

            eX=np.linalg.eigvalsh(X[i-1]+alphap*dX)
            flag_X=(np.round(eX[0],8)>=0)
        if alphap<1:
            alphap*=0.95

        alphad=1

        eZ=np.linalg.eigvalsh(Z[i-1]+alphad*dZ)
        
        flag_Z=(np.round(eZ[0],8)>=0)
        
        while flag_Z==False:
            alphad=alphad*0.8

            eZ=np.linalg.eigvalsh(Z[i-1]+alphad*dZ)
            flag_Z=(np.round(eZ[0],8)>=0)
        
        if alphad<1:
            alphad*=0.95
        
        X[i]=X[i-1]+alphap*dX
        y[i]=y[i-1]+alphad*dy
        Z[i]=Z[i-1]+alphad*dZ
        
        mu[i]=np.trace(np.dot(Z[i-1],X[i-1]))/2/n
        
        if alphap+alphad>1.8:
            mu[i]=mu[i]/2
            
        
        dual[i]=np.dot(b,y[i])
        primal[i]=np.trace(np.dot(L,X[i]))
            
        if np.abs(dual[i]-primal[i])<max(1,np.abs(dual[i]))*10**(-digits):
            break
        

    return primal[:i+1],dual[:i+1],X[:i+1]





def GW_algorithm(edges,X_opt,v_ran):
    n=int(np.max(edges[:,:2]))+1
    e=np.ones(n)
    A=np.zeros((n,n))
    for i,j,w in edges:
        A[int(i)][int(j)]=w
        A[int(j)][int(i)]=w
    L=np.diag(np.dot(A,e))-A
    x_v=np.linalg.cholesky(X_opt)
    R=len(v_ran)
    sol=np.zeros((R,n))
    E=np.zeros(R)    
    for i in range(R):
        for j in range(n):
            if np.dot(x_v[j],v_ran[i])>=0:
                sol[i][j]=1
            else:
                sol[i][j]=-1
        E[i]=np.dot(sol[i],np.dot(L,sol[i]))/4
    p=np.where(E==np.max(E))[0][0]
    e=E[p]
    v=sol[p]
    return e,v,E


def GW_maxcut_cvx(graph,v_ran):
    #GW algorithm
    n=len(graph.nodes)
    
    e=np.ones(n)
    
    adjacency=np.zeros((n,n))
    
    for i,j in graph.edges:
        adjacency[i][j]=1
        adjacency[j][i]=1
    
    L=np.diag(np.dot(adjacency,e))-adjacency
    
    
    
    n=graph.number_of_nodes()
    ones_matrix=np.ones((n,n))
    x_matrix=cvx.Variable((n, n), PSD=True)
     
    objective=1/4 * cvx.sum(cvx.multiply(adjacency, ones_matrix - x_matrix))
    constraints = [cvx.diag(x_matrix) == 1]
    problem = cvx.Problem(cvx.Maximize(objective), constraints)
    problem.solve()
    
    x_opti=x_matrix.value
    eigs= np.linalg.eigh(x_opti)[0]
    
    #positive semidefinite
    if min(eigs) < 0:
        x_opti = x_opti + (1.00001 * abs(min(eigs)) * np.identity(n))
    elif min(eigs) == 0:
        x_opti = x_opti + 0.0000001 * np.identity(n)
    
    x_v = np.linalg.cholesky(x_opti)
    
    R=len(v_ran)
    
    sol=np.zeros((R,n))

    E=np.zeros(R)    

    for i in range(R):
        for j in range(n):
            if np.dot(x_v[j],v_ran[i])>=0:
                sol[i][j]=1
            else:
                sol[i][j]=-1
        
        
        E[i]=np.dot(sol[i],np.dot(L,sol[i]))/4
    
    
    
    return np.max(E)





