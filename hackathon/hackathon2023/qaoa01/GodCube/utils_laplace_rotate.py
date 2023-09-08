import numpy as np

def rotate_to_z0(G,L,z0):
    #z0 = np.sign(np.sign(eigvec[:,-1])+0.01)
    n = len(z0)
    D = np.diag(G.dot(z0)*z0)
    Lp = D-G
    eigval, eigvec = np.linalg.eigh(Lp)
    #angle = np.zeros(n)
    #cut   = np.zeros(n)
    #for ii in range(n):
    #    angle[ii] = np.dot(eigvec[:,ii], np.sign(eigvec[:,ii]))/np.sqrt(n)
    #    cut[ii] = np.einsum('i,ij,j->',
    #                    np.sign(np.sign(eigvec[:,ii])+0.01), L,
    #                    np.sign(np.sign(eigvec[:,ii])+0.01))/4
        #if (cut[ii]==1): print(np.sign(np.sign(eigvec[:,ii])+0.01))
    #print("angle:")
    #print(angle[-18:])
    #print("cut:")
    #print(cut)
    #max_id = np.argmax(cut)

    #ideal = np.rint(eigval*len(eigval)/4.)
    #print("ideal:")
    #print(ideal - ideal[0])
    #print(np.rint(eigval*len(eigval)/4.) +313)
    #print(1./(eigval*len(eigval)/4.+1e-4)*cut)

    return eigval,eigvec#,max_id


def find_subspace(eigval,eigvec,N_=15,eps=1e-8):
    subspace = np.where(eigval>eps)
    #print(subspace)
    eigvec_sub = eigvec[:,subspace[0]]
    #print(eigvec_sub.shape)
    eigvec_sub_proj = np.sqrt(np.sum(eigvec_sub**2,axis=1))
    topk_index = np.argpartition(eigvec_sub_proj.copy(),-N_)[-N_:]
    #print(topk_index)
    #print(eigvec_sub_proj[topk_index])
    return topk_index


def get_eigvec_sub_proj(eigval,eigvec,N_=15,eps=1e-8):
    subspace = np.where(eigval>eps)
    eigvec_sub = eigvec[:,subspace[0]]
    eigvec_sub_proj = np.sqrt(np.sum(eigvec_sub**2,axis=1))
    return eigvec_sub_proj



