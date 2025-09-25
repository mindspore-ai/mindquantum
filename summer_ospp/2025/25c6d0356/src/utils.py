import numpy as np
from scipy.special import binom


# Functions for permutation matrices
def permutation_idx2mat(col_ind):
    col_ind = col_ind.flatten()
    n = len(col_ind)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, col_ind[i]] = 1
    return P

def t(n):
    T = []
    for i in range(n):
        for j in range(i + 1, n):
            Ti = np.eye(n)
            Ti[i, i] = Ti[j, j] = 0
            Ti[i, j] = Ti[j, i] = 1
            T.append(Ti.reshape(n*n, 1))
    T = np.hstack(T)
    return T


def eval_p(T, x, P_rec=None):
    n = int(np.sqrt(T.shape[0]))
    n_trans = T.shape[1]
    I = np.eye(n)
    P_rec = I if P_rec is None else P_rec

    P = P_rec
    dP = np.zeros_like(T)
    for i in range(n_trans):
        P = P @ (x[i, 0] * (T[:, i].reshape(n, n) - I) + I)

        # Eval dP
        dp = P_rec
        for j in range(i):
            dp = dp @ (x[j, 0] * (T[:, j].reshape(n, n) - I) + I)
        dp = dp @ (T[:, i] - I.flatten()).reshape(n, n)
        for j in range(i+1, n_trans):
            dp = dp @ (x[j, 0] * (T[:, j].reshape(n, n) - I) + I)
        dP[:, i] = dp.flatten()

    P = P.reshape(n*n, 1)
    return P, dP


# Functions for 2D rotation matrices
def r_2d(x):
    return np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])


def eval_r_2d(x):
    I = np.eye(2, dtype='float64').flatten()[:, None]
    M = np.array([[0, -1], [1, 0]], dtype='float64').flatten()[:, None]
    R = np.cos(x) * I + np.sin(x) * M
    dR = -np.sin(x) * I + np.cos(x) * M
    return R, dR


# Functions for 3D rotation matrices
def M(x):
    x = x.flatten()
    result = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype='float64')

    d_result = np.zeros((3, 3, 3))
    d_result[0, :, :] = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype='float64')
    d_result[1, :, :] = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype='float64')
    d_result[2, :, :] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype='float64')
    return result, d_result

def g(x):
    m, n = x.shape
    x = x.flatten()
    theta = np.linalg.norm(x)
    if theta != 0.:
        result = np.sin(theta) / theta

        x = x.reshape(3, 1)
        d_result = x * (theta * np.cos(theta) - np.sin(theta)) / theta**3
    else:
        result = 1
        d_result = np.zeros((m, n))
    return result, d_result

def h(x):
    m, n = x.shape
    x = x.flatten()
    theta = np.linalg.norm(x)
    if theta != 0:
        result = (1 - np.cos(theta)) / theta**2

        x = x.reshape(3, 1)
        d_result = x * (theta * np.sin(theta) - 2 * (1 - np.cos(theta))) / theta**4
    else:
        result = 1/2
        d_result = np.zeros((m, n))
    return result, d_result


def r_3d(x):
    g_fun, g_grad = g(x)
    h_fun, h_grad = h(x)
    M_fun, M_grad = M(x)

    I = np.eye(3)
    result = I + g_fun * M_fun + h_fun * M_fun @ M_fun

    p = 3
    d_result = g_grad.reshape(p, 1, 1) * np.tile(M_fun, (p, 1, 1)) \
                + g_fun * M_grad \
                + h_grad.reshape(p, 1, 1) * np.tile(M_fun @ M_fun, (p, 1, 1)) \
                + h_fun * (M_grad @ np.tile(M_fun, (p, 1, 1)) + np.tile(M_fun, (p, 1, 1)) @ M_grad)
    return result, d_result


def eval_r_3d(x):
    R, dR = r_3d(x)

    R = R.flatten()[:, None]
    dR = np.hstack([dr.flatten()[:, None] for dr in dR])
    return R, dR


# Random 2D points and fancy Bézier curves
# (https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib)
bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_2d_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec >= 200:
        points = (a*scale).T
        points -= np.mean(points, axis=1, keepdims=True)
        return points
    else:
        return get_random_2d_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

# random 3D shapes
def get_random_3d_points(dataset_name, N=30):
    if dataset_name == 'synthetic':
        # Fully randomly genarated points
        var = 10
        points = var * np.random.randn(3, N)

    if dataset_name == 'cylinder':
        # Generate synthetic cylinder
        if N==50:
            theta = np.linspace(0, 2 * np.pi, 25)  # For timing
            w = np.linspace(-0.25, 0.25, 2)
        else:
            theta = np.linspace(0, 2 * np.pi, N)  # For timing
            w = np.linspace(-0.25, 0.25, 1)
        w, theta = np.meshgrid(w, theta)
        phi = 0.5 * theta
        # radius in x-y plane
        r = 1 + w * np.cos(phi)
        x = np.ravel(r * np.cos(theta))
        y = np.ravel(r * np.sin(theta))
        z = np.ravel(w * np.sin(phi))
        points = np.vstack((x, y, z))
        points = points[:, np.linspace(0, points.shape[1] - 1, N).astype('int32')]

    if dataset_name == 'spiral':
        # Generate synthetic spiral
        z = np.linspace(0, 1, N) / 2
        x = z * np.sin(25 * z)
        y = z * np.cos(25 * z)
        points = np.vstack((x, y, z))
        points = points[:, np.linspace(0, points.shape[1] - 1, N).astype('int32')]
    return points
