import os
import pickle
import numpy as np
import matplotlib.path as mpath
import scipy.sparse as sp
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, CubicSpline, interp1d, PchipInterpolator, RegularGridInterpolator
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from spektral.data import Dataset, Graph
from spektral.data.loaders import DisjointLoader, SingleLoader
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.regularizers import l2
from spektral.layers import MessagePassing, GCNConv, GATConv, ECCConv
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import calfem.geometry as cfg
import calfem.mesh as cfm


def PlotMesh(nodes, boundaryNodes, elements,lineWidth=1):
    plt.figure(figsize=(10,10))
    plt.scatter(nodes[:,0],nodes[:,1])
    if elements.shape[1] == 4:
        for el in elements:
            plt.plot(nodes[[el[0],el[1],el[3],el[2],el[0]],0], nodes[[el[0],el[1],el[3],el[2],el[0]],1], 'k',linewidth=lineWidth)
    elif elements.shape[1] == 3:
        for el in elements:
            plt.plot(nodes[[el[0],el[1],el[2],el[0]],0], nodes[[el[0],el[1],el[2],el[0]],1], 'k',linewidth=lineWidth)
    plt.plot(nodes[boundaryNodes,0],nodes[boundaryNodes,1],'r',linewidth=lineWidth)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()

def GetK_el_triang(A,nodes):
    r = int(A.shape[0]/2)
    # a = np.roll(nodes[:,0],1)*np.roll(nodes[:,1],2) - np.roll(nodes[:,0],2)*np.roll(nodes[:,1],1)
    b = np.roll(nodes[:,1],1) - np.roll(nodes[:,1],2)
    c = np.roll(nodes[:,0],2) - np.roll(nodes[:,0],1)
    Area = np.abs(np.dot(nodes[:,0],b))/2
    B = np.concatenate([
        np.concatenate([b[i]*np.eye(r) for i in range(3)],1),
        np.concatenate([c[i]*np.eye(r) for i in range(3)],1)
    ],0)/(2*Area)
    return np.dot(np.dot(B.T,A),B)*Area


def GetK(nodes_els, A):
    r = int(A.shape[0]/2)
    b = (np.roll(nodes_els[:,:,1],1,axis=1) - np.roll(nodes_els[:,:,1],2,axis=1)).reshape(-1,3,1)
    c = (np.roll(nodes_els[:,:,0],2,axis=1) - np.roll(nodes_els[:,:,0],1,axis=1)).reshape(-1,3,1)
    Area = np.abs(np.matmul(nodes_els[:,:,0].reshape(-1,1,3),b))/2
    B = np.concatenate([
        np.concatenate([b[:,i:i+1]*np.eye(r).reshape(1,r,r) for i in range(3)],-1),
        np.concatenate([c[:,i:i+1]*np.eye(r).reshape(1,r,r) for i in range(3)],-1)
    ],-2)/(2*Area)
    B_T = np.transpose(B,(0,2,1))
    return np.matmul(np.matmul(B_T,A),B)*Area

def SolveFEM(nodes, elements, boundaryNodes, BCfunc, alpha, internalNodes, r, A, A_nl=False, l=None):
    if l is None:
        l = np.zeros((nodes.shape[0], r))
    if not A_nl:
        A_l = A

    # Assemble the global stiffness matrix
    K = np.zeros((nodes.shape[0]*r, nodes.shape[0]*r))
    for el in elements:
        el_idx = [[r*k+j for j in range(r)] for k in el]
        el_idx = np.concatenate(el_idx)
        nodes_el = tf.gather(nodes, indices=el)
        X_idx,Y_idx = np.meshgrid(el_idx,el_idx)
        if A_nl:
            A_l = A(l[el_idx])
        # print(A_l)
        K_el = GetK_el_triang(A_l,nodes_el)
        K[Y_idx,X_idx] += K_el

    # Apply Dirichlet BC
    l_BC = BCfunc(alpha*2*np.pi)
    bc_idx = [[r*i+j for j in range(r)] for i in boundaryNodes]
    bc_idx = np.concatenate(bc_idx)
    internal_idx = [[r*i+j for j in range(r)] for i in internalNodes]
    internal_idx = np.concatenate(internal_idx)

    f = - (K[:,bc_idx] @ l_BC.flatten().reshape(-1,1))

    K_BC = K[internal_idx,:][:,internal_idx]
    f = f[internal_idx]

    # Solve the system
    l_internal = np.linalg.solve(K_BC, f)
    n_CDOF = int(l_internal.shape[0]/r)
    l_internal = l_internal.reshape(n_CDOF, r)

    l[internalNodes,:] = l_internal
    l[boundaryNodes,:] = l_BC.reshape(-1,r)
    return l

def SolveFEM_itt(nodes, elements, boundaryNodes, BCfunc, alpha, internalNodes, r, A, tol=1e-8,show_err=False, max_iter=10):

    l_prev = SolveFEM(nodes, elements, boundaryNodes, BCfunc, alpha, internalNodes, r, A, A_nl=True, l=None)

    err = 1
    i = 1
    while err>tol and i<max_iter:
        l = SolveFEM(nodes, elements, boundaryNodes, BCfunc, alpha, internalNodes, r, A, A_nl=True, l=l_prev.copy())
        err = ((l-l_prev)**2).mean()
        if show_err:
            print(f'Iteration {i}; err = {err}')
        i+=1
        l_prev = l.copy()

    return l, i

def PlotFEMsolution(nodes, elements,l):
    if elements.shape[1] == 4:
        # Convert quadrlateral mesh to triangular mesh
        elements = np.concatenate([elements[:,:3],elements[:,1:]],0)

    # Create a Triangulation object
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Plotting
    r = l.shape[1]
    plt.figure(figsize=(6*r,5))
    for i in range(r):
        plt.subplot(1,r,i+1)
        plt.tricontourf(triangulation, l[:,i],10)
        plt.colorbar()
        # plt.scatter(nodes[:,0],nodes[:,1],s=1,c='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')
    return 

def PlotFEMsoluttionDict(solution):
    nodes = np.array(solution['nodes'])
    elements = np.array(solution['elements'])
    l = np.array(solution['l'])
    PlotFEMsolution(nodes, elements,l)

def GenerateGeometry(p):
    g = cfg.Geometry()
    for i in range(p.shape[0]):
        g.point(list(p[i]))
    g.spline(list(range(p.shape[0]))+[0])
    g.surface([0])
    return g

def MeshSurface(g,elSize):
    mesh = cfm.GmshMesh(g)
    mesh.elType = 2       # Degrees of freedom per node.
    mesh.dofsPerNode = 1     # Factor that changes element sizes.
    mesh.elSizeFactor = elSize # Element size Factor
    nodes, edof, dofs, bdofs, elementmarkers = mesh.create()

    elements = edof-1
    boundaryNodes = np.array(bdofs[0])-1
    internalNodes = np.setdiff1d(np.arange(nodes.shape[0]), boundaryNodes)

    alpha = GetDistAlongBoundary(nodes,boundaryNodes)
    return nodes, elements, boundaryNodes, internalNodes, alpha

def GetRandomBCfuncAlpha(n_order=3, r=1):
    p = np.random.randn(2,n_order,r)
    BCfunc_unscaled = lambda alpha: np.array([[p[0,j,i]*np.cos((j+1)*alpha) + p[1,j,i]*np.sin((j+1)*alpha) for i in range(r)] for j in range(n_order)]).T.sum(axis=-1)
    alpha = np.linspace(0,2*np.pi,100)
    vals = BCfunc_unscaled(alpha)
    vals_max = np.max(vals)
    vals_min = np.min(vals)
    BCfunc = lambda alpha: (BCfunc_unscaled(alpha)-vals_min)/(vals_max-vals_min)*2-1
    return BCfunc

def SortBoundaryNodes(boundaryNodes,nodes):
    boundaryNodesSorted = [boundaryNodes[0]]
    boundaryNodesNotSorted = np.delete(boundaryNodes,0)
    for i in range(1,len(boundaryNodes)):
        idx = ((nodes[boundaryNodes[i]]-nodes[boundaryNodesNotSorted])**2).sum(axis=1).argmin()
        boundaryNodesSorted.append(boundaryNodesNotSorted[idx])
        boundaryNodesNotSorted = np.delete(boundaryNodesNotSorted,idx)
    return np.array(boundaryNodesSorted)

def GetDistAlongBoundary(nodes,boundaryNodes):
    ds = [0]+[np.sqrt(((nodes[boundaryNodes[i-1]]-nodes[boundaryNodes[i]])**2).sum()) for i in range(boundaryNodes.shape[0])]
    s = np.cumsum(ds)
    s = s[:-1]/s[-1]
    return s

def GetRandomFixedPoints(n_min = 4,n_max = 10):
    n_points = np.random.randint(n_min,n_max)
    i = 0
    while True:
        angles = np.random.lognormal(0,1,n_points)
        angles = angles/np.sum(angles)*2*np.pi
        i += 1
        if np.all(angles<np.pi) and np.all(angles>np.pi/6):
            break
    # print(i)
    angles = np.cumsum(angles)-angles[0]
    # r = np.abs(np.random.randn(n_points)+1)+0.5
    # r[r>3] = 3
    r = np.random.uniform(0.5,1.5,n_points)
    points = np.zeros((n_points,2))
    for i in range(angles.shape[0]):
        points[i,:] = r[i]*np.array([np.cos(angles[i]),np.sin(angles[i])])
    return points

def GenerateRandomSolution(n_min = 4, n_max = 10, elSize = 0.07, n_order = 3):
    points = GetRandomFixedPoints(n_min,n_max)
    g = GenerateGeometry(points)
    nodes, elements, boundaryNodes, internalNodes, alpha = MeshSurface(g,elSize)
    boundaryNodes = SortBoundaryNodes(boundaryNodes,nodes)
    A = lambda u: np.concatenate([
    np.concatenate([[[10*np.max([u.mean(),0])**2+0.5]],[[0]]],axis=1),
    np.concatenate([[[0]],[[10*np.max([u.mean(),0])**2+0.5]]],axis=1)
    ],axis=0)
    r = 1
    BCfunc = GetRandomBCfuncAlpha(n_order,r=r)
    l,_ = SolveFEM_itt(nodes, elements, boundaryNodes, BCfunc, alpha, internalNodes, r, A, tol=1e-8,show_err=True, max_iter=20)
    mesh = {'nodes':nodes,'elements':elements,'boundaryNodes':boundaryNodes,'internalNodes':internalNodes,'points':points, 'alpha':alpha}
    return l, mesh, points


def SolveFEM(nodes, elements, boundaryNodes, l_BC, internalNodes, r, A_l):
    l = np.zeros((nodes.shape[0], r))

    # Assemble the global stiffness matrix
    K = np.zeros((nodes.shape[0]*r, nodes.shape[0]*r))
    for el in elements:
        el_idx = [[r*k+j for j in range(r)] for k in el]
        el_idx = np.concatenate(el_idx)
        nodes_el = tf.gather(nodes, indices=el)
        X_idx,Y_idx = np.meshgrid(el_idx,el_idx)
        # print(A_l)
        K_el = GetK_el_triang(A_l,nodes_el)
        K[Y_idx,X_idx] += K_el

    # Apply Dirichlet BC
    bc_idx = [[r*i+j for j in range(r)] for i in boundaryNodes]
    bc_idx = np.concatenate(bc_idx)
    internal_idx = [[r*i+j for j in range(r)] for i in internalNodes]
    internal_idx = np.concatenate(internal_idx)

    f = - (K[:,bc_idx] @ l_BC.flatten().reshape(-1,1))

    K_BC = K[internal_idx,:][:,internal_idx]
    f = f[internal_idx]

    # Solve the system
    l_internal = np.linalg.solve(K_BC, f)
    n_CDOF = int(l_internal.shape[0]/r)
    l_internal = l_internal.reshape(n_CDOF, r)

    l[internalNodes,:] = l_internal
    l[boundaryNodes,:] = l_BC.reshape(-1,r)
    return l


def read_am_file(file_path,slices=None,sigma=0):    
    # Search for the binary data section marker "@1"
    marker = b"@1"
    with open(file_path, 'rb') as file:
        content = file.read()
        start = content.find(marker) + len(marker)
        
        # Assuming there are two newline characters after the "@1" marker
        # Adjust if necessary based on the actual file format
        start += 32
        
        # The total number of data points is the product of the lattice dimensions
        # Each point has 2 float components
        num_points = 512 * 512 * 1001 * 2
        
        # Set the file pointer to the start of the binary data and read it
        file.seek(start)
        data = np.fromfile(file, dtype=np.float32)
        
        # Reshape the data to the correct dimensions (512, 512, 1001, 2)
        # The last dimension is 2 for the two components of velocity at each grid point
        data = data.reshape((1001,512, 512, 2))

        if sigma != 0:
            data = gaussian_filter(data, sigma=(0, sigma, sigma, 0), mode='wrap')
        data = data*(1/data.std())
        # return data[:,200:400,200:400,:]
        # return data[:,:250,:250,100:800]
        if slices:
            data = data[slices[0],slices[1],slices[2],:]
            print(data.shape)
        return data
    
class IsInDomain:
    def __init__(self, nodesCurves):
        self.loops = [mpath.Path(nodesCurves[i]) for i in range(len(nodesCurves))]
        
    def __call__(self, points):
        return np.logical_and(self.loops[0].contains_points(points),np.logical_not(np.array([loop.contains_points(points) for loop in self.loops[1:]]).any(0)))
    
class Interp2dAcrossTimesteps:
    def __init__(self, data, x_coords, y_coords, kind='linear'):
        # self.data = data
        # self.x_coords = x_coords
        # self.y_coords = y_coords
        self.interp = [RegularGridInterpolator((x_coords,y_coords),  np.transpose(data[i,:,:,:],(1,0,2)), method=kind, bounds_error=False, fill_value=None) for i in range(data.shape[0])]

    def __call__(self, points, timesteps):
        return np.concatenate([self.interp[timestep](points) for timestep in timesteps],-1)
    
    def stack(self, points, timesteps):
        return np.stack([self.interp[timestep](points) for timestep in timesteps],0)
    
    
class Interp2Dslice:
    def __init__(self, interp2dAT, dT_arr):
        self.interp2dAT = interp2dAT
        self.dT_arr = dT_arr
        self.nDims = len(dT_arr)*2
    
    def __call__(self, points):
        return self.interp2dAT(points, self.dT_arr)
        
    
class Interp1DPeriodic:
    def __init__(self, x, y, kind = 'linear'):
        self.x = x
        self.y = y
        self.kind = kind
        self.spline = interp1d(self.x,self.y,kind=self.kind,axis=0)
    
    def __call__(self, x):
        return self.spline(x%self.x[-1])
    
class Interp1DPchipPeriodic:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.spline = PchipInterpolator(self.x,self.y,axis=0)
    
    def __call__(self, x):
        return self.spline(x%self.x[-1])

class Interp1dAcrossTimesteps:
    def __init__(self, dataB, distance, kind='pchip'):
        self.nDims = dataB.shape[-1]
        if kind == 'pchip':
            self.interp = [Interp1DPchipPeriodic(distance, dataB[i]) for i in range(dataB.shape[0])]
        else:
            self.interp = [Interp1DPeriodic(distance, dataB[i],kind=kind) for i in range(dataB.shape[0])]

    def __call__(self, x, timesteps):
        # return np.stack([self.interp[timestep](x) for timestep in timesteps],0)
        return np.concatenate([self.interp[timestep](x) for timestep in timesteps],-1)
    
class Interp1Dslice:
    def __init__(self, interp1dAT, dT_arr):
        self.interp1dAT = interp1dAT
        self.dT_arr = dT_arr
        self.nDims = len(dT_arr)*interp1dAT.nDims
    
    def __call__(self, x):
        return self.interp1dAT(x, self.dT_arr)

# input: file_path,dT_arr
def loadData(file_path, dT_arr, gradsBC = False, slices=None, sigma=0):
    data = read_am_file(file_path,slices=slices,sigma=sigma)

    NT = data.shape[0]
    NX = data.shape[1]
    NY = data.shape[2]
    Nv = data.shape[3]

    x = np.linspace(0, 1, NX)
    X,Y = np.meshgrid(x, x)

    # Comon for all timesteps
    nodes = np.stack([X.flatten(),Y.flatten()],-1)
    idxCorner = np.array([0,NX-1,NX*(NY-1),NX*NY-1],dtype=int)
    elementsBoundaryNodesOnly = np.array([[idxCorner[0],idxCorner[2],idxCorner[1]],[idxCorner[2],idxCorner[3],idxCorner[1]]],dtype=int)
    areaElementsBoundaryNodesOnly = np.array([0.5,0.5])
    isInDomain = IsInDomain([nodes[[idxCorner[0],idxCorner[1],idxCorner[3],idxCorner[2],idxCorner[0]]]])
    idxCurves = [np.concatenate([np.arange(NX),np.arange(2*NX-1,NY*NX,NX),np.arange(NY*NX-2,(NY-1)*NX-1,-1),np.arange((NY-2)*NX,-1,-NX)])]
    distance = [np.arange(0,idxCurve.shape[0])*1.0/(NX-1) for idxCurve in idxCurves]
    lengthCurves = [4]
    nodesCurves = [nodes[idxCurve] for idxCurve in idxCurves]
    idxCurveCorner = np.array([0,0,0,0],dtype=int)
    distanceCornerCurve = np.array([0,1,3,2],dtype=int)
    distance2boundary = np.min(np.stack([np.abs(X),np.abs(X-1),np.abs(Y),np.abs(Y-1)],-1),-1)
    interpD2B = RegularGridInterpolator((x,x),distance2boundary, method='linear', bounds_error=False, fill_value=None)
    interpBC = [Interp1DPeriodic(distance[0],nodesCurves[0],kind='linear')]
    normalCurves = [np.zeros((idxCurves[0].shape[0],2))]
    normalCurves[0][nodesCurves[0][:,0]==0,0] = -1
    normalCurves[0][nodesCurves[0][:,0]==1,0] = 1
    normalCurves[0][nodesCurves[0][:,1]==0,1] = -1
    normalCurves[0][nodesCurves[0][:,1]==1,1] = 1
    normalCurves[0] = normalCurves[0]/np.sqrt(np.sum(normalCurves[0]**2,1,keepdims=True))
    interpBN = [Interp1DPeriodic(distance[0],normalCurves[0],kind='linear')]
    # idxCorner = [0,511,512*512-1,512*511]
    idxCorner = [0,NX-1,NX*NY-1,NX*(NY-1)]

    interp2dAcrossTimesteps = Interp2dAcrossTimesteps(data, x, x)
    dataB = [data.reshape((NT,NX*NY,Nv))[:,idxCurves[0],:]]
    if gradsBC:
        nodesB = [nodes[idxCurves[0]]]
        nodesB_offset = [nodesB[0]-normalCurves[0]*1.0/(NX-1)]
        dataB_offset = [interp2dAcrossTimesteps.stack(nodesB_offset[0], np.arange(NT))]
        dataB_dn = [(dataB[0]-dataB_offset[0])*(NX-1)*0.7]
        dataB = [np.concatenate([dataB_dn[0],dataB[0]],-1)]
        
    interp1dAcrossTimesteps = [Interp1dAcrossTimesteps(dataB[0], distance[0], kind='pchip')]

    data_processed = []
    for i in range(-dT_arr[0],NT-dT_arr[-1]):
        # interpSE = lambda points, idxs = dT_arr+i: interp2dAcrossTimesteps(points, idxs)
        # interpSD = lambda points, idxs = [i]: interp2dAcrossTimesteps(points, idxs)
        interpSE = Interp2Dslice(interp2dAcrossTimesteps, dT_arr+i)
        interpSD = Interp2Dslice(interp2dAcrossTimesteps, [i])
        # interpBS = [lambda x,idxs=dT_arr+i: interp1dAcrossTimesteps[0](x, idxs)]
        interpBS = [Interp1Dslice(interp1dAcrossTimesteps[0], dT_arr+i)]
        data_processed.append(
            {'nodes': nodes,'elementsBoundaryNodesOnly': elementsBoundaryNodesOnly,'areaElementsBoundaryNodesOnly': areaElementsBoundaryNodesOnly,
            'isInDomainF': isInDomain,'interpSE': interpSE,'interpSD': interpSD,'interpD2B': interpD2B,'nodesCurves': nodesCurves,
            'lengthCurves': lengthCurves,'interpBC': interpBC,'interpBS': interpBS,'interpBN': interpBN,
            'distaceCornerCurve': distanceCornerCurve, 'idxCurveCorner': idxCurveCorner, 'idxCorner': idxCorner, 'dT': i, 'distance':  distance}
        )

    return data_processed

def GenerateGeometry2(p):
    g = cfg.Geometry()
    for i in range(p.shape[0]):
        g.point(list(p[i]))
    
    for i in range(4):
        g.line([i,(i+1)%4],marker=1)
    g.surface([0,1,2,3])
    return g

def MeshSurface(g,elSize):
    mesh = cfm.GmshMesh(g)
    mesh.elType = 2       # Degrees of freedom per node.
    mesh.dofsPerNode = 1     # Factor that changes element sizes.
    mesh.elSizeFactor = elSize # Element size Factor
    nodes, edof, dofs, bdofs, elementmarkers = mesh.create()

    elements = edof-1
    boundaryNodes = [np.array(bdofs[1])-1]
    internalNodes = np.setdiff1d(np.arange(nodes.shape[0]), np.concatenate(boundaryNodes))
    return nodes, elements, boundaryNodes, internalNodes

def computeLengthAlongCurve(nodesB):
    dl = np.sqrt(((nodesB[1:]-nodesB[:-1])**2).sum(1))
    l = np.cumsum(dl)
    l = np.concatenate([[0],l],0)
    return l

def RemeshData(data,elSize):
    nodes = data['nodes']
    idxSquare = data['idxCorner']
    nodesB = nodes[idxSquare]

    g = GenerateGeometry2(nodesB)
    nodes,elements,boundaryNodes,internalNodes = MeshSurface(g,elSize)
    boundaryNodes = np.concatenate(boundaryNodes)

    node_labels = data['interpSD'](nodes)

    nodesB_orig = data['nodesCurves'][0]
    distance = data['distance']
    bcVals = data['interpBS'][0](distance)[0]
    normalVals = data['interpBN'][0](distance)[0]
    bcVals = np.concatenate([bcVals,normalVals],-1)
    interpBC = LinearNDInterpolator(nodesB_orig,bcVals)
    vals_interp = interpBC(nodes)
    boundary_mask = np.zeros((nodes.shape[0], 1), dtype=np.float32)
    boundary_node_indices = np.array(boundaryNodes) # Ensure it's a numpy array
    boundary_mask[boundary_node_indices] = 1.0
    node_features = np.hstack((vals_interp, boundary_mask)).astype(np.float32)

    return {'nodes': nodes, 'elements': elements, 'node_features': node_features, 'node_labels': node_labels, 'boundaryNodes': boundaryNodes, 'internalNodes': internalNodes}


# --- Graph Conversion Function (from your query) ---
def ConvertFemSolutionToGraph(solution):
    """
    Converts a dictionary representing an FEM solution into a Spektral Graph object.
    """
    nodes_coords = solution['nodes'] # Renamed to avoid conflict with Graph attribute
    elements = solution['elements']
    boundaryNodes = solution['boundaryNodes']
    internalNodes = solution['internalNodes']
    node_features = solution['node_features'] # These are the 'x' attributes for the graph
    node_labels = solution['node_labels']     # These are the 'y' attributes for the graph

    N = nodes_coords.shape[0] # Number of nodes

    # Construct Adjacency Matrix (a)
    edge_list = set()
    for i, j, k in elements:
        edge_list.add(tuple(sorted((i, j))))
        edge_list.add(tuple(sorted((i, k))))
        edge_list.add(tuple(sorted((j, k))))

    row_indices = []
    col_indices = []
    for i, j in edge_list:
        row_indices.extend([i, j])
        col_indices.extend([j, i])

    adj_data = np.ones(len(row_indices), dtype=np.float32)
    adj_matrix_sparse = sp.csr_matrix((adj_data, (row_indices, col_indices)), shape=(N, N))

    # Construct Edge Features (e)
    adj_matrix_coo = adj_matrix_sparse.tocoo()
    source_nodes_idx = adj_matrix_coo.row
    target_nodes_idx = adj_matrix_coo.col

    # Calculate edge features:
    # - Relative position (dx, dy)
    # - Edge length
    relative_pos = nodes_coords[source_nodes_idx] - nodes_coords[target_nodes_idx] # Shape: (num_edges, 2)
    edge_length = np.linalg.norm(relative_pos, axis=1, keepdims=True) # Shape: (num_edges, 1)

    # Concatenate edge features: [dx, dy, length]
    edge_features = np.hstack((relative_pos, edge_length)).astype(np.float32)

    # Create the Spektral Graph object
    return Graph(x=node_features.astype(np.float32),  # Ensure float32 for consistency
                 a=adj_matrix_sparse,
                 e=edge_features.astype(np.float32), # Ensure float32 for consistency
                 y=node_labels,
                 # Store original FEM solution parts as custom attributes if needed
                 fem_nodes=nodes_coords, # Using 'fem_nodes' to avoid potential conflict if Spektral uses 'nodes'
                 fem_elements=elements,
                 fem_internalNodes=internalNodes,
                 fem_boundaryNodes=boundaryNodes,
                 fem_edge_vector=relative_pos.astype(np.float32),
                 fem_edge_length=edge_length.astype(np.float32))


# --- FEMDataset Class (from your query) ---
class FEMDataset(Dataset):
    """
    A Spektral Dataset for FEM simulation results.
    It takes a list of solution dictionaries and converts them to Graph objects.
    """
    def __init__(
        self,
        data_dict_list = [], # Expects a list of solution dictionaries
        cache_file="data/fem_dataset_graphs.pkl",
        force_regenerate=False, # Added for easier testing
        **kwargs
    ):
        self.data_dict_list = data_dict_list
        self.cache_file = cache_file
        self.force_regenerate = force_regenerate

        # Ensure cache directory exists
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"Created cache directory: {cache_dir}")

        super().__init__(**kwargs)

    def read(self):
        # 1) If we already generated & cached, just load that:
        if not self.force_regenerate and os.path.exists(self.cache_file):
            print(f"Loading cached dataset from {self.cache_file}")
            try:
                with open(self.cache_file, "rb") as f:
                    graphs = pickle.load(f)

                print(f"Successfully loaded {len(graphs)} graphs from cache.")
                return graphs
            except Exception as e:
                print(f"Error loading from cache: {e}. Regenerating.")

        # 2) Otherwise, generate from scratch:
        print(f"Generating {len(self.data_dict_list)} new graph samples...")
        graphs = []
        for i, sol_dict in enumerate(self.data_dict_list):
            # print(f"Processing solution {i+1}/{len(self.data_dict_list)}")
            g = ConvertFemSolutionToGraph(sol_dict)
            graphs.append(g)

        # 3) Cache for next time:
        if self.cache_file:
            print(f"Caching {len(graphs)} graphs to {self.cache_file}")
            try:
                with open(self.cache_file, "wb") as f:
                    pickle.dump(graphs, f)
            except Exception as e:
                print(f"Error saving to cache: {e}")
        else:
            print("Cache file not specified. Skipping caching.")
            
        return graphs

# --- Feature Normalization Function (Modified) ---
def normalize_dataset_features(train_dataset, val_dataset=None, epsilon=1e-8):
    """
    Normalizes node features (x) and edge features (e).
    Calculates mean and std from train_dataset ONLY.
    Applies normalization to train_dataset and optionally to val_dataset.
    Modifies the Graph objects in the datasets in-place.

    Args:
        train_dataset (spektral.data.Dataset): The training dataset.
        val_dataset (spektral.data.Dataset, optional): The validation dataset. Defaults to None.
        epsilon (float): A small value for numerical stability.
    Returns:
        tuple: (x_mean, x_std, e_mean, e_std) calculated from the training set.
               Returns None for e_mean, e_std if no edge features.
               Returns None for x_mean, x_std if no node features.
    """
    if not train_dataset:
        print("Training dataset is empty. Nothing to normalize.")
        return None, None, None, None

    print("Starting feature normalization...")
    x_mean, x_std, e_mean, e_std = None, None, None, None

    # --- Collect and Normalize Node Features (x) ---
    all_train_node_features = []
    for g in train_dataset:
        if g.x is not None and g.x.shape[0] > 0:
            all_train_node_features.append(g.x)

    if all_train_node_features:
        all_train_node_features_np = np.vstack(all_train_node_features)
        x_mean = np.mean(all_train_node_features_np, axis=0)
        x_std = np.std(all_train_node_features_np, axis=0)
        print(f"Node feature global mean (from train_dataset, shape {x_mean.shape}): {x_mean}")
        print(f"Node feature global std (from train_dataset, shape {x_std.shape}): {x_std}")

        # Apply normalization to training node features
        for g in train_dataset:
            if g.x is not None and g.x.shape[0] > 0:
                g.x = (g.x - x_mean) / (x_std + epsilon)
        print("Training node features normalized.")

        # Apply normalization to validation node features (if provided)
        if val_dataset:
            for g in val_dataset:
                if g.x is not None and g.x.shape[0] > 0:
                    g.x = (g.x - x_mean) / (x_std + epsilon)
            print("Validation node features normalized.")
    else:
        print("No node features found in the training dataset to normalize.")

    # --- Collect and Normalize Edge Features (e) ---
    all_train_edge_features = []
    for g in train_dataset:
        if g.e is not None and g.e.shape[0] > 0:
            all_train_edge_features.append(g.e)

    if all_train_edge_features:
        all_train_edge_features_np = np.vstack(all_train_edge_features)
        e_mean = np.mean(all_train_edge_features_np, axis=0)
        e_std = np.std(all_train_edge_features_np, axis=0)
        print(f"Edge feature global mean (from train_dataset, shape {e_mean.shape}): {e_mean}")
        print(f"Edge feature global std (from train_dataset, shape {e_std.shape}): {e_std}")

        # Apply normalization to training edge features
        for g in train_dataset:
            if g.e is not None and g.e.shape[0] > 0:
                g.e = (g.e - e_mean) / (e_std + epsilon)
        print("Training edge features normalized.")

        # Apply normalization to validation edge features (if provided)
        if val_dataset:
            for g in val_dataset:
                if g.e is not None and g.e.shape[0] > 0:
                    g.e = (g.e - e_mean) / (e_std + epsilon)
            print("Validation edge features normalized.")
    else:
        print("No edge features found in the training dataset to normalize.")
    
    print("Feature normalization complete.")
    return x_mean, x_std, e_mean, e_std


class PatchedGCNConv(GCNConv):
    def call(self, inputs, mask=None):
        # inputs: a list [X, A] (or [X, A, E] depending on the layer)
        # mask:   can be None, a Tensor, or a list of Tensors
        if isinstance(mask, list) and mask and mask[0] is None:
            mask = None
        # Delegate to the real implementation
        # Note: super().call expects signature call(self, inputs, **kwargs)
        return super().call(inputs, mask=mask)
    

class PatchedGATConv(GATConv):
    def call(self, inputs, mask=None):
        if isinstance(mask, list) and mask and mask[0] is None:
            mask = None
        return super().call(inputs, mask=mask)

class PatchedECCConv(ECCConv):
    def call(self, inputs, mask=None):
        if isinstance(mask, list) and mask and mask[0] is None:
            mask = None
        return super().call(inputs, mask=mask) 

# --- main model -----------------------------------------------------------
class FEMGNN(tf.keras.Model):                # Sub-classing is the recommended way
    def __init__(self, hidden_dim, n_gnn_layers, out_dim, ecc_conv_kernel_network = None, **kwargs):
        """
        hidden_dim : units in the encoder GCNs
        out_dim    : units in the final GCN output
        """
        super().__init__(**kwargs)

        self.ecc_conv_kernel_network = ecc_conv_kernel_network

        # ===== graph nerual network layers ==============================================
        self.in_dense = tf.keras.layers.Dense(hidden_dim, activation=None)
        self.n_gnn_layers = n_gnn_layers
        self.gcn_arr = [PatchedECCConv(hidden_dim, activation="relu", kernel_network=self.ecc_conv_kernel_network) for _ in range(self.n_gnn_layers)]
        # self.gcn_arr = [PatchedGCNConv(hidden_dim, activation="relu", kernel_network=self.ecc_conv_kernel_network) for _ in range(self.n_gnn_layers)]
        self.mlp_arr = [Dense(hidden_dim, activation="relu") for _ in range(self.n_gnn_layers)]
        self.out_dense = tf.keras.layers.Dense(out_dim, activation=None)   # per-node RHS/source

    def call(self, inputs, training=False):
        """
        inputs = { 'x': node features  [N,F],
                   'a': SparseTensor   [N,N],
                   'mesh': dict(nodes, elements, boundaryNodes, internalNodes, r) }
        """
        x, a, e, mesh = inputs["x"], inputs["a"], inputs["e"], inputs["mesh"]

        # edge_vector = mesh['edge_vector']
        # edge_length = mesh['edge_length']

        h = self.in_dense(x)
        for i in range(self.n_gnn_layers):
            h_gcn  = self.gcn_arr[i]([h, a, e], training=training)
            # h_gcn  = self.gcn_arr[i]([h, a], training=training)
            h_mlp = self.mlp_arr[i](h,training=training)
            h = h_gcn + h_mlp + h
            # h = h_mlp + h
        
        out = self.out_dense(h)

        return out                     # we expose l_pred for monitoring`
