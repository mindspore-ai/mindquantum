import sys
import time
from math import log10, log2
from os.path import dirname, abspath, exists
from copy import deepcopy
from traceback import print_exc
import numpy as np
# sys.path.append('/data/panfeng/dev/multi-cache')
# from multicache import (
#     Contraction, QuantumCircuit, ctg2normal, normal2ctg
# )
from load_circuits import QuantumCircuit
# sys.path.append('/data/panfeng/dev/artensor')
# from artensor_experimenting import (
#     find_order,
#     log10sumexp2,
#     contraction_scheme_sparse,
#     tensor_contraction_sparse,
#     ContractionTree, 
#     AbstractTensorNetwork, 
#     # simulate_annealing,
#     # GreedyOrderFinder,
#     # contraction_scheme_sparse_einsum_1,
# )
from artensor import (
    find_order,
    log10sumexp2,
    contraction_scheme_sparse,
    tensor_contraction_sparse,
    ContractionTree, 
    AbstractTensorNetwork, 
)
import os
import torch
import time
from copy import deepcopy
# from cutensor.torch import EinsumGeneral

def tensor_contraction_sparse(tensors, contraction_scheme, scientific_notation=False, use_cutensor=False):
    if scientific_notation:
        factor = torch.tensor(0, dtype=tensors[0].dtype, device=tensors[0].device)

    if use_cutensor:
        pass
        # einsum_fuc = EinsumGeneral
    else:
        einsum_fuc = torch.einsum

    for step in contraction_scheme:
        i, j = step[0]
        batch_i, batch_j = step[2]
        try:
            if len(batch_i) > 1:
                tensors[i] = [tensors[i]]
                for k in range(len(batch_i)-1, -1, -1):
                    if k != 0:
                        if step[3]:
                            tensors[i].insert(
                                1, 
                                einsum_fuc(
                                    step[1],
                                    tensors[i][0][batch_i[k]], 
                                    tensors[j][batch_j[k]], 
                                ).reshape(step[3])
                            )
                        else:
                            tensors[i].insert(
                                1, 
                                einsum_fuc(
                                    step[1],
                                    tensors[i][0][batch_i[k]], 
                                    tensors[j][batch_j[k]])
                            )
                    else:
                        if step[3]:
                            tensors[i][0] = einsum_fuc(
                                step[1],
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]], 
                            ).reshape(step[3])
                        else:
                            tensors[i][0] = einsum_fuc(
                                step[1],
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]], 
                            )
                tensors[j] = []
                tensors[i] = torch.cat(tensors[i], dim=0)
            elif len(step) > 3 and len(batch_i) == len(batch_j) == 1:
                tensors[i] = tensors[i][batch_i[0]]
                tensors[j] = tensors[j][batch_j[0]]
                tensors[i] = einsum_fuc(step[1], tensors[i], tensors[j])
            elif len(step) > 3:
                tensors[i] = einsum_fuc(
                    step[1],
                    tensors[i],
                    tensors[j],
                ).reshape(step[3])
                if len(batch_i) == 1:
                    tensors[i] = tensors[i][batch_i[0]]
                tensors[j] = []
            else:
                tensors[i] = einsum_fuc(step[1], tensors[i], tensors[j])
                tensors[j] = []
        except:
            print(step[:2], len(step[2][0]))
            for k in range(len(tensors)):
                if type(tensors[k]) is not list:
                    print(k, np.log2(np.prod(tensors[k].shape)))
                elif len(tensors[k]) > 1:
                    for l, tensor in enumerate(tensors[k]):
                        print(l, np.log2(np.prod(tensor.shape)))
            print_exc()
            sys.exit(1)

        if scientific_notation:
            norm_factor = tensors[i].abs().max()
            tensors[i] /= norm_factor
            factor += torch.log10(norm_factor)

    if scientific_notation:
        return factor, tensors[i]
    else:
        return tensors[i]

def read_samples(filename):
    import os
    if os.path.exists(filename):
        samples_data = []
        with open(filename, 'r') as f:
            l = f.readlines()
        f.close()
        for line in l:
            ll = line.split()
            if len(ll) == 1:
                samples_data.append(ll[0])
            else:
                samples_data.append([ll[0], float(ll[1]) + 1j*float(ll[2])])
        return samples_data
    else:
        raise ValueError("{} does not exist".format(filename))

def slicing_order_complexity_multibitstring(neighbors, shapes, order, slicing_edge=None, num_fq=None, max_bitstring=1):
    neighbors_new = deepcopy(neighbors)
    shapes_new = deepcopy(shapes)
    if num_fq is None:
        num_fq = [0 for i in range(len(neighbors_new))]
    else:
        num_fq = list(num_fq)
    if slicing_edge is not None:
        if type(slicing_edge) is tuple:
            slicing_edge = [slicing_edge]
        for edge in slicing_edge:
            i, j = edge
            if i in neighbors_new[j] and j in neighbors_new[i]:
                idxi_j = neighbors_new[i].index(j)
                idxj_i = neighbors_new[j].index(i)
                shapes_new[i].pop(idxi_j) #shapes_new[i][idxi_j] = 1
                shapes_new[j].pop(idxj_i) # shapes_new[j][idxj_i] = 1
                neighbors_new[i].pop(idxi_j)
                neighbors_new[j].pop(idxj_i)
            else:
                raise ValueError('({}, {}) is not a valid edge'.format(i, j))
    time_complexity = []
    space_complexity = [log2(np.prod(shapes_new[i])) + min(log2(max_bitstring), num_fq[i]) for i in range(len(shapes_new))]
    memory_complexity = []
    for edge in order:
        i, j = edge
        if i in neighbors_new[j] and j in neighbors_new[i]:
            flag = True
        else:
            flag = False
        try:
            if flag:
                idxi_j = neighbors_new[i].index(j)
                idxj_i = neighbors_new[j].index(i)
                shapeij = shapes_new[i].pop(idxi_j)
                shapes_new[j].pop(idxj_i)
                neighbors_new[i].pop(idxi_j)
                neighbors_new[j].pop(idxj_i)
            else:
                shapeij = 1
            shapei, shapej = np.prod(shapes_new[i]), np.prod(shapes_new[j])
            for node in neighbors_new[j]:
                idxj_n = neighbors_new[j].index(node)
                idxn_j = neighbors_new[node].index(j)
                if node in neighbors_new[i]:
                    idxi_n = neighbors_new[i].index(node)
                    idxn_i = neighbors_new[node].index(i)
                    shapes_new[i][idxi_n] *= shapes_new[j][idxj_n]
                    shapes_new[node][idxn_i] *= shapes_new[node][idxn_j]
                    shapes_new[node].pop(idxn_j)
                    neighbors_new[node].pop(idxn_j)
                else:
                    shapes_new[i].append(shapes_new[j][idxj_n])
                    neighbors_new[i].append(node)
                    neighbors_new[node][idxn_j] = i
        except:
            raise ValueError('error when contract {} and {}'.format(i, j))
        neighbors_new[j] = []
        shapes_new[j] = []
        tc_step = log2(shapei) + log2(shapeij) + log2(shapej)
        factor = min(log2(max_bitstring), num_fq[i] + num_fq[j])
        if num_fq[i] + num_fq[j] > log2(max_bitstring):
            memory_complexity.append(
                log2(
                    2 ** (sum([log2(shape) for shape in shapes_new[edge[0]]]) + factor) +
                    2 ** (log2(shapei) + log2(shapeij) + factor) + 
                    2 ** (log2(shapej) + log2(shapeij) + factor)
                )
            )
        else:
            memory_complexity.append(
                log2(
                    2 ** (sum([log2(shape) for shape in shapes_new[edge[0]]]) + factor) +
                    2 ** (log2(shapei) + log2(shapeij) + num_fq[i]) + 
                    2 ** (log2(shapej) + log2(shapeij) + num_fq[j])
                )
            )
        # if (num_fq[i] < log2(max_bitstring) and num_fq[j] < log2(max_bitstring) and num_fq[i] + num_fq[j] > np.ceil(log2(max_bitstring))) or (max(num_fq[i], num_fq[j]) >= log2(max_bitstring) and min(num_fq[i], num_fq[j]) > 0):
        if tc_step + factor > 34 or num_fq[i] + num_fq[j] > 0:
            print(
                edge, f'{order.index(edge)}/{len(order)}', num_fq[i], num_fq[j], factor, 
                (log2(shapei), log2(shapeij), log2(shapej)), tc_step, memory_complexity[-1],
                sum([log2(shape) for shape in shapes_new[edge[0]]]) + factor
            )
        num_fq[i] += num_fq[j]
        try:
            time_complexity.append(tc_step + factor)
        except:
            print(i, j, shapei, shapej, shapeij, shapei * shapej * shapeij, factor)
            print_exc()
            exit(0)
        space_complexity.append(sum([log2(shape) for shape in shapes_new[edge[0]]]) + factor)
    tc = log10sumexp2(time_complexity)
    sc = max(space_complexity)
    mc = log10sumexp2(memory_complexity)
    print(mc)

    return tc, sc, time_complexity, space_complexity

def detect_open_qubits(n=53, m=12, seq='ABCDCDAB', device='cuda:1', sc_target=31, max_bitstrings=2**20, select=0):
    qc = QuantumCircuit(n, m, seq=seq)
    edges = []
    for i in range(len(qc.neighbors)):
        for j in qc.neighbors[i]:
            if i < j:
                edges.append((i, j))
    neighbors = list(qc.neighbors)
    shapes = [list(qc.tensors[i].shape) for i in range(len(neighbors))]
    tensor_bonds = {i:[edges.index((min(i, j), max(i, j))) for j in neighbors[i]] for i in range(len(neighbors))}
    bond_dims = {i:2.0 for i in range(len(edges))}
    np.random.seed(0)
    final_qubits = set(range(len(neighbors) - n, len(neighbors)))

    tensor_network = AbstractTensorNetwork(
        deepcopy(tensor_bonds), 
        deepcopy(bond_dims), 
        deepcopy(final_qubits), 
        max_bitstrings)
    bonds_tensor_original = deepcopy(tensor_network.bond_tensors)

    
    slicing_bonds = {554: 2.0, 566: 2.0, 384: 2.0, 609: 2.0, 560: 2.0, 466: 2.0, 418: 2.0, 520: 2.0, 506: 2.0, 616: 2.0, 374: 2.0, 328: 2.0, 697: 2.0, 526: 2.0, 571: 2.0, 467: 2.0, 568: 2.0, 617: 2.0, 530: 2.0, 458: 2.0, 426: 2.0, 429: 2.0, 561: 2.0, 419: 2.0}
    # open 0 qubit
    order_slicing = [(51, 31), (51, 74), (51, 75), (117, 94), (51, 117), (52, 32), (51, 52), (160, 136), (160, 179), (201, 221), (202, 201), (160, 202), (51, 160), (164, 121), (140, 113), (164, 140), (175, 197), (132, 175), (132, 156), (132, 183), (164, 132), (51, 164), (98, 79), (55, 35), (55, 78), (98, 55), (98, 13), (70, 47), (27, 9), (27, 5), (70, 27), (90, 71), (70, 90), (98, 70), (51, 98), (143, 118), (161, 180), (137, 152), (137, 110), (161, 137), (143, 161), (95, 76), (143, 95), (51, 143), (24, 10), (24, 45), (87, 67), (66, 44), (66, 23), (87, 66), (24, 87), (28, 6), (28, 48), (24, 28), (51, 24), (56, 36), (33, 14), (56, 33), (29, 11), (56, 29), (51, 56), (102, 82), (59, 39), (102, 59), (102, 17), (81, 38), (81, 58), (81, 101), (102, 81), (103, 83), (103, 60), (103, 40), (103, 16), (126, 18), (103, 126), (102, 103), (124, 37), (125, 15), (124, 125), (102, 124), (51, 102), (186, 167), (53, 34), (186, 53), (51, 186), (25, 7), (51, 25), (165, 141), (145, 122), (165, 145), (169, 296), (188, 169), (273, 253), (273, 230), (273, 210), (188, 273), (165, 188), (168, 99), (144, 119), (168, 144), (165, 168), (51, 165), (123, 114), (100, 57), (100, 80), (123, 100), (51, 123), (96, 77), (97, 54), (96, 97), (91, 72), (91, 49), (96, 91), (51, 96), (187, 166), (142, 115), (142, 92), (187, 142), (228, 209), (208, 251), (228, 208), (252, 272), (252, 271), (252, 229), (228, 252), (187, 228), (51, 187), (162, 138), (111, 88), (162, 111), (51, 162), (30, 12), (30, 8), (93, 50), (93, 73), (30, 93), (155, 69), (46, 26), (155, 46), (89, 68), (155, 89), (30, 155), (51, 30), (250, 227), (250, 270), (269, 292), (250, 269), (226, 207), (250, 226), (248, 268), (249, 248), (249, 225), (249, 205), (249, 206), (250, 249), (139, 120), (139, 163), (139, 112), (250, 139), (51, 250), (157, 133), (157, 184), (185, 158), (157, 185), (134, 108), (157, 134), (51, 157), (105, 85), (64, 42), (105, 64), (107, 1), (105, 107), (63, 20), (63, 84), (128, 62), (41, 19), (128, 41), (63, 128), (105, 63), (21, 2), (21, 3), (61, 0), (61, 127), (21, 61), (105, 21), (65, 43), (22, 4), (65, 22), (106, 86), (65, 106), (105, 65), (51, 105), (146, 104), (146, 129), (146, 172), (177, 150), (146, 177), (153, 130), (146, 153), (51, 146), (203, 199), (203, 181), (176, 149), (176, 171), (203, 176), (218, 198), (241, 217), (218, 241), (203, 218), (51, 203), (174, 148), (174, 190), (154, 131), (174, 154), (195, 182), (174, 195), (51, 174), (173, 147), (173, 191), (223, 204), (173, 223), (215, 194), (173, 215), (51, 173), (267, 247), (224, 159), (267, 224), (267, 219), (200, 178), (200, 116), (135, 109), (135, 151), (200, 135), (267, 200), (51, 267), (259, 238), (259, 263), (239, 196), (239, 216), (239, 325), (259, 239), (243, 220), (259, 243), (51, 259), (290, 231), (290, 170), (51, 290), (282, 211), (232, 189), (282, 232), (261, 242), (261, 262), (282, 261), (51, 282), (312, 285), (293, 336), (312, 293), (289, 266), (312, 289), (246, 265), (245, 222), (246, 245), (312, 246), (51, 312), (255, 234), (255, 212), (255, 256), (279, 235), (213, 192), (279, 213), (255, 279), (51, 255), (276, 233), (276, 254), (318, 301), (276, 318), (297, 275), (276, 297), (51, 276), (344, 324), (344, 309), (344, 333), (311, 284), (344, 311), (51, 344), (363, 412), (367, 363), (321, 305), (321, 348), (329, 286), (321, 329), (367, 321), (51, 367), (328, 304), (278, 258), (328, 278), (236, 193), (214, 236), (257, 214), (257, 237), (328, 257), (342, 320), (342, 347), (299, 274), (299, 316), (342, 299), (328, 342), (51, 328), (319, 298), (319, 341), (303, 277), (319, 303), (346, 327), (319, 346), (387, 419), (387, 418), (387, 366), (319, 387), (51, 319), (300, 281), (300, 317), (323, 308), (300, 323), (365, 343), (417, 416), (417, 386), (365, 417), (300, 365), (51, 300), (411, 410), (411, 384), (411, 362), (383, 408), (411, 383), (352, 420), (352, 371), (411, 352), (51, 411), (429, 428), (429, 391), (429, 340), (429, 406), (382, 405), (361, 407), (382, 361), (360, 404), (382, 360), (429, 382), (51, 429), (375, 355), (359, 403), (375, 359), (437, 402), (395, 438), (437, 395), (375, 437), (51, 375), (370, 351), (332, 409), (370, 332), (378, 357), (338, 295), (314, 338), (378, 314), (370, 378), (445, 426), (390, 427), (445, 390), (398, 444), (445, 398), (370, 445), (51, 370), (339, 315), (339, 454), (358, 339), (358, 335), (358, 354), (394, 374), (436, 435), (394, 436), (358, 394), (401, 453), (381, 451), (401, 381), (401, 452), (434, 433), (434, 393), (401, 434), (358, 401), (51, 358), (345, 326), (345, 302), (334, 291), (353, 334), (353, 310), (345, 353), (368, 421), (345, 368), (330, 287), (330, 306), (330, 349), (330, 376), (372, 430), (264, 244), (372, 264), (330, 372), (260, 439), (260, 240), (260, 283), (330, 260), (345, 330), (396, 440), (396, 441), (396, 377), (345, 396), (356, 331), (313, 294), (313, 288), (337, 313), (356, 337), (307, 280), (356, 307), (432, 431), (432, 392), (356, 432), (373, 350), (364, 322), (373, 364), (356, 373), (345, 356), (379, 413), (389, 424), (389, 425), (379, 389), (423, 422), (423, 388), (423, 369), (423, 446), (379, 423), (345, 379), (51, 345), (450, 443), (400, 449), (450, 400), (450, 397), (380, 442), (450, 380), (399, 448), (399, 447), (450, 399), (51, 450), (415, 414), (415, 385), (51, 415)]

    # open 10 qubit
    # order_slicing = [(51, 31), (51, 74), (51, 75), (117, 94), (51, 117), (52, 32), (51, 52), (160, 136), (160, 179), (201, 221), (202, 201), (160, 202), (51, 160), (164, 121), (140, 113), (164, 140), (175, 197), (132, 175), (132, 156), (132, 183), (164, 132), (51, 164), (98, 79), (55, 35), (55, 78), (98, 55), (98, 13), (70, 47), (27, 9), (27, 5), (70, 27), (90, 71), (70, 90), (98, 70), (51, 98), (143, 118), (161, 180), (137, 152), (137, 110), (161, 137), (143, 161), (95, 76), (143, 95), (51, 143), (24, 10), (24, 45), (87, 67), (66, 44), (66, 23), (87, 66), (24, 87), (28, 6), (28, 48), (24, 28), (51, 24), (56, 36), (33, 14), (56, 33), (29, 11), (56, 29), (51, 56), (102, 82), (59, 39), (102, 59), (102, 17), (81, 38), (81, 58), (81, 101), (102, 81), (103, 83), (103, 60), (103, 40), (103, 16), (126, 18), (103, 126), (102, 103), (124, 37), (125, 15), (124, 125), (102, 124), (51, 102), (186, 167), (53, 34), (186, 53), (51, 186), (25, 7), (51, 25), (165, 141), (145, 122), (165, 145), (169, 296), (188, 169), (273, 253), (273, 230), (273, 210), (188, 273), (165, 188), (168, 99), (144, 119), (168, 144), (165, 168), (51, 165), (123, 114), (100, 57), (100, 80), (123, 100), (51, 123), (96, 77), (97, 54), (96, 97), (91, 72), (91, 49), (96, 91), (51, 96), (187, 166), (142, 115), (142, 92), (187, 142), (228, 209), (208, 251), (228, 208), (252, 272), (252, 271), (252, 229), (228, 252), (187, 228), (51, 187), (162, 138), (111, 88), (162, 111), (51, 162), (30, 12), (30, 8), (93, 50), (93, 73), (30, 93), (155, 69), (46, 26), (155, 46), (89, 68), (155, 89), (30, 155), (51, 30), (250, 227), (250, 270), (269, 292), (250, 269), (226, 207), (250, 226), (248, 268), (249, 248), (249, 225), (249, 205), (249, 206), (250, 249), (139, 120), (139, 163), (139, 112), (250, 139), (51, 250), (157, 133), (157, 184), (185, 158), (157, 185), (134, 108), (157, 134), (51, 157), (105, 85), (64, 42), (105, 64), (107, 1), (105, 107), (63, 20), (63, 84), (128, 62), (41, 19), (128, 41), (63, 128), (105, 63), (21, 2), (21, 3), (61, 0), (61, 127), (21, 61), (105, 21), (65, 43), (22, 4), (65, 22), (106, 86), (65, 106), (105, 65), (51, 105), (146, 104), (146, 129), (146, 172), (177, 150), (146, 177), (153, 130), (146, 153), (51, 146), (203, 199), (203, 181), (176, 149), (176, 171), (203, 176), (218, 198), (241, 217), (218, 241), (203, 218), (51, 203), (174, 148), (174, 190), (154, 131), (174, 154), (195, 182), (174, 195), (51, 174), (173, 147), (173, 191), (223, 204), (173, 223), (215, 194), (173, 215), (51, 173), (267, 247), (224, 159), (267, 224), (267, 219), (200, 178), (200, 116), (135, 109), (135, 151), (200, 135), (267, 200), (51, 267), (259, 238), (259, 263), (239, 196), (239, 216), (239, 325), (259, 239), (243, 220), (259, 243), (51, 259), (290, 231), (290, 170), (51, 290), (282, 211), (232, 189), (282, 232), (261, 242), (261, 262), (282, 261), (51, 282), (312, 285), (293, 336), (312, 293), (289, 266), (312, 289), (246, 265), (245, 222), (246, 245), (312, 246), (51, 312), (255, 234), (255, 212), (255, 256), (279, 235), (213, 192), (279, 213), (255, 279), (51, 255), (276, 233), (276, 254), (318, 301), (276, 318), (297, 275), (276, 297), (51, 276), (344, 324), (344, 309), (344, 333), (311, 284), (344, 311), (51, 344), (363, 412), (367, 363), (321, 305), (321, 348), (329, 286), (321, 329), (367, 321), (51, 367), (328, 304), (278, 258), (328, 278), (236, 193), (214, 236), (257, 214), (257, 237), (328, 257), (342, 320), (342, 347), (299, 274), (299, 316), (342, 299), (328, 342), (51, 328), (319, 298), (319, 341), (303, 277), (319, 303), (346, 327), (319, 346), (387, 419), (387, 418), (387, 366), (319, 387), (51, 319), (300, 281), (300, 317), (323, 308), (300, 323), (365, 343), (417, 416), (417, 386), (365, 417), (300, 365), (51, 300), (411, 410), (411, 384), (411, 362), (383, 408), (411, 383), (352, 420), (352, 371), (411, 352), (51, 411), (429, 428), (429, 391), (429, 340), (429, 406), (382, 405), (361, 407), (382, 361), (360, 404), (382, 360), (429, 382), (51, 429), (375, 355), (359, 403), (375, 359), (437, 402), (395, 438), (437, 395), (375, 437), (51, 375), (370, 351), (332, 409), (370, 332), (378, 357), (338, 295), (314, 338), (378, 314), (370, 378), (445, 426), (390, 427), (445, 390), (398, 444), (445, 398), (370, 445), (51, 370), (339, 315), (339, 454), (358, 339), (358, 335), (358, 354), (394, 374), (436, 435), (394, 436), (358, 394), (401, 453), (381, 451), (401, 381), (401, 452), (434, 433), (434, 393), (401, 434), (358, 401), (51, 358), (345, 326), (345, 302), (334, 291), (353, 334), (353, 310), (345, 353), (368, 421), (345, 368), (330, 287), (330, 306), (330, 349), (330, 376), (372, 430), (264, 244), (372, 264), (330, 372), (260, 439), (260, 240), (260, 283), (330, 260), (345, 330), (396, 440), (396, 441), (396, 377), (345, 396), (356, 331), (313, 294), (313, 288), (337, 313), (356, 337), (307, 280), (356, 307), (432, 431), (432, 392), (356, 432), (373, 350), (364, 322), (373, 364), (356, 373), (345, 356), (379, 413),                         (379, 389), (423, 388), (423, 369),                         (379, 423), (345, 379), (51, 345),                         (450, 400), (450, 397),             (450, 380),                         (450, 399), (51, 450),             (415, 385), (51, 415)]

    
    contracted_tensors = list(set(sum([list(o) for o in order_slicing], start=[])))
    exclude_final_qubits = [i for i in range(len(neighbors)) if i not in contracted_tensors] # [414, 415, 416, 417, 424, 425, 433, 434, 452, 453]
    exclude_final_qubits_id = [i-len(neighbors)+n for i in exclude_final_qubits]
    print('open_final_qubits_id', exclude_final_qubits_id)

    slicing_edges = [edges[i] for i in slicing_bonds]
    tc, sc, tcs, scs = slicing_order_complexity_multibitstring(neighbors, shapes, order_slicing, slicing_edges, tensor_network.num_fq, max_bitstrings)
    print(tc, sc, tc + len(slicing_bonds) * log10(2))

    final_qubits_new = set(range(len(contracted_tensors) - n + len(exclude_final_qubits), len(contracted_tensors)))
    tensor_network = AbstractTensorNetwork(
        {i:tensor_bonds[j] for i, j in enumerate(contracted_tensors)}, 
        deepcopy(bond_dims), 
        final_qubits_new, 
        max_bitstrings)
    print(len(final_qubits_new))
    # assert final_qubits_new == final_qubits
    # assert {i:tensor_bonds[j] for i, j in enumerate(contracted_tensors)} == tensor_bonds

    order_slicing = [(contracted_tensors.index(order[0]), contracted_tensors.index(order[1])) for order in order_slicing]
    print('after reindex')
    print(order_slicing)

    for bond in slicing_bonds:
        tensor_network.slicing(bond)
    ctree = ContractionTree(tensor_network, order_slicing, 0)
    tc, sc, mc = ctree.tree_complexity()
    order_slicing = ctree.tree_order_dfs()
    print(tc, sc, mc)

    samples_data = read_samples(f'measurements_n53_m20_s0_e0_pABCDCDAB.txt') # /data/public/sycamore_circuits/n{n}_m{m}/measurements_n{n}_m{m}_s0_e0_p{seq}.txt
    bitstrings = [samples_data[i] for i in range(len(samples_data))][:max_bitstrings]
    max_bitstrings = len(np.unique(bitstrings))
    bitstrings_new = [''.join([b[i] for i in range(len(b)) if i not in exclude_final_qubits_id]) for b in bitstrings]

    dtype = torch.complex64
    slicing_edges = [edges[i] for i in slicing_bonds]
    slicing_indices = {}.fromkeys(slicing_edges)

    tensors_save = []
    for x in range(len(contracted_tensors)):
        if x in final_qubits_new:
            tensors_save.append(
                torch.tensor([[1, 0], [0, 1]], dtype=dtype, device='cpu')
            )
        else:
            tensors_save.append(qc.tensors[x].to('cpu'))

    neighbors_copy = [neighbors[i] for i in contracted_tensors]
    for x, y in slicing_edges:
        idxi_j = neighbors_copy[x].index(y)
        idxj_i = neighbors_copy[y].index(x)
        neighbors_copy[x].pop(idxi_j)
        neighbors_copy[y].pop(idxj_i)
        slicing_indices[(x, y)] = (idxi_j, idxj_i)

    scheme_filename = abspath(dirname(__file__)) + f'/contraction_scheme/640G_reproduce_scheme_n{n}_m{m}_{seq}_{max_bitstrings}_einsum_{select}_open_2.pt'
    print(scheme_filename)
    if exists(scheme_filename):
        _, scheme, _, bitstrings_sorted = torch.load(scheme_filename)
    else:
        t0 = time.time()
        scheme, bonds_final, bitstrings_sorted = contraction_scheme_sparse(ctree, bitstrings_new, sc_target=sc_target)
        torch.save((tensors_save, scheme, slicing_indices, bitstrings_sorted), scheme_filename)
        t1 = time.time()
        print(f'construct scheme time {t1 - t0}')
        print(bonds_final) # [810, 811, 802, 803, 794, 795, 796, 797, 826, 827]
        print([bonds_tensor_original[bond] for bond in bonds_final]) # [{393}, {393}, {389}, {389}, {385}, {385}, {386}, {386}, {401}, {401}]
        permute_idx = []

        # [810, 802, 803, 794, 795, 796, 797, 826, 827]
        # [{393, 433}, {424, 389}, {425, 389}, {385, 414}, {385, 415}, {416, 386}, {417, 386}, {401, 452}, {401, 453}]
        for bond in bonds_final:
            x, y = bonds_tensor_original[bond]
            if x in final_qubits:
                y = x
            elif y in final_qubits:
                y = y
            else:
                raise ValueError('Wrong final qubits')
            permute_idx.append(sorted(exclude_final_qubits).index(y))
        print(permute_idx)

    tensors = [tensor.to(device) for tensor in tensors_save]
    collect_tensor = torch.zeros([len(bitstrings_sorted)] + [2] * len(exclude_final_qubits), dtype=dtype, device=device)
    min_time = 10**8

    for s in range(1):
        # torch.cuda.empty_cache()
        print(f'subtasks {s}:')
        torch.cuda.synchronize(device)
        t0 = time.time()
        configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
        sliced_tensors = tensors.copy()
        for x in range(len(slicing_edges)):
            k, l = slicing_edges[x]
            idxk_l, idxl_k = slicing_indices[(k, l)]
            sliced_tensors[k] = sliced_tensors[k].select(idxk_l, configs[x]).clone()
            sliced_tensors[l] = sliced_tensors[l].select(idxl_k, configs[x]).clone()
        collect_tensor += tensor_contraction_sparse(sliced_tensors, scheme, use_cutensor=False)
        torch.cuda.synchronize(device)
        t1 = time.time()
        print(t1-t0)
        if (t1-t0) < min_time:
            min_time = (t1-t0)

    computational_ability = 156 * 10 ** 12

    efficiency = 8 * 10 ** tc / (computational_ability * min_time)
    print(f'subtask running time {min_time:.4f}s , overall running time {2**len(slicing_edges) * min_time/3600} hours, corresponding GPU efficiency {efficiency:.4f}')
    print('-'*20)
    # print(bitstrings_sorted[:10])
    idx = bitstrings_sorted.index('00000000000100000000001000110010001001000110')
    print(collect_tensor[idx, 1, 1, 1, 1, 0, 0, 0, 1, 1])

if __name__ == '__main__':
    detect_open_qubits(53, 20, 'ABCDCDAB', 'cuda:0', 35, 3000000, select=0) # sc_target = 30 means 24G memory