# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Stabilizer tableau class. A binary table represents a stabilizer circuit.
Reference:
    S. Aaronson, D. Gottesman, Improved Simulation of Stabilizer Circuits,
        Phys. Rev. A 70, 052328 (2004).  arXiv:quant-ph/0406196
"""
import numpy as np
import time
from clifford import Clifford
from clifford import clifford_to_id, id_to_clifford, get_initial_id


def search_pure_Clifford(num_qubits=2):
    # tree search Clifford
    if num_qubits>2: print("WARNING: only num_qubits=2 works well !!!")
    n = num_qubits
    c2 = Clifford(num_qubits=n)
    tid_list = []
    cli_list = []
    tid, _ = clifford_to_id(c2)
    tid_list.append(tid)
    cli_list.append(c2.copy())

    #count = 1
    #iturn = 0 
    is_alive = 1
    ncli_old = -1
    while is_alive:
        is_alive = 0
        #iturn += 1
        #print("turn:", iturn)
        ncli = len(cli_list)
        for ii in range(ncli_old+1, ncli):
            for igate in range(-n,n*n):
                icli_copy = cli_list[ii].copy()
                if igate<0:
                    icli_copy.PhaseGate(-igate-1)
                else:
                    iq = igate % n
                    jq = igate //n
                    if iq==jq:  icli_copy.Hadamard(iq)
                    else:       icli_copy.CNOT(iq,jq)
                icli_tid,_ = clifford_to_id(icli_copy)
                # different tableau
                if not np.isin(icli_tid, tid_list):
                    is_alive = 1   # found a new Clifford, keep alive
                    #count += 1
                    tid_list.append(icli_tid)
                    cli_list.append(icli_copy.copy())
        ncli_old = ncli

    #print(tid_list)
    #print("num. of pure Clifford:", count)
    del cli_list

    return tid_list


def reverse_tid_list(tid_list):
    tid2ind  = {}
    for ii in range(len(tid_list)): tid2ind[tid_list[ii]] = ii
    return tid2ind

def get_rr_table(num_qubits=2):
    # lookup table for G(I,r1)G(I,r2) = G(I,r3)
    id0 = get_initial_id(num_qubits=num_qubits)
    rr_table = np.zeros((2**(2*num_qubits), 2**(2*num_qubits), ), dtype=np.int16)
    for ir in range(2**(2*num_qubits)):
        for jr in range(2**(2*num_qubits)):
            cli1 = id_to_clifford(tableid=id0, phaseid=ir, num_qubits=num_qubits)
            cli2 = id_to_clifford(tableid=id0, phaseid=jr, num_qubits=num_qubits)
            cli = cli1*cli2
            _ , pid = clifford_to_id(cli)
            rr_table[ir,jr] = pid
    return rr_table


def get_rT_table(tid_list, tid2ind=None, num_qubits=2):
    # lookup table for G(I,r1)G(T2,0) = G(T3,r3)
    if tid2ind is None: tid2ind = reverse_tid_list(tid_list)
    id0 = get_initial_id(num_qubits=num_qubits)
    rT_table = np.zeros((2**(2*num_qubits), len(tid_list), 2), dtype=np.int16)
    for ir in range(2**(2*num_qubits)):
        for jT in range(len(tid_list)):
            cli1 = id_to_clifford(tableid=id0, phaseid=ir, num_qubits=num_qubits)
            cli2 = id_to_clifford(tableid=tid_list[jT], phaseid=0, num_qubits=num_qubits)
            cli = cli1*cli2
            tid , pid = clifford_to_id(cli)
            rT_table[ir,jT,0] = tid2ind[tid]
            rT_table[ir,jT,1] = pid
    return rT_table

def check_Tr_identical(tid_list, tid2ind=None, num_qubits=2):
    # check relation G(T,r) = G(T,0)G(I,r)
    if tid2ind is None: tid2ind = reverse_tid_list(tid_list)
    id0 = get_initial_id(num_qubits=num_qubits)
    for ir in range(2**(2*num_qubits)):
        for jT in range(len(tid_list)):
            cli1 = id_to_clifford(tableid=id0, phaseid=ir, num_qubits=num_qubits)
            cli2 = id_to_clifford(tableid=tid_list[jT], phaseid=0, num_qubits=num_qubits)
            cli = cli2*cli1
            tid , pid = clifford_to_id(cli)
            try:
                assert tid==tid_list[jT]
                assert pid==ir
            except:
                print("check Tr identical G(T,r) = G(T,0)G(I,r): FAIL")
    print("check Tr identical G(T,r) = G(T,0)G(I,r): PASS")


def get_TT_table(tid_list, tid2ind=None, num_qubits=2):
    # lookup table for G(T1,0)G(T2,0) = G(T3,r3)
    print('be patient! TT table may take 6 minutes.')
    if tid2ind is None: tid2ind = reverse_tid_list(tid_list)
    TT_table = np.zeros((len(tid_list), len(tid_list), 2), dtype=np.int16)
    for iT in range(len(tid_list)):
        if iT%20==0: print("run TT %d of %d"%(iT+1,len(tid_list)))
        for jT in range(len(tid_list)):
            cli1 = id_to_clifford(tableid=tid_list[iT], phaseid=0, num_qubits=num_qubits)
            cli2 = id_to_clifford(tableid=tid_list[jT], phaseid=0, num_qubits=num_qubits)
            cli = cli1*cli2
            tid , pid = clifford_to_id(cli)
            TT_table[iT,jT,0] = tid2ind[tid]
            TT_table[iT,jT,1] = pid
    return TT_table

def get_TrTr_table(tid_list,num_qubits=2,
        rr_table=None, rT_table=None, TT_table=None):
    # get full multiplication table TrTr
    # full TrTr, G(T1,r1)G(T2,r2) = G(T3,r3)
    #       G(T1,r1)G(T2,r2) = G(T1,0)G(I,r1)G(T2,0)G(I,r2)
    #                        = G(T1,0)  G(T2',r1')  G(I,r2)
    #                        = G(T1,0)G(T2',0)G(I,r1')G(I,r2)
    #                        =   G(T3,r3)  G(I,r1')G(I,r2)
    #                        = G(T3,0)G(I,r3)G(I,r1')G(I,r2)
    #                        = G(T3,0)G(I,r3') = G(T3,r3')
    if (rr_table is None) or (rT_table is None) or (rT_table is None):
        raise ValueError("Error! rr/rT/TT table should be provided.")
    print('be patient! TrTr table may take 5 minutes.')
    TrTr_mul_table = np.zeros(( len(tid_list), 2**(2*num_qubits), \
                                len(tid_list), 2**(2*num_qubits), 2), dtype=np.int16)
    for iT in range(len(tid_list)):
        if iT%20==0: print("run TrTr %d of %d"%(iT,len(tid_list)))
        for ir in range(2**(2*num_qubits)):
            for jT in range(len(tid_list)):
                for jr in range(2**(2*num_qubits)):
                    r1T2  = rT_table[ir,jT,:]                   # G(I,r1)G(T2,0) => G(T2',r1')
                    T1T2p = TT_table[iT,r1T2[0],:]              # G(T1,0)G(T2',0) => G(T3,r3)
                    r1pr2 = rr_table[r1T2[1],jr]                # G(I,r1')G(I,r2) => G(I,r2')
                    r3r2p = rr_table[T1T2p[1],r1pr2]            # G(I,r3)G(I,r2') => G(I,r3')
                    TrTr_mul_table[iT,ir,jT,jr,0] = T1T2p[0]
                    TrTr_mul_table[iT,ir,jT,jr,0] = r3r2p
    return TrTr_mul_table


def mul_by_table(cli1, cli2, tid_list=None, TrTr_mul_table=None):
    if TrTr_mul_table is None:
        raise ValueError("Error! TrTr table should be provided.")
    if tid_list is None:
        raise ValueError("Error! pure Clifford tid_list should be provided.")
    assert cli1.num_qubits==cli2.num_qubits

    tid2ind = reverse_tid_list(tid_list)
    tid1, pid1 = clifford_to_id(cli1)
    tid2, pid2 = clifford_to_id(cli2)
    res = TrTr_mul_table[tid2ind[tid1],pid1,tid2ind[tid2],pid2,:]
    tid3, pid3 = tid_list[res[0]], res[1]
    cli3 = id_to_clifford(tableid=tid3, phaseid=pid3, num_qubits=cli1.num_qubits)

    return cli3.copy()



if __name__=="__main__":

    num_qubits = 2

    ######  step01: search pure Clifford(r=0)  ######
    # for n=2, #Clifford = 11520, see `Hadamard-free circuits expose the structure of the Clifford group`
    print('be patient! search pure Clifford may take 1 minutes.')
    tid_list = search_pure_Clifford(num_qubits=num_qubits)  #pure Clifford, which tableau phase r=0
    print("num. of pure Clifford(n=2):", len(tid_list))
    print("num. of all Clifford(n=2):", len(tid_list) * 2**(2*num_qubits)) # all Clifford, any r


    ######  step02: lookup table of rr/rT/TT   ######
    try:
        f=open('TT.npy'); f.close()
        print("find TT.npy")
    except FileNotFoundError:
        rr_table = get_rr_table(num_qubits=2)                       # rr, G(I,r1)G(I,r2) = G(I,r3)
        rT_table = get_rT_table(tid_list, num_qubits=num_qubits)    # rT, G(I,r1)G(T2,0) = G(T3,r3)
        check_Tr_identical(tid_list, num_qubits=num_qubits)         # Tr, check G(T,r) = G(T,0)G(I,r)
        TT_table = get_TT_table(tid_list, num_qubits=num_qubits)    # TT, G(T1,0)G(T2,0) = G(T3,r3)
        # save rr/rT/TT
        TT_dict = {}
        TT_dict['tid_list'] = np.array(tid_list, dtype=np.int64)
        TT_dict['rr'] = rr_table
        TT_dict['rT'] = rT_table
        TT_dict['TT'] = TT_table
        np.save('TT.npy', TT_dict)
    
    TT_dict = np.load('TT.npy', allow_pickle=True).item()
    print(TT_dict.keys())
    tid_list = TT_dict['tid_list']
    rr_table = TT_dict['rr']
    rT_table = TT_dict['rT']
    TT_table = TT_dict['TT']


    ######  step03: multiplication table TrTr  ######
    # full TrTr, G(T1,r1)G(T2,r2) = G(T3,r3)
    #           TrTr => TTrr => Trrr => Tr
    try:
        f=open('TrTr.npy'); f.close()
        print("find TrTr.npy")
    except FileNotFoundError:
        TrTr_mul_table = get_TrTr_table(tid_list, num_qubits=num_qubits,
                            rr_table=rr_table, rT_table=rT_table, TT_table=TT_table)
        TrTr_dict = {}
        TrTr_dict['tid_list'] = np.array(tid_list, dtype=np.int64)
        TrTr_dict['TrTr'] = TrTr_mul_table
        np.save('TrTr.npy', TrTr_dict)

    TrTr_dict = np.load('TrTr.npy', allow_pickle=True).item()
    tid_list = TrTr_dict['tid_list']
    TrTr_mul_table = TrTr_dict['TrTr']


    ######  step04: test random multi  ######
    print('be patient! random multi test may take 1 minutes.')
    num_loop = 10000
    # test fast multi by multiplication table TrTr
    t_start = time.time()
    for ii in range(num_loop):
        ind1 = np.random.randint(len(tid_list))
        pid1 = np.random.randint(2**(2*num_qubits))
        ind2 = np.random.randint(len(tid_list))
        pid2 = np.random.randint(2**(2*num_qubits))
        cli1 = id_to_clifford(tableid=tid_list[ind1], phaseid=pid1, num_qubits=num_qubits)
        cli2 = id_to_clifford(tableid=tid_list[ind1], phaseid=pid2, num_qubits=num_qubits)
        cli3 = mul_by_table(cli1, cli2, tid_list=tid_list, TrTr_mul_table=TrTr_mul_table)
    t_end   = time.time()
    print("fast %d multi(n=2) takes %.5f s, avg %.5f s"
                %(num_loop, t_end - t_start, (t_end-t_start)/num_loop))
    
    # test direct multi  C1*C2
    t_start2 = time.time()
    for ii in range(num_loop):
        ind1 = np.random.randint(len(tid_list))
        pid1 = np.random.randint(2**(2*num_qubits))
        ind2 = np.random.randint(len(tid_list))
        pid2 = np.random.randint(2**(2*num_qubits))
        cli1 = id_to_clifford(tableid=tid_list[ind1], phaseid=pid1, num_qubits=num_qubits)
        cli2 = id_to_clifford(tableid=tid_list[ind1], phaseid=pid2, num_qubits=num_qubits)
        cli3 = cli1*cli2
    t_end2   = time.time()
    print("direct %d multi(n=2) takes %.5f s, avg %.5f s"
                %(num_loop, t_end2-t_start2, (t_end2-t_start2)/num_loop))


