import sys
sys.path.append("./src")
from hiqfermion.drivers import MolecularData
from openfermionpyscf import run_pyscf
from src.main import Main,Plot

def geometry_lih(blens):
    geom = []
    for blen in blens:
        geom.append([('Li', [0, 0, 0]), ('H', [0, 0, blen])])

    return geom

def geometry_ch4(blens):
    geom = []
    for blen in blens:
        geom.append([('C', [0.0, 0.0, 0.0]),
                         ('H', [blen, blen, blen]),
                         ('H', [blen, -blen, -blen]),
                         ('H', [-blen, blen, -blen]),
                         ('H', [-blen, -blen, blen])
                         ])

    return geom

if __name__ == "__main__":
    main = Main()
    plt = Plot()

    
    molecule1 = 'LiH.hdf5'
    molecule2 = 'CH4.hdf5'

    #blen1 = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
    #blen2 = [0.4, 0.6, 0.8]
    blen1 = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
    blen2 = [0.4,0.8]

    geom1 = geometry_lih(blen1)
    geom2 = geometry_ch4(blen2)


    en1, time1 = main.run('LiH', molecule1, geom1)
    en2, time2 = main.run('CH4', molecule2, geom2)

    plt.plot('LiH', blen1, en1, time1)
    plt.plot('CH4', blen2, en2, time2)
