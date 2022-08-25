import matplotlib.pyplot as plt
import json
import numpy as np

markers_dict = dict(zip(["Hartree-Fock", "full-CI", "CCSD", "1-UpCCGSD",
                         "2-UpCCGSD", "UCCSD0", "UCCSD", "QUCC", "HEA", "ADAPT", "qubit-ADAPT", "QCC", "2-LDCA", "BRC"],
                        ['.', '|', 'x', '+', '1', '2', '3', '4', '1', '2', '3', '4', '1', 'x']))
colos_dict = dict(zip(["Hartree-Fock", "full-CI", "CCSD", "1-UpCCGSD",
                       "2-UpCCGSD", "UCCSD0", "UCCSD", "QUCC", "HEA", "ADAPT", "qubit-ADAPT", "QCC", "2-LDCA", "BRC"],
                      ['aquamarine', 'azure', 'teal', 'limegreen', 'chocolate', 'steelblue', 'magenta',
                       'violet', 'orange', 'darkseagreen', 'tomato', 'orchid', 'black', 'limegreen']))
# , "HEA", "ADAPT", "qubit-ADAPT"

font = {'family': 'serif',
        'weight': 'normal',
        'size': 16}


def draw_energies(molecular='LiH',
                  ansatzes=None,
                  xlim=None, ylim=None,
                  savefig=True, figsize=(8, 6)
                  ):
    if ansatzes is None:
        ansatzes = ["Hartree-Fock", "full-CI", "CCSD", "1-UpCCGSD",
                    "2-UpCCGSD", "HEA", "UCCSD0", "ADAPT", "qubit-ADAPT",
                    "QUCC", "BRC"]
    with open(f"mindquantum_energies_{molecular}.json") as f:
        json_data = json.load(f)
    engy = json_data["energies"]
    klen = json_data["bond_lengths"]
    plt.figure(figsize=figsize)
    for asz in ansatzes:
        plt.plot(klen, engy[asz], label=asz, linewidth=1, marker=markers_dict[asz], markersize=100)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.xticks(klen)
    plt.legend(fontsize=16)
    if savefig:
        plt.savefig(f'{molecular}_energies_{ansatzes}.png', dpi=300)
    plt.show()


def draw_errors(molecular='LiH',
                ansatzes=None,
                xlim=None, ylim=(0, 0.005),
                savefig=True, figsize=(8, 6)
                ):
    if ansatzes is None:
        ansatzes = ["Hartree-Fock", "CCSD", "1-UpCCGSD",
                    "2-UpCCGSD", "HEA", "UCCSD0", "ADAPT", "qubit-ADAPT",
                    "QUCC", "BRC"]
    with open(f"mindquantum_energies_{molecular}.json") as f:
        json_data = json.load(f)
    engy = json_data["energies"]
    klen = json_data["bond_lengths"]
    fcie = engy["full-CI"]
    plt.figure(figsize=figsize)
    for asz in ansatzes:
        plt.semilogy(klen, np.array(engy[asz]) - np.array(fcie), label=asz, linewidth=1, marker=markers_dict[asz],
                 color=colos_dict[asz], markersize=10, markeredgewidth=1.5)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend(fontsize=16)
    plt.axhline(y=0.0, color='black', linestyle='dashed')
    # plt.xticks(klen)
    plt.tick_params(labelsize=16)
    plt.xlabel("Bond Length [$\AA$]", font)
    plt.ylabel("Energy Deviation [Hartree]", font)
    plt.title(molecular, loc='left', fontsize=20, pad=12.0)
    # chemical accuracy
    plt.axhspan(-0.0016, 0.0016, color='grey', alpha=0.15, lw=0)
    if savefig:
        plt.savefig(f'{molecular}_errors_{ansatzes}.png', dpi=300)
    plt.show()


def draw_runtimes(molecular='LiH',
                  ansatzes=None,
                  xlim=None, ylim=None,
                  savefig=True, figsize=(8, 6)
                  ):
    ansatzes_legal = ["1-UpCCGSD",
                      "2-UpCCGSD", "HEA", "UCCSD0", "ADAPT", "qubit-ADAPT",
                      "QUCC", "BRC"]
    if ansatzes is None:
        ansatzes = ansatzes_legal
    with open(f"mindquantum_times_{molecular}.json") as f:
        json_data = json.load(f)
    times = json_data["times"]
    klen = json_data["bond_lengths"]
    plt.figure(figsize=figsize)
    for asz in ansatzes:
        if asz not in ansatzes_legal:
            continue
        plt.plot(klen, times[asz], label=asz, linewidth=1, marker=markers_dict[asz],
                 color=colos_dict[asz], markersize=10, markeredgewidth=1.5)
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylim(bottom=0)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend(fontsize=16)
    # plt.xticks(klen)
    plt.tick_params(labelsize=16)
    plt.xlabel("Bond Length [$\AA$]", font)
    plt.ylabel("Time [s]", font)
    plt.title(molecular, loc='left', fontsize=20, pad=12.0)
    if savefig:
        plt.savefig(f'{molecular}_runtimes_{ansatzes}.png', dpi=300)
    plt.show()


def draw_paras(molecular='LiH',
               ansatzes=None,
               xlim=None, ylim=None,
               savefig=True, figsize=(8, 6)
               ):
    ansatzes_legal = ["1-UpCCGSD",
                      "2-UpCCGSD", "HEA", "UCCSD0", "ADAPT", "qubit-ADAPT",
                      "QUCC", "BRC"]
    if ansatzes is None:
        ansatzes = ansatzes_legal
    with open(f"mindquantum_parameters_{molecular}.json") as f:
        json_data = json.load(f)
    parameters = json_data["parameters"]
    klen = json_data["bond_lengths"]
    plt.figure(figsize=figsize)
    for asz in ansatzes:
        if asz not in ansatzes_legal:
            continue
        plt.plot(klen, parameters[asz], label=asz, linewidth=1, marker=markers_dict[asz],
                 color=colos_dict[asz], markersize=10, markeredgewidth=1.5)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend(fontsize=16)
    # plt.xticks(klen)
    plt.tick_params(labelsize=16)
    plt.xlabel("Bond Length [$\AA$]", font)
    plt.ylabel("Parameter Number", font)
    plt.title(molecular, loc='left', fontsize=20, pad=12.0)
    if savefig:
        plt.savefig(f'{molecular}_parameters_{ansatzes}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    molecule = 'H4'
    # molecule = 'LiH'
    # molecule = 'BeH2'
    # molecule = 'H2O'
    # molecule = 'CH4'
    # molecule = 'N2'
    figsize = (10, 7.5)
    # ylim = (-0.002, 0.02)  # setting for error
    ylim = (-0.002, 1)  # setting for error

    # ansatzes = ["CCSD", "HEA", "ADAPT", "qubit-ADAPT", "QCC"]
    ansatzes = ["CCSD", "1-UpCCGSD", "2-UpCCGSD", "UCCSD", "UCCSD0", "QUCC", "2-LDCA", "BRC"]
    # draw_energies(molecular=molecule, figsize=figsize)
    draw_errors(molecular=molecule, figsize=figsize, ansatzes=ansatzes, ylim=ylim)
    draw_runtimes(molecular=molecule, figsize=figsize, ansatzes=ansatzes)
    draw_paras(molecular=molecule, figsize=figsize, ansatzes=ansatzes)
