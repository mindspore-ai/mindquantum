import matplotlib.pyplot as plt
import os

def show_plt_p(p):
    # 使用当前目录下的 statistic 文件作图
    filename = os.path.dirname(os.path.normpath(__file__))
    filename += os.path.sep + "statistic_" + str(p) + ".txt"
    stat = []
    with open(filename, "r") as file:
        data_list = file.readlines()
        for l in data_list:
            l = l.strip('] [\n')
            l = l.split(",")
            stat.append([float(x) for x in l])
    x = [_/6 for _ in range(1, 60, 2)]
    y = stat[0]
    yerr = [_/p for _ in stat[1]]
    plt.errorbar(x[:len(y)], y, yerr = yerr, fmt='s', label=f"p={p}")

def show_all_p():
    show_plt_p(15)
    show_plt_p(25)
    show_plt_p(35)
    plt.legend()
    plt.show()
    
def show_plt_pa(p):
    filename = os.path.dirname(os.path.normpath(__file__))
    filename += os.path.sep + "statistic2_" + str(p) + ".txt"
    stat = []
    with open(filename, "r") as file:
        data_list = file.readlines()
        for l in data_list:
            l = l.strip('] [\n')
            l = l.split(",")
            stat.append([float(x) for x in l])
    x = [_/6 for _ in range(1, 60, 2)]
    y = stat[0]
    yerr = [_/p for _ in stat[1]]
    plt.errorbar(x[:len(y)], y, yerr = yerr, fmt='s', label=f"p={p}")

def show_all_pa():
    show_plt_pa(15)
    show_plt_pa(25)
    show_plt_pa(35)
    plt.legend()
    plt.show()
    
def show_plt_conv(p):
    filename = os.path.dirname(os.path.normpath(__file__))
    filename += os.path.sep + "statistic3_" + str(p) + ".txt"
    stat = []
    with open(filename, "r") as file:
        data_list = file.readlines()
        for l in data_list:
            l = l.strip('] [\n')
            l = l.split(",")
            stat.append([float(x) for x in l])
    x = [_/6 for _ in range(1, 60, 8)]
    y = stat[0]
    yerr = [_/p for _ in stat[1]]
    plt.errorbar(x[:len(y)], y, yerr = yerr, fmt='s', label=f"p={p}")

def show_all_conv():
    show_plt_conv(15)
    show_plt_conv(25)
    show_plt_conv(35)
    plt.legend()
    plt.show()