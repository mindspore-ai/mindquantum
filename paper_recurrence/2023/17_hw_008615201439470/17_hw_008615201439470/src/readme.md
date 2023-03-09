# 请将你对论文复现的源代码放置于当下src文件夹内。
import numpy as np 
import scipy
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import mindspore as ms
from mindquantum.core.gates import X, Y, Z, RX, RY, RZ, H
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import inner_product, Simulator
from mindquantum.core.gates import Measure
from mindquantum.core.operators import Hamiltonian    
from mindquantum.core.operators import QubitOperator 
from numpy import arcsin

#代码中的一些常数，为保证准确，需要导入原始哈密顿量和微扰
alpha=-0.24497866312686586;
cor_theta=2.7741246588994954;
data0 = sio.loadmat(os.getcwd()+'/n_17.mat');
H0=data0['A'];
e0,v0=np.linalg.eig(H0);
Vex=data0['B'];
Egs=np.min(e0);
Em1=-1;
E0=0;
E1=1;
E2=2;
Eh=np.max(e0);
C=Em1-Egs;
theta0=2*arcsin(C/(E0-Egs));
theta2=2*arcsin(C/(E2-Egs))-theta0;
thetags=-theta0-theta2;
thetam1=2*arcsin(C/(Em1-Egs))-theta0;
theta1=2*arcsin(C/(E1-Egs))-theta0;
thetah=2*arcsin(C/(Eh-Egs))-theta0-thetam1;

##################Udis 线路
Udis=Circuit();
############这一段是修正的线路
Udis+=X(1,0);
Udis+=X(2,0);
Udis+=X(3,0);
Udis+=X(1);
Udis+=X(2);
Udis+=X(3);
Udis+=X(0,{1,2,3});
Udis+=RY(-cor_theta).on(0);
Udis+=X(0,{1,2,3});
Udis+=RY(cor_theta).on(0);
Udis+=X(1);
Udis+=X(2);
Udis+=X(3);
Udis+=X(1,0);
Udis+=X(2,0);
Udis+=X(3,0);
###############从这里开始就是论文中的线路了
Udis+=X(0);
Udis+=X(2);
Udis+=X(3);
Udis+=X(0,3);
Udis+=X(3);
Udis+=X(1,3);
Udis+=X(2,3);
Udis+=X(3,{0,1,2});
Udis+=RY(-np.pi/4).on(3);
Udis+=X(3,{0,1,2});
Udis+=RY(np.pi/4).on(3);
Udis+=X(3,{0,1,2});
Udis+=X(2,3);
Udis+=X(1,3);
Udis+=X(3);
Udis+=X(0,3);
Udis+=X(3);
Udis+=X(3);
Udis+=X(1,3);
Udis+=X(3);
Udis+=X(0,3);
Udis+=X(2,3);
Udis+=X(3);
Udis+=X(3,{0,1,2});
Udis+=RY(-alpha).on(3);
Udis+=X(3,{0,1,2});
Udis+=RY(alpha).on(3);
Udis+=X(3,{0,1,2});
Udis+=X(3);
Udis+=X(2,3);
Udis+=X(0,3);
Udis+=X(3);
Udis+=X(1,3);
Udis+=X(3);
Udis+=X(1,0);
Udis+=H(0,1);
Udis+=X(1,0);
Udis+=Z(1,0);
Udis+=X(3,2);
Udis+=H(2,3);
Udis+=X(3,2);
Udis+=Z(3,2);

############Udis_dag 线路
Udis_dag=Circuit();
Udis_dag+=Z(3,2);
Udis_dag+=X(3,2);
Udis_dag+=H(2,3);
Udis_dag+=X(3,2);
Udis_dag+=Z(1,0);
Udis_dag+=X(1,0);
Udis_dag+=H(0,1);
Udis_dag+=X(1,0);
Udis_dag+=X(3);
Udis_dag+=X(1,3);
Udis_dag+=X(3);
Udis_dag+=X(0,3);
Udis_dag+=X(2,3);
Udis_dag+=X(3);
Udis_dag+=X(3,{0,1,2});
Udis_dag+=RY(-alpha).on(3);
Udis_dag+=X(3,{0,1,2});
Udis_dag+=RY(alpha).on(3);
Udis_dag+=X(3,{0,1,2});
Udis_dag+=X(3);
Udis_dag+=X(2,3);
Udis_dag+=X(0,3);
Udis_dag+=X(3);
Udis_dag+=X(1,3);
Udis_dag+=X(3);
Udis_dag+=X(3);
Udis_dag+=X(0,3);
Udis_dag+=X(3);
Udis_dag+=X(1,3);
Udis_dag+=X(2,3);
Udis_dag+=X(3,{0,1,2});
Udis_dag+=RY(-np.pi/4).on(3);
Udis_dag+=X(3,{0,1,2});
Udis_dag+=RY(np.pi/4).on(3);
Udis_dag+=X(3,{0,1,2});
Udis_dag+=X(2,3);
Udis_dag+=X(1,3);
Udis_dag+=X(3);
Udis_dag+=X(0,3);
Udis_dag+=X(3);
Udis_dag+=X(2);
Udis_dag+=X(0);
############这一段是修正的线路
Udis_dag+=X(1,0);
Udis_dag+=X(2,0);
Udis_dag+=X(3,0);
Udis_dag+=X(1);
Udis_dag+=X(2);
Udis_dag+=X(3);
Udis_dag+=RY(-cor_theta).on(0);
Udis_dag+=X(0,{1,2,3});
Udis_dag+=RY(cor_theta).on(0);
Udis_dag+=X(0,{1,2,3});
Udis_dag+=X(1);
Udis_dag+=X(2);
Udis_dag+=X(3);
Udis_dag+=X(1,0);
Udis_dag+=X(2,0);
Udis_dag+=X(3,0);

#######e^(i*lambda*V)线路
V=Circuit();
V+=X(1,0);
V+=RZ('lambdam2').on(1);
V+=X(1,0);
V+=X(3,0);
V+=RZ('lambdam2').on(3);
V+=X(3,0);
V+=X(1,2);
V+=RZ('lambdam2').on(1);
V+=X(1,2);
V+=X(1,3);
V+=RZ('lambdam2').on(1);
V+=X(1,3);

########exp(i lambda V/2)-exp(-i lambda V/2)线路,引入一个新的辅助比特
Vpr=Circuit();
Vpr+=X(5);
Vpr+=H(5);
Vpr+=X(1,0);
Vpr+=RZ('lambdam').on(1);
Vpr+=X(1,0);
Vpr+=X(3,0);
Vpr+=RZ('lambdam').on(3);
Vpr+=X(3,0);
Vpr+=X(1,2);
Vpr+=RZ('lambdam').on(1);
Vpr+=X(1,2);
Vpr+=X(1,3);
Vpr+=RZ('lambdam').on(1);
Vpr+=X(1,3);
Vpr+=X(1,{0,5});
Vpr+=X(1,5);
Vpr+=RZ('lambdam').on(1);
Vpr+=X(1,5);
Vpr+=RZ('lambda').on(1);
Vpr+=X(1,{0,5});
Vpr+=X(3,{0,5});
Vpr+=X(3,5);
Vpr+=RZ('lambdam').on(3);
Vpr+=X(3,5);
Vpr+=RZ('lambda').on(3);
Vpr+=X(3,{0,5});
Vpr+=X(1,{2,5});
Vpr+=X(1,5);
Vpr+=RZ('lambdam').on(1);
Vpr+=X(1,5);
Vpr+=RZ('lambda').on(1);
Vpr+=X(1,{2,5});
Vpr+=X(1,{3,5});
Vpr+=X(1,5);
Vpr+=RZ('lambdam').on(1);
Vpr+=X(1,5);
Vpr+=RZ('lambda').on(1);
Vpr+=X(1,{3,5});
Vpr+=H(5);
Vpr+=X(5);

#U_e线路,相较于原文，不需要引入辅助比特q'
Ue=Circuit();
Ue+=RY(theta0).on(4);
Ue+=X(0);
Ue+=X(2);
Ue+=RY(theta2/2).on(4);
Ue+=X(4,{0,2});
Ue+=RY(-theta2/2).on(4);
Ue+=X(4,{0,2});
Ue+=X(0);
Ue+=X(2);
Ue+=X(0);
Ue+=X(1);
Ue+=X(2);
Ue+=X(3);
Ue+=RY(thetags/2).on(4);
Ue+=X(4,{0,1,2,3});
Ue+=RY(-thetags/2).on(4);
Ue+=X(4,{0,1,2,3});
Ue+=X(0);
Ue+=X(1);
Ue+=X(2);
Ue+=X(3);
Ue+=RY(thetam1/2).on(4);
Ue+=X(4,{0,2});
Ue+=RY(-thetam1/2).on(4);
Ue+=X(4,{0,2});
Ue+=X(1);
Ue+=X(3);
Ue+=RY(-thetam1/2).on(4);
Ue+=X(4,{0,1,2,3});
Ue+=RY(thetam1/2).on(4);
Ue+=X(4,{0,1,2,3});
Ue+=X(1);
Ue+=X(3);
Ue+=RY(thetah/2).on(4);
Ue+=X(4,{0,1,2,3});
Ue+=RY(-thetah/2).on(4);
Ue+=X(4,{0,1,2,3});
Ue+=X(2);
Ue+=X(3);
Ue+=RY(theta1/2).on(4);
Ue+=X(4,{0,2,3});
Ue+=RY(-theta1/2).on(4);
Ue+=X(4,{0,2,3});
Ue+=X(2);
Ue+=X(3);
Ue+=X(0);
Ue+=X(1);
Ue+=X(3);
Ue+=RY(theta1/2).on(4);
Ue+=X(4,{0,1,2,3});
Ue+=RY(-theta1/2).on(4);
Ue+=X(4,{0,1,2,3});
Ue+=X(0);
Ue+=X(1);
Ue+=X(3);

####图3a)，一阶能量修正，与经典算法作比较
_lambda=0.0;
Fst_energy=np.array([]);
list_of_lambda=np.array([]);
while _lambda<1:
    bra_simulator = Simulator('projectq', 4);
    ket_simulator = Simulator('projectq', 4);
    bra_simulator.apply_circuit(Udis);
    ket_simulator.apply_circuit(Udis+V.apply_value({'lambdam': -_lambda,'lambdam2': -2*_lambda,'lambda': _lambda}));
    Fst_energy=np.append(Fst_energy,inner_product(bra_simulator, ket_simulator).imag);
    list_of_lambda=np.append(list_of_lambda,_lambda);
    _lambda+=0.04;
plt.scatter(list_of_lambda,Fst_energy,c='m',marker='D');

Vpert=QubitOperator('Z0 Z1')+QubitOperator('Z0 Z3')+QubitOperator('Z1 Z2')+QubitOperator('Z1 Z3');
Vpert=Hamiltonian(Vpert);
pur_simulator = Simulator('projectq', 4);
pur_simulator.apply_circuit(Udis);
_y=(pur_simulator.get_expectation(Vpert)).real*list_of_lambda;
plt.plot(list_of_lambda,_y,'--',color='r')

####图3b)，一阶本征态修正，与经典算法作比较
_lambda=0.02;
Fst_eig_real=np.array([]);
Fst_eig_imag=np.array([]);
list_of_lambda=np.array([]);
while _lambda<0.8:
    sim1=Simulator('projectq', 5);
    sim1.apply_circuit(Udis+V.apply_value({'lambdam': -_lambda,'lambdam2': -2*_lambda,'lambda': _lambda})+Udis_dag+Ue);
    state1=sim1.get_qs();
    prj1=0.5*QubitOperator('Z4')+0.5*QubitOperator('');
    prj1=Hamiltonian(prj1);
    prob1=1-sim1.get_expectation(prj1);
    Cprm=C/np.power(prob1,0.5);
    cir_meas1=Circuit();
    cir_meas1+=Measure().on(4);
    while True :
        sim1=Simulator('projectq', 5);
        sim1.set_qs(state1);
        sim1.apply_circuit(cir_meas1);
        res=sim1.get_expectation(prj1)
        if res.real<0.1:
            break
    phi=sim1.get_qs()/Cprm;
    Fst_eig_real=np.append(Fst_eig_real,phi[31].real);
    Fst_eig_imag=np.append(Fst_eig_imag,phi[31].imag);
    list_of_lambda=np.append(list_of_lambda,_lambda);
    _lambda+=0.04;
plt.scatter(list_of_lambda,Fst_eig_real,c='',marker='o',edgecolors='r');
plt.scatter(list_of_lambda,Fst_eig_imag,c='b',marker='^');

pert_sim=Simulator('projectq', 4);
pert_sim.apply_circuit(Udis);
phi_gs=pert_sim.get_qs();
pert_sim=Simulator('projectq', 4);
pert_sim.apply_gate(X(0));
pert_sim.apply_gate(X(1));
pert_sim.apply_gate(X(2));
pert_sim.apply_gate(X(3));
pert_sim.apply_circuit(Udis);
phi_h=pert_sim.get_qs();
cor_phi=np.dot(phi_gs,Vex);
cor_phi=np.dot(cor_phi,np.transpose(phi_h));
_y_real=cor_phi.real/(Eh-Egs)*list_of_lambda;
_y_imag=cor_phi.imag/(Eh-Egs)*list_of_lambda;
plt.plot(list_of_lambda,_y_real,color='r');
plt.plot(list_of_lambda,_y_imag,'--',color='g');

####图3c)，一阶本征态修正，采用改进的量子算法，与经典算法作比较
_lambda=0.02;
Fst_eig_real2=np.array([]);
Fst_eig_imag2=np.array([]);
Sec_energy=np.array([]);
list_of_lambda=np.array([]);

prj1=0.5*QubitOperator('Z4')+0.5*QubitOperator('');
prj1=Hamiltonian(prj1);
prj2=0.5*QubitOperator('Z5')+0.5*QubitOperator('');
prj2=Hamiltonian(prj2);

while _lambda<0.8:
    sim2=Simulator('projectq', 6);
    sim2.apply_circuit(Udis+Vpr.apply_value({'lambdam': -_lambda,'lambdam2': -2*_lambda,'lambda': _lambda})+Udis_dag);
    state2=sim2.get_qs();
    prob2=1-sim2.get_expectation(prj2);
    scale=np.power(prob2,0.5)*2;
    
    cir_meas1=Circuit();
    cir_meas1+=Measure().on(5);
    while True :
        sim3=Simulator('projectq', 6);
        sim3.set_qs(state2);
        sim3.apply_circuit(cir_meas1);
        res=sim3.get_expectation(prj2);    
        if  res.real<0.1:
            break
    state3=sim3.get_qs();
    sim3.apply_circuit(Ue);
    state4=sim3.get_qs();
    prob1=1-sim3.get_expectation(prj1);
    Cprm=C/np.power(prob1,0.5);
    cir_meas2=Circuit();
    cir_meas2+=Measure().on(4);
    while True :
        sim4=Simulator('projectq', 6);
        sim4.set_qs(state4);
        sim4.apply_circuit(cir_meas2);
        res=sim4.get_expectation(prj1);    
        if  res.real<0.1:
            break
    state5=sim4.get_qs();
    phi=sim4.get_qs()/Cprm*scale;
    Fst_eig_real2=np.append(Fst_eig_real2,phi[63].real);
    Fst_eig_imag2=np.append(Fst_eig_imag2,phi[63].imag);
    
    bra_simulator = Simulator('projectq', 6);
    ket_simulator = Simulator('projectq', 6);
    bra_simulator.set_qs(state3);
    ket_simulator.set_qs(state5);
    bra_simulator.apply_gate(X(4));
    cur_energy=inner_product(bra_simulator, ket_simulator)/Cprm*scale*scale;
    Sec_energy=np.append(Sec_energy,cur_energy.real);
                         
    list_of_lambda=np.append(list_of_lambda,_lambda);
    _lambda+=0.04;
plt.scatter(list_of_lambda,Fst_eig_real2,c='',marker='o',edgecolors='r');
plt.scatter(list_of_lambda,Fst_eig_imag2,c='b',marker='^');

_y_real=cor_phi.real/(Eh-Egs)*list_of_lambda;
_y_imag=cor_phi.imag/(Eh-Egs)*list_of_lambda;
plt.plot(list_of_lambda,_y_real,color='r');
plt.plot(list_of_lambda,_y_imag,'--',color='g');

####图3d)，二阶能量修正，采用改进的量子算法，与经典算法作比较
plt.scatter(list_of_lambda,Sec_energy,c='b',marker='^');
_y_energy=cor_phi.real*cor_phi.real/(Eh-Egs)*list_of_lambda**2;
plt.plot(list_of_lambda,_y_energy,color='r');