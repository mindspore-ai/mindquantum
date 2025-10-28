import os
import numpy as np
import tensorflow as tf
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

# 设置 NumPy 打印选项
np.set_printoptions(threshold=np.inf)  # threshold=np.inf 表示不省略任何内容

# 数据集路径
base_path = r'demo_data\{}.json'

def calc_energy(x,H_ragged,coeff_high_order,coeff_one_order):
    # x坍缩为±1
    x_sign = tf.sign(x)
    # 计算高阶项的多比特乘法：σi×σj×σk×σl
    poly_term = tf.reduce_prod(tf.gather(x_sign, H_ragged, axis=-1, batch_dims=0), axis=-1)
    # 计算总能量
    energy = tf.reduce_sum(poly_term * coeff_high_order[None, :], axis=-1) + tf.reduce_sum(
        x_sign * coeff_one_order[None, :], axis=-1)
    return -energy

# 通过with tf.device('/CPU:0'):或with tf.device('/GPU:0'):来选择CPU和GPU之间的切换
with tf.device('/CPU:0'):
    file_num = 0
    while True:
        file_path = base_path.format(file_num)
        if not os.path.exists(file_path):
            break
        print(f"processing:{file_path}")

        data = json.load(open(file_path, 'r'))
        J = data['J']
        coeff = data['coeff']
        J_H = [row for row in J if len(row) != 1]  # list 高阶相互作用关系

        num_spins = sum(1 for row in J if len(row) == 1)
        high_order_term = []

        for index, sublist in enumerate(J_H):
            for num in sublist:
                high_order_term.append([index, num])

        high_order_term = np.array(high_order_term)
        coeff_high_order = tf.ragged.constant(coeff[:len(J_H)], dtype=np.float32)
        coeff_one_order = tf.ragged.constant(coeff[-num_spins:], dtype=np.float32)

        rows, cols = high_order_term[:, 0], high_order_term[:, 1]
        rows = rows.astype(np.int64)
        cols = cols.astype(np.int64)
        H_ragged = tf.RaggedTensor.from_value_rowids(cols, rows)

        # 高阶项中总共有32个四阶项，由于计算σi翻转前后哈密顿量变化△H（σi）（即局部场）仅需计算与σi有关的项，于是下列代码用于统计带有σi的项的编号（0-31）
        sortids = tf.argsort(cols)
        cols = tf.gather(cols, sortids)
        rows = tf.gather(rows, sortids)
        H_col_ragged = tf.RaggedTensor.from_value_rowids(rows, cols)
        batch_size = 100


        x = tf.random.normal(mean = 0,stddev=10**-4,shape = (batch_size, num_spins))
        y = tf.random.normal(mean = 0,stddev=10**-4,shape = (batch_size, num_spins))
        e = tf.ones((batch_size, num_spins))
        max_iters = 100
        xi = np.sqrt(2 * num_spins / np.sum(coeff_high_order**2))
        dt = 0.075
        Tr = int(0.9 * max_iters)
        Tp = max_iters - Tr
        delta = 1
        p = np.hstack([np.linspace(-0.5, 1, Tr), np.ones(Tp)])
        alpha = 1.0
        beta = 0.15
        start = time.time()
        for i in range(max_iters):
            x_sign = tf.sign(x)
            # 计算高阶项的多比特乘法：σi×σj×σk:根据H_ragged的分类，通过tf.gather来取出x_sign中对应的值σi、σj、σk，最后通过tf.reduce_prod来将取到的对应值相乘σi×σj×σk
            poly_term = tf.reduce_prod(tf.gather(x,H_ragged,axis=-1,batch_dims=0),axis=-1)
            # 计算与σi有关的项：∑Jijkl×σi×σj×σk×σl:根据H_col_ragged的分类，通过tf.gather来取出poly_term * coeff_high_order中与σi有关的值Jijkl×σi×σj×σk×σl，最后通过tf.reduce_sum来将取到的对应值相加得到∑Jijkl×σi×σj×σk×σl
            poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * coeff_high_order[None,:],H_col_ragged,axis=-1,batch_dims=0),axis=-1)

            grad_val = poly_sum_bit / x + coeff_one_order[None,:]

            x = x + (-x**3 + (p[i] - 1) * x + xi * e * grad_val)*dt
            e = e + (-beta * e * ((xi * e * grad_val) ** 2 - alpha)) * dt

            cond = tf.abs(x) > 1.5
            x = tf.where(cond,1.5 * tf.sign(x),x)
            e = tf.where(e < 0.01, 0.01, e)
            energy = calc_energy(x,H_ragged,coeff_high_order,coeff_one_order)
            opt_ind = tf.argmin(energy)
            #能量最低的解opt_x，大小为batch_size（副本数）*64
            opt_x = x[opt_ind]

        time_used = time.time() - start
        result = tf.cast(((opt_x - 1) / -2), dtype=tf.int32)
        result = tf.reshape(result, (1, num_spins))
        print("result:", result)

        print("TIME(S):", time_used)

        file_num += 1
