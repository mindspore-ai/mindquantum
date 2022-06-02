.. py:method:: mindquantum.simulator.Simulator.flush()

    适用于projectq模拟器的flush门。 projectq模拟器将缓存一些门并将这些门融合到一个更大的门中，然后作用在量子态上。 flush命令使模拟器刷新当前存储的门并作用在量子状态上。