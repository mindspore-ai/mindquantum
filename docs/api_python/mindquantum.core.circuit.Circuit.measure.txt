mindquantum.core.circuit.Circuit.measure(key, obj_qubit=None)

        添加测量门。

        参数:
            key (Union[int, str]): 如果`obj_qubit`为无，则`key`应该是int，表示要测量哪个量子位，
                否则，'key'应该是一个str，表示此度量门的名称。
            obj_qubit (int): 要测量的量子位。默认值：None。
        