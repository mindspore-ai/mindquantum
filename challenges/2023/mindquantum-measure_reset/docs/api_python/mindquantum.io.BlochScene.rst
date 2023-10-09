mindquantum.io.BlochScene
==========================

.. py:class:: mindquantum.io.BlochScene(config=None)

    在布洛赫球中展示单量子比特的量子态。

    参数：
        - **config** (Union[dict, str]) - 布洛赫球风格配置文件。如果为 ``None`` ，系统将会使用内置的风格配置文件。除了内置格式，当前还支持 `dark` 风格。默认值： ``None`` 。

    .. py:method:: add_3d_arrow(ax, data, *args, **kwargs)

        在给定Axes中添加一个三维方向箭头。

        参数：
            - **ax** (mpl_toolkits.mplot3d.axes3d.Axes3D) - 想要添加三维箭头的三维Axes。
            - **data** (list) - 具有六个元素的列表，其中前三个表示箭头的起始坐标，后三个表示箭头相对与起始位置的偏移量。
            - **args** (tuple) - `FancyArrowPatch` 所需要的其他位置参数。
            - **kwargs** (dict) - `FancyArrowPatch` 所需要的其他关键字参数。

    .. py:method:: add_ket_label(ax, *args, fontsize=None, **kwargs)

        在给定的Axes中设置布洛赫球坐标轴的右矢标签。

        参数：
            - **ax** (mpl_toolkits.mplot3d.axes3d.Axes3D) - 想要添加右矢标签的三维Axes。
            - **args** (tuple) - 右矢标签中 `matplotlib` 的 `text` 对象的其他位置参数。
            - **kwargs** (dict) - 右矢标签中 `matplotlib` 的 `text` 对象的其他关键字参数。
            - **fontsize** (int) - 标签字体大小。如果为 ``None`` ，字体大小将取 `BlochScene` 的配置文件中键 `ket_label_fs` 对应的值。默认值： ``None`` 。

    .. py:method:: add_state(ax, amp, linecolor=None, linewidth=None, pointcolor=None, pointsize=None, marker=None, projcolor=None, mode=None, with_proj=None, stick_args=None, stick_kwargs=None, point_args=None, point_kwargs=None, proj_args=None, proj_kwargs=None)

        在布洛赫球中添加一个单比特量子态。

        参数：
            - **ax** (mpl_toolkits.mplot3d.axes3d.Axes3D) - 想要添加量子态的三维Axes。
            - **amp** (numpy.ndarray) - 量子态。
            - **linecolor** (str) - 线的颜色。如果为 ``None`` ，系统将会从 `BlochScene` 的 `config` 中取键 `stick_c` 对应的值。默认值： ``None`` 。
            - **linewidth** (numbers.Number) - 线宽度。如果为 ``None`` ，系统将会从 `BlochScene` 的 `config` 中取键 `stick_w` 对应的值。默认值： ``None`` 。
            - **pointcolor** (str) - 顶点的颜色。如果为 ``None`` ，系统将会从 `BlochScene` 的 `config` 中取键 `point_c` 对应的值。默认值： ``None`` 。
            - **pointsize** (numbers.Number) - 顶点的大小。如果为 ``None`` ，系统将会从 `BlochScene` 的 `config` 中取键 `point_s` 对应的值。默认值： ``None`` 。
            - **marker** (str) - 顶点的标记。如果为 ``None`` ，系统将会从 `BlochScene` 的 `config` 中取键 `point_m` 对应的值。默认值： ``None`` 。
            - **projcolor** (str) - 投影线的颜色。如果为 ``None`` ， 系统将会从 `BlochScene` 的 `config` 中取键 `proj_c` 对应的值。默认值： ``None`` 。
            - **mode** (str) - 怎样展示量子态。可以是 ``'stick'``、 ``'point'`` 或 ``'both'``。如果为 ``None`` ，系统将会从 `BlochScene` 的 `config` 中取键 `state_mode` 对应的值。默认值： ``None`` 。
            - **with_proj** (bool) - 是否展示量子态在x、y和z轴上的投影。如果为 ``None`` ，系统将会从 `BlochScene` 的 `config` 中取键 `with_proj` 对应的值。默认值： ``None`` 。
            - **stick_args** (tuple) - 绘制量子态的轴线的位置参数。这些参数会被传入到 `Axes3D.plot` 中。默认值： ``None`` 。
            - **stick_kwargs** (dict) - 绘制量子态的轴线的关键字参数。这些参数会被传入到 `Axes3D.plot` 中。默认值： ``None`` 。
            - **point_args** (tuple) - 量子态端点的其他位置参数。这些参数会被传入到 `Axes3D.scatter` 中。默认值： ``None`` 。
            - **point_kwargs** (dict) - 量子态端点的其他关键字参数。这些参数会被传入到 `Axes3D.scatter` 中。默认值： ``None`` 。
            - **proj_args** (tuple) - 投影线的其他位置参数。这些参数会被传入到 `Axes3D.plot` 中。默认值： ``None`` 。
            - **proj_kwargs** (dict) - 投影线的其他关键字参数。这些参数会被传入到 `Axes3D.plot` 中。默认值： ``None`` 。

        返回：
            dict，由折线、端点和投影线构成的字典。

    .. py:method:: animation(fig, ax, objs, new_amps: np.ndarray, interval=15, with_trace=True, history_len=None, **kwargs)

        在布洛赫球上动画展示给定量子态。

        参数：
            - **fig** (matplotlib.figure.Figure) - 布洛赫球场景所在的figure。
            - **ax** (mpl_toolkits.mplot3d.axes3d.Axes3D) - 布洛赫球场景所在的Axes。
            - **objs** (dict) - 通过 `BlochScene.add_state` 产生的对象。
            - **new_amps** (numpy.ndarray) - 所有需要动画展示的量子态。
            - **interval** (int) - 帧之间的时间间隔，单位为毫秒。默认值： ``15``。
            - **with_trace** (bool) - 是否展示量子态动画的轨迹。默认值： ``True``。
            - **history_len** (int) - 量子态动画的轨迹长度。如果为 ``None`` ，长度为所有需要动画展示的量子态的个数。默认自： ``None`` 。
            - **kwargs** (dict) - 传入 `animation.FuncAnimation` 的其他关键字参数。

        返回：
            animation.FuncAnimation，matplotlib中的动画句柄对象。

    .. py:method:: circle_xy(ax, *args, angle=np.pi / 2, **kwargs)

        在布洛赫球上绘制于xy平面平行的圆。

        参数：
            - **ax** (mpl_toolkits.mplot3d.axes3d.Axes3D) - 想要绘制平行于xy平面的圆的Axes。
            - **args** (tuple) - 传入给 `Axes3D.plot` 的位置参数。
            - **angle** (numbers.Number) - 圆相对于z轴的仰角。默认值：π / 2。
            - **kwargs** (dict) - 传入给 `Axes3D.plot` 的关键字参数。

    .. py:method:: circle_yz(ax, *args, angle=0, **kwargs)

        在布洛赫球上绘制于yz平面平行的圆。

        参数：
            - **ax** (mpl_toolkits.mplot3d.axes3d.Axes3D) - 想要绘制平行于yz平面的圆的Axes。
            - **args** (tuple) - 传入给 `Axes3D.plot` 的位置参数。
            - **angle** (numbers.Number) - 相对与z轴的旋转角度。默认值： ``0``。
            - **kwargs** (dict) - 传入给 `Axes3D.plot` 的关键字参数。

    .. py:method:: create_scene()

        根据 `BlochScene.config` 创建默认的布洛赫球场景。

    .. py:method:: gen_fig_ax(boxes=None)

        创建一个三维的画布。

        参数：
            - **boxes** (list) - 四维的浮点数数组，分别为左边界，底边，宽度和高度。如果为 ``None`` ，左边界和底边将为0，宽度和高度将为1。默认值： ``None`` 。

    .. py:method:: plot_slice(ax, x, y, z, frame_color, frame_alpha, surface_color, surface_alpha, frame_args=None, frame_kwargs=None, surface_args=None, surface_kwargs=None)

        在xy、yz和zx平面上创建参考平面。

        参数：
            - **ax** (mpl_toolkits.mplot3d.axes3d.Axes3D) - 需要添加参考面的三维Axes。
            - **x** (numpy.ndarray) - 参考面的x轴坐标。
            - **y** (numpy.ndarray) - 参考面的y轴坐标。
            - **z** (numpy.ndarray) - 参考面的z轴坐标。
            - **frame_color** (str) - 边框的颜色。
            - **frame_alpha** (numbers.Number) - 边框的透明度。
            - **surface_color** (str) - 参考面的颜色。
            - **surface_alpha** (numbers.Number) - 参考面的透明度。
            - **frame_args** (tuple) - 传入给 `Axes3D.plot_wireframe` 的其他位置参数。
            - **frame_kwargs** (dict) - 传入给 `Axes3D.plot_wireframe` 的其他关键字参数。
            - **surface_args** (tuple) - 传入给 `Axes3D.plot_surface` 的其他位置参数。
            - **surface_kwargs** (dict) - 传入给 `Axes3D.plot_surface` 的其他关键字参数。

    .. py:method:: set_view(ax, elev=0, azim=0)

        以合适的角度来展示布洛赫球。

        参数：
            - **ax** (mpl_toolkits.mplot3d.axes3d.Axes3D) - 需要设置视图的三维Axes。
            - **elev** (numbers.Number) - 以度为单位时，当前方位绕z轴转动的角度。
            - **azim** (numbers.Number) - 以度为单位时，当前方位相对与 (x, y) 平面的仰角。默认值： ``0``。

    .. py:method:: state_to_cor(amp: np.ndarray)
        :staticmethod:

        将单比特量子态转化为布洛赫球上的三维坐标。

        参数：
            - **amp** (numpy.ndarray) - 单比特量子态。

        返回：
            numpy.ndarray，量子态在布洛赫球中的三维坐标。

    .. py:method:: update(objs: dict, new_amp: np.ndarray)

        利用给定的量子态来更新通过 `BlochScene.add_state` 接口输出的对象。

        参数：
            - **objs** (dict) - 通过 `BlochScene.add_state` 接口输出的对象。
            - **new_amp** (numpy.ndarray) - 新的量子态。
