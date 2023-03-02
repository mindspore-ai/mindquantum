import numpy as np
import matplotlib.pyplot as plt


class GDOpt:

    def __init__(self, func, grad, args_init: np.array, eta: float):
        self.func = func
        self.grad = grad
        self.args_init = args_init
        self.args = args_init
        self.curve = [self.func(self.args_init)]
        self.eta = eta

    def one_step_opt(self) -> None:
        """
        Train 1 step.
        """
        self.args += - self.eta * self.grad(self.args)
        self.curve.append(self.func(self.args))

    def multi_step_opt(self, steps: int) -> None:
        """
        Train multiple times.
        :param steps: training steps.
        """
        for _ in range(steps):
            self.one_step_opt()

    def plotCurve(self, y: str = None, saveas: str = None) -> None:
        """
        :param y: name of y axis;
        :param saveas: file name to save as *.pdf;
        """
        fig, axe = plt.subplots(1)
        axe.plot(self.curve)
        axe.set(xlabel='Steps', ylabel=y)
        if saveas is not None:
            plt.savefig(saveas + '.pdf')
        else:
            pass
        plt.show()
