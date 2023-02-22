import numpy as np
import matplotlib.pyplot as plt


class AdamOpt:
    def __init__(self, func, grad, args_init: np.array, betas=(0.9, 0.999), eps=1e-8):
        """
        Constructor of AdamOpt class.
        :param func: the cost function that need to be minimize;
        :param grad: the gradient of "func";
        :param args_init: the initial guess parameters;
        :param betas: two super parameters in Adam optimizer;
        :param eps: a super parameter in Adam optimizer.
        """
        self.func = func
        self.grad = grad
        self.args_init = args_init
        self.args = args_init

        self.curve = [self.func(self.args_init)]

        # initialize Adam args
        self.iteration = 1
        self.m = np.zeros(args_init.shape)
        self.v = np.zeros(args_init.shape)
        self.betas = betas
        self.eps = eps

        # linear search on learning rate
        self.eta_list = np.zeros(5)
        for i in range(len(self.eta_list)):
            self.eta_list[i] = 0.1 / (4 ** i)

    def one_step_opt(self) -> None:
        """
        Train 1 step.
        """
        try_args = np.zeros((len(self.eta_list),) + self.args.shape)
        try_func = np.zeros(len(self.eta_list))
        grad = self.grad(self.args)
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grad
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * np.square(grad)
        # add corrected bias here
        m_hat = self.m / (1 - self.betas[0] ** self.iteration)
        v_hat = self.v / (1 - self.betas[1] ** self.iteration)

        for i in range(len(self.eta_list)):
            try_args[i] = self.args - self.eta_list[i] * m_hat / (np.sqrt(v_hat) + self.eps)
            try_func[i] = self.func(try_args[i])
        self.curve.append(np.min(try_func))
        self.args = try_args[np.argmin(try_func)]
        self.iteration += 1

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
