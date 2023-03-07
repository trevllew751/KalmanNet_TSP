"""# **Class: Kalman Filter**
Theoretical Linear Kalman
"""
import torch

class KalmanFilter:

    def __init__(self, SystemModel):
        self.F = SystemModel.F#.to("cuda:0");
        self.F_T = torch.transpose(self.F, 0, 1)#.to("cuda:0");
        self.m = SystemModel.m

        self.Q = SystemModel.Q#.to("cpu");

        self.H = SystemModel.H#.to("cuda:0");
        self.H_T = torch.transpose(self.H, 0, 1);
        self.n = SystemModel.n

        self.R = SystemModel.R#.to("cpu");

        self.T = SystemModel.T;
        self.T_test = SystemModel.T_test;

    # Predict

    def Predict(self):
        # Predict the 1-st moment of x
        # print(f"F device: {self.F.get_device()}")
        # print(f"m1x_posterior device: {self.m1x_posterior.get_device()}")
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)#.to("cuda:0");

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)#.to("cuda:0");
        # print(f"m2x_prior device: {self.m2x_prior.get_device()}")
        # print(f"F_T device: {self.F_T.get_device()}")
        # print(f"Q device: {self.Q.get_device()}")
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q;

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior);

        # Predict the 2-nd moment of y
        # print(f"H device: {self.H.get_device()}")
        # print(f"m2x_prior device: {self.m2x_prior.get_device()}")
        # print(f"H_T device: {self.H_T.get_device()}")
        # print(f"R device: {self.R.get_device()}")
        self.m2y = torch.matmul(self.H, self.m2x_prior);
        # print(f"m2y device: {self.m2y.get_device()}")
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R;

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    # Innovation
    def Innovation(self, y):
        # print(f"y device: {y.get_device()}")
        # print(f"m1y device: {self.m1y.get_device()}")
        y.to("cpu")
        self.dy = y - self.m1y;

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = (self.m1x_prior + torch.matmul(self.KG, self.dy))#.to("cuda:0");

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))#.to("cuda:0")
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)#.to("cuda:0")

    def Update(self, y):
        self.Predict();
        self.KGain();
        self.Innovation(y);
        self.Correct();

        return self.m1x_posterior,self.m2x_posterior;

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])

        self.m1x_posterior = self.m1x_0#.to("cuda:0")
        self.m2x_posterior = self.m2x_0#.to("cuda:0")

        for t in range(0, T):
            yt = torch.unsqueeze(y[:, t], 1);
            xt,sigmat = self.Update(yt);
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)