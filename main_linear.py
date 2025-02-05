import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from Linear_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
from Pipeline_KF import Pipeline_KF
from KalmanNet_nn import KalmanNetNN
from datetime import datetime

import pandas as pd

from KalmanFilter_test import KFTest

from Plot import Plot_RTS as Plot

# if torch.cuda.is_available():
#    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
#    torch.set_default_tensor_type(torch.cuda.FloatTensor)
#    print("Running on the GPU")
# else:
#    dev = torch.device("cpu")
#    torch.set_default_tensor_type(torch.FloatTensor)
#    print("Running on the CPU")
#
# print(f"Default device: {torch.tensor([1.,2]).get_device()}")
dev = torch.device("cpu")
torch.set_default_tensor_type('torch.FloatTensor')
print("Running on the CPU")



print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'RTSNet/'

####################
### Design Model ###
####################
r2 = torch.tensor([10,1.,0.1,1e-2,1e-3])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)



#train_data = pd.read_csv('avl_log.csv')
#train_tensor = torch.tensor(train_data.values)
#print("Training input data: " + str(train_input))
#print("Training target data: " + str(train_target))
#print("N_CV: " + str(N_CV))

for index in range(0,len(r2)):

   print("1/r2 [dB]: ", 10 * torch.log10(1/r2[index]))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q2[index]))

   # True model
   r = torch.sqrt(r2[index])
   q = torch.sqrt(q2[index])
   sys_model = SystemModel(F, q, H, r, T, T_test)
   sys_model.InitSequence(m1_0, m2_0)

   # Mismatched model
   sys_model_partialh = SystemModel(F, q, H_rotated, r, T, T_test)
   sys_model_partialh.InitSequence(m1_0, m2_0)

   ###################################
   ### Data Loader (Generate Data) ###
   ###################################
   dataFolderName = 'Simulations/Linear_canonical/H=I' + '/'
   dataFileName = ['2x2_rq-1010_T100.pt','2x2_rq020_T100.pt','2x2_rq1030_T100.pt','2x2_rq2040_T100.pt','2x2_rq3050_T100.pt']
   # print("Start Data Gen")
   # DataGen(sys_model, dataFolderName + dataFileName[index], T, T_test,randomInit=False)
   print("Data Load")
   # [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName[index])
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName[index])
   # print(f"train input device: {train_input.get_device()}")
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())

   ##############################
   ### Evaluate Kalman Filter ###
   ##############################
   print("Evaluate Kalman Filter True")
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target)
   print("Evaluate Kalman Filter Partial")
   [MSE_KF_linear_arr_partialh, MSE_KF_linear_avg_partialh, MSE_KF_dB_avg_partialh] = KFTest(sys_model_partialh, test_input, test_target)



   DatafolderName = 'Filters/Linear' + '/'
   DataResultName = 'KF_'+ dataFileName[index]
   torch.save({
      'MSE_KF_linear_arr': MSE_KF_linear_arr,
      'MSE_KF_dB_avg': MSE_KF_dB_avg,
      'MSE_KF_linear_arr_partialh': MSE_KF_linear_arr_partialh,
      'MSE_KF_dB_avg_partialh': MSE_KF_dB_avg_partialh,
   }, DatafolderName+DataResultName)

   print("MSE_KF_linear_arr length: " + str(len(MSE_KF_linear_arr)))
   #print("MSE_KF_dB_avg length: " + str(len(MSE_KF_dB_avg)))

   ##################
   ###  KalmanNet ###
   ##################
   print("Start KNet pipeline")
   print("KNet with full model info")
   modelFolder = 'KNet' + '/'
   KNet_Pipeline = Pipeline_KF(strTime, "KNet", "KNet_"+ dataFileName[index])
   KNet_Pipeline.setssModel(sys_model)
   KNet_model = KalmanNetNN()
   KNet_model.Build(sys_model)
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setTrainingParams(n_Epochs=10, n_Batch=30, learningRate=1E-3, weightDecay=1E-5)

   KNet_Pipeline.model = torch.load(modelFolder+"model_KalmanNet.pt")
   
   KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
   [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
   #[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, train_tensor, test_target)
   KNet_Pipeline.PlotTrain_KF(KNet_MSE_test_linear_arr, KNet_MSE_test_dB_avg)
   
   KNet_Pipeline.save()



   #print("KNet with partial model info")
   #modelFolder = 'KNet' + '/'
   #KNet_Pipeline = Pipeline_KF(strTime, "KNet", "KNetPartial_"+ dataFileName[index])
   #KNet_Pipeline.setssModel(sys_model_partialh)
   #KNet_model = KalmanNetNN()
   #KNet_model.Build(sys_model_partialh)
   #KNet_Pipeline.setModel(KNet_model)
   #KNet_Pipeline.setTrainingParams(n_Epochs=10, n_Batch=30, learningRate=1E-3, weightDecay=1E-5)

   #KNet_Pipeline.model = torch.load(modelFolder+"model_KalmanNet.pt")
   #KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
   #[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
   #KNet_Pipeline.save()


