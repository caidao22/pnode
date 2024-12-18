# This code is a driver function for image classification on Cifar10 dataset using PNODE for training.
# It utilizes a SqueezeNext network architecture, where residual blocks are replaced by ODE blocks.
# Based on an ANODE training driver based on arxiv:1902.10298.
#
# Example of usage:
#   python3 train-Cifar10.py -ts_adapt_type none -ts_trajectory_type memory --num_epochs 200 --method euler
#
# Prerequisites:
#   pnode torchvision tensorboardX torchsummary nvidia-ml-py3 petsc4py

import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
import math
import sys
import os
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, choices=["resnet", "sqnxt"], default="sqnxt")
parser.add_argument(
    "--method",
    type=str,
    choices=["euler", "rk2", "bosh3", "rk4", "dopri5"],
    default="euler",
)  # Time stepping schemes for ODE solvers
parser.add_argument("--num_epochs", type=int, default=200)  # Number of Epochs in total
parser.add_argument("--lr", type=float, default=0.1)  # Learning rate (initial)
parser.add_argument("--Nt", type=int, default=1)  # Number of time steps
parser.add_argument("--t0", type=float, default=0.0)
parser.add_argument("--t1", type=float, default=1.0)
parser.add_argument(
    "--batch_size", type=int, default=256
)  # Batch size used for training
parser.add_argument(
    "--test_batch_size", type=int, default=128
)  # Batch size used for testing
parser.add_argument("--gpu", type=int, default=0)  # Number of GPU
parser.add_argument(
    "--save", type=str, default=None
)  # Save log files in this directory
parser.add_argument("--deterministic", action="store_true")  # Deterministic mode on/off
parser.add_argument("--seed", type=int, default=0)  # Random seed

args, unknown = parser.parse_known_args()

sys.argv = [sys.argv[0]] + unknown

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tensor_type = torch.float32
if is_use_cuda:
    import nvidia_smi

    nvidia_smi.nvmlInit()
    total_cuda_mem = 0.0
# Experienced PETSc users may switch archs by setting the petsc4py path manually
# petsc4py_path = os.path.join(os.environ["PETSC_DIR"], os.environ["PETSC_ARCH"], "lib")
# sys.path.append(petsc4py_path)
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc

# Set the random seed in deterministic mode
if args.deterministic:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
# Import PNODE
# sys.path.append("../") # for quick debugging
from pnode import petsc_adjoint

if args.network == "sqnxt":
    from models.sqnxt_PETSc import SqNxt_23_1x, lr_schedule
elif args.network == "resnet":
    from models.resnet_PETSc import ResNet18, lr_schedule
if args.save is None:
    args.save = "sqnxt/" + args.method + "_Nt_" + str(args.Nt) + "/"
writer = SummaryWriter(args.save)

num_epochs = int(args.num_epochs)
lr = float(args.lr)
start_epoch = 1
batch_size = int(args.batch_size)
test_batch_size = int(args.test_batch_size)

global TrainMode
TrainMode = True


# This class defines an ODE block through PNODE:
class ODEBlock_PNODE(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock_PNODE, self).__init__()

        self.odefunc = odefunc.to(device)
        self.options = {}
        # Specify step size
        self.step_size = 1.0 / float(args.Nt)
        # Specify time stepper
        self.method = args.method

        self.ode = petsc_adjoint.ODEPetsc()

        # Specify range of integration: from t0 to t1
        # self.integration_time = torch.tensor( [args.t0,args.t1] ).float()
        self.integration_time = torch.tensor([args.t1]).float()

    def forward(self, x):
        # Define foward pass
        if TrainMode:
            self.ode.setupTS(
                x.to(tensor_type),
                self.odefunc,
                step_size=self.step_size,
                method=self.method,
                enable_adjoint=True,
            )
        else:  # Disable adjoint method, as test does not require backpropagation
            self.ode.setupTS(
                x.to(tensor_type),
                self.odefunc,
                step_size=self.step_size,
                method=self.method,
                enable_adjoint=False,
            )
        out = self.ode.odeint_adjoint(x.to(tensor_type), self.integration_time)
        return out[-1]

    @property
    def nfe(self):
        # Number of function evaluations
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find("BatchNorm") != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# Data Preprocess
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", transform=transform_train, train=True, download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", transform=transform_test, train=False, download=True
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    num_workers=0,
    shuffle=False,
    drop_last=True,
)

# Define the ODEBlock
ODEBlock = ODEBlock_PNODE

if args.network == "sqnxt":
    # Import the SqueezeNext network
    net = SqNxt_23_1x(10, ODEBlock)
    # net_test = SqNxt_23_1x(10, ODEBlock, Train=False)
    # net_test.load_state_dict(net.state_dict())
elif args.network == "resnet":
    net = ResNet18(ODEBlock)

net.apply(conv_init)
# print(net)

if is_use_cuda:
    net.to(device)
    # net_test.to(device)

# Objective function
criterion = nn.CrossEntropyLoss().to(device)


def get_logger(
    logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    # Initialize the logger file
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


# Function for training
def train(epoch):
    global total_cuda_mem
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(
        net.parameters(), lr=lr_schedule(lr, epoch), momentum=0.9, weight_decay=5e-4
    )

    print("Training Epoch: #%d, LR: %.4f" % (epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar(
            "Train/Loss", loss.item(), epoch * 50000 + batch_size * (idx + 1)
        )
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()

        sys.stdout.write("\r")
        if is_use_cuda:
            peak_torch_cuda_mem = torch.cuda.max_memory_allocated(device)
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            total_cuda_mem = info.used
            sys.stdout.write(
                "[%s] Training Epoch [%d/%d] Iter[%d/%d]    Loss: %.4f Acc@1: %.3f TotalMem: %.3f PeakTorchMem: %.3f"
                % (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                    epoch,
                    num_epochs,
                    idx + 1,
                    len(train_dataset) // batch_size,
                    train_loss / (batch_size * (idx + 1)),
                    correct / total,
                    info.used / 1e9,
                    peak_torch_cuda_mem / 1e9,
                )
            )
        else:
            sys.stdout.write(
                "[%s] Training Epoch [%d/%d] Iter[%d/%d]    Loss: %.4f Acc@1: %.3f"
                % (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                    epoch,
                    num_epochs,
                    idx + 1,
                    len(train_dataset) // batch_size,
                    train_loss / (batch_size * (idx + 1)),
                    correct / total,
                )
            )
        sys.stdout.flush()
        logger.info(
            "[%s] Training Epoch [%d/%d] Iter[%d/%d]    Loss: %.4f Acc@1: %.3f"
            % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                epoch,
                num_epochs,
                idx + 1,
                len(train_dataset) // batch_size,
                train_loss / (batch_size * (idx + 1)),
                correct / total,
            )
        )
    writer.add_scalar("Train/Accuracy", correct / total, epoch)
    if is_use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        writer.add_scalar("Memory", info.used / 1e9)


# Function for test:
def test(epoch):
    # net_test.load_state_dict(net.state_dict())
    # net_test.eval()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        # outputs = net_test(inputs)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        writer.add_scalar(
            "Test/Loss", loss.item(), epoch * 50000 + test_loader.batch_size * (idx + 1)
        )

        sys.stdout.write("\r")
        sys.stdout.write(
            "[%s] Testing Epoch [%d/%d] Iter[%d/%d]    Loss: %.4f Acc@1: %.3f"
            % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                epoch,
                num_epochs,
                idx + 1,
                len(test_dataset) // test_loader.batch_size,
                test_loss / (100 * (idx + 1)),
                correct / total,
            )
        )
        sys.stdout.flush()
        logger.info(
            "[%s] Testing Epoch [%d/%d] Iter[%d/%d]    Loss: %.4f Acc@1: %.3f"
            % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                epoch,
                num_epochs,
                idx + 1,
                len(test_dataset) // test_loader.batch_size,
                test_loss / (100 * (idx + 1)),
                correct / total,
            )
        )
    writer.add_scalar("Test/Accuracy", correct / total, epoch)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if __name__ == "__main__":
    makedirs(args.save)

    logger = get_logger(
        logpath=os.path.join(args.save, "log"),
        filepath=os.path.abspath(__file__),
        saving=False,
        displaying=False,
    )
    logger.info(args)

    logger.info(net)
    logger.info("Number of parameters: {}".format(count_parameters(net)))
    for _epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()
        TrainMode = True
        train(_epoch)
        print()
        TrainMode = False
        test(_epoch)
        print()
        print()
        end_time = time.time()
        print("Epoch #%d Cost %ds" % (_epoch, end_time - start_time))
        logger.info("Epoch #%d Cost %ds" % (_epoch, end_time - start_time))

    writer.close()
    if is_use_cuda:
        f = open("memstat.txt", "a")
        framework = "PNODE"
        if args.method == "euler":
            method = "Euler"
        elif args.method == "rk2":
            method = "RK2"
        elif args.method == "fixed_bosh3":
            method = "RK3"
        elif args.method == "rk4":
            method = "RK4"
        elif args.method == "fixed_dopri5":
            method = "Dopri5"
        f.write(
            "{}, {:.3f}, {:.3f}, {}, {}\n".format(
                args.Nt, total_cuda_mem / 1e9, end_time - start_time, method, framework
            )
        )
        f.close()
