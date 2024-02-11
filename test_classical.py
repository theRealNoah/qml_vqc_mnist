import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *


class CustomQuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots):
        self.qubit_count = n_qubits
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.theta_list = []
        for i in range(n_qubits):
            self.theta_list.append(qiskit.circuit.Parameter('theta' + str(i)))

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        for i in range(n_qubits):
            print(i)
            self._circuit.ry(self.theta_list[i], all_qubits[i])

        self._circuit.measure_all()
        # ---------------------------
        print(self._circuit.draw())

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        # Fix from the example on the line below to get the parameter to be bound within the quantum circuit...
        # print(thetas)
        parameters = {self.theta_list[idx]: [value] for (idx, value) in enumerate(thetas)}
        # print(parameters)
        job = self.backend.run(t_qc, parameter_binds=[parameters], shots=self.shots)
        result = job.result().get_counts()

        classical_reg = {key: 0 for key in range(self.qubit_count)}
        for i in range(self.qubit_count):
            for key in result.keys():
                if (key[i] == '1'):
                    classical_reg[i] += (result[key] / self.shots)
                else:
                    classical_reg[i] -= (result[key] / self.shots)
        # print(classical_reg)
        # counts = np.array(list(result.values()))
        # states = np.array(list(result.keys())).astype(int)
        #
        # # print(states)
        # # print(counts)
        # # Compute probabilities for each state
        # probabilities = counts / self.shots
        # states_bin = [int(str(state), base=2) for state in states]
        # # print(states_bin)
        # # Get state expectation
        # expectation = np.sum(states_bin * probabilities)
        expectation = np.array(list(classical_reg.values()))

        return expectation


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        for i in range(1, len(input)):
            expectation_z = ctx.quantum_circuit.run(input[i].tolist())
            sub_result = torch.tensor([expectation_z])
            result = torch.cat([result, sub_result], 0)

        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = (CustomQuantumCircuit(10, backend, shots))
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


def train(model, train_loader, loss_func, optimizer, epoch):
    # switch model to train mode (dropout enabled)
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, loss_func):
    # switch model to eval model (dropout becomes pass through)
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_func(output, target).item() * data.shape[0]
            pred = output.argmax(dim=1)
            # pred = output.topk(1)[1].flatten()
            correct += (pred == target).long().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.06f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # seed pytorch random number generator for reproducablity
    torch.manual_seed(2)

    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=2)

    model = nn.Sequential(
        # input [B, 1, 28, 28]
        nn.Conv2d(1, 10, kernel_size=5),  # [B, 10, 24, 24]
        nn.MaxPool2d(kernel_size=2),  # [B, 10, 12, 12]
        nn.ReLU(),
        nn.Dropout2d(),
        nn.Conv2d(10, 20, kernel_size=5),  # [B, 20, 8, 8]
        nn.MaxPool2d(kernel_size=2),  # [B, 20, 4, 4]
        nn.ReLU(),
        nn.Flatten(),  # [B, 20*4*4] = [B, 320]
        nn.Linear(320, 50),  # [B, 50]
        nn.ReLU(),
        nn.Linear(50, 10))  # [B, 10]
        # , Hybrid(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 2))

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(4):
        train(model, train_loader, loss_func, optimizer, epoch)
        test(model, test_loader, loss_func)

    print('Saving model to mnist.pt')
    torch.save(model.state_dict(), 'mnist.pt')


if __name__ == "__main__":
    main()