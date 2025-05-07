import logging
import torch
import torch.nn as nn
import torch.optim as optim
import updateModel
from ADM import *

class Client(object):
    def __init__(self, client_id):
        self.client_id = client_id

    def __repr__(self):
        '''
        ADM

        self.data_distribution()
        return 'Client #{}: {} samples in labels: {} & data distribution: {}\n'.format(
            self.client_id, len(self.data), set([label for _, label in self.data]), 
            [len(self.number_data[i]) for i in range(10)])
        '''
        return 'Client #{}: {} samples in labels: {}\n'.format(
            self.client_id, len(self.data), set([label for _, label in self.data]))
    
    def pt_data_distribution(self, file_logger_data=None):
        if file_logger_data is None:
            logging.info('Client #{}: {} samples & data distribution: {}'.format(
                self.client_id, len(self.data), [self.label_number[i] for i in range(10)]))
        else:
            file_logger_data.info('Client #{}: {} samples & data distribution: {}'.format(
                self.client_id, len(self.data), [self.label_number[i] for i in range(10)]))
    
    def data_distribution(self):
        self.number_data = {label: []
                        for _, label in self.data}
        # Populate grouped data dict
        for datapoint in self.data:
            # print(datapoint) # tensor, label로 구성됨
            _, label = datapoint  # Extract label
            self.number_data[label].append(  # pylint: disable=no-member
                datapoint)
        
        

    # Set non-IID data configurations
    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias

    def set_shard(self, shard):
        self.shard = shard

    def transfer(self, argv):
        # Download from the server.
        try:
            return argv.copy()
        except:
            return argv
    
    def set_data(self, data):
        # Download data
        self.data = self.transfer(data)
        # self.data = data
        data = self.data
        self.trainset = data
        # self.trainset = data[:int(len(data) * (1 - 0.2))]
        # self.testset = data[int(len(data) * (1 - 0.2)):]

    def adm_algorithm_1(self, vn):
        self.label_number = []
        trainset = []
        delta = 1
        vn = float(round(vn, 2))

        # 각 값의 길이를 알아내기
        # 값의 개수의 합 계산
        total_values_count = sum(len(value) for value in self.number_data.values())

        reduction = [0 for c in range(10)]

        # 값의 개수가 가장 많은 키 찾기
        sorted_keys = sorted(self.number_data, 
                             key=lambda k: len(self.number_data[k]), 
                             reverse=True)

        # 해당 키의 값 개수 구하기
        max_value_count = len(self.number_data[sorted_keys[0]])
        reduction_level = max_value_count - delta

        # print(vn)
        # print(total_values_count * vn)
        while int(total_values_count * vn) != (total_values_count - sum(reduction)):
            for c in range(10):
                num_data = len(self.number_data[c]) # 해당 라벨의 샘플 개수
                if num_data - reduction_level >= 0:
                    reduction[c] = num_data - reduction_level
                else:
                    reduction[c] = 0
            reduction_level = reduction_level - delta

        for c in range(10):
            reduced = len(self.number_data[c]) - reduction[c]
            extract = self.number_data[c][:reduced]
            self.label_number.append(len(extract))     
            trainset.extend(extract)
        self.set_data(trainset)


    # Non-IID
    def configure(self, config, model):
        self.config = config
        self.epochs = self.config.local_ep
        self.batch_size = self.config.local_bs
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr, momentum=0.5)

    # IID
    def configure_manual(self, config, model):
        self.config = config
        self.epochs = self.config.local_ep
        self.batch_size = 16
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr, momentum=0.5)

    def train(self):
        # logging.info('Training on client #{} / batch_size {}'.format(self.client_id, self.batch_size))

        trainloader = updateModel.get_trainloader(self.trainset, self.batch_size)
        updateModel.train(self.model, trainloader, self.optimizer, self.epochs)
        
        weights = updateModel.extract_weights(self.model)

        self.report = Report(self)
        self.report.weights = weights

        # testloader = updateModel.get_testloader(self.testset, 1000)
        # self.report.accuracy = updateModel.test(self.model, testloader)

    def get_report(self):
        # Report results to server.
        return self.transfer(self.report)
    
class Report(object):
    """Federated learning client report."""

    def __init__(self, client):
        self.client_id = client.client_id
        self.num_samples = len(client.data)
