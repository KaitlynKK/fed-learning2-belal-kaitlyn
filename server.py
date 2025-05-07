import torch
import logging
import time
import utils
import random
from models import *
from ADM import *
import threading
import client
import copy
import updateModel
import dists

class Server(object):
    def __init__(self, args, file_logger=None):
        self.config = args
        self.file_logger = file_logger

    def boot(self):
        logging.info('Booting server...')

        self.load_model()
        self.make_clients()

    def load_model(self):
        IID = self.config.IID
        config = self.config
        howto = self.config.loader
        dataset = self.config.dataset
        logging.info('dataset: {}'.format(dataset))

        generator = utils.get_data(dataset)
        generator.load_data()
        data = generator.generate()
        labels = generator.labels

        logging.info('Dataset size: {}'.format(
            sum([len(x) for x in [data[label] for label in labels]])))
        logging.info('Labels ({}): {}'.format(
            len(labels), labels))
        
        if not IID :
            if howto == "shard":
                self.loader = utils.ShardLoader(config, generator)
            elif howto == "bias":
                self.loader = utils.BiasLoader(config, generator)

        else: # IID
            self.loader = utils.Loader(config, generator)

        self.model = get_model(dataset)
        
    
    def make_clients(self):
        IID = self.config.IID
        labels = self.loader.labels
        hybrid = self.config.hybrid
        num_clients = self.config.num_clients

        clients = []
        for client_id in range(num_clients):
            new_client = client.Client(client_id)
            
            if not IID:
                if self.config.loader == "shard":
                    shard = self.config.shard
                    new_client.set_shard(shard)
                elif self.config.loader == "bias":
                    dist = dists.uniform(num_clients, len(labels))
                    self.bias = 0.8 # 0.8
                    pref = random.choices(labels, weights=dist)[0]

                    new_client.set_bias(pref, self.bias)

            
            clients.append(new_client)

        logging.info('Total clients: {}'.format(len(clients)))

        if not IID and self.config.loader == "bias":
            logging.info('Label distribution: {}'.format(
                [[client.pref for client in clients].count(label) for label in labels]))

        if not IID:
            if not hybrid:
                if self.config.loader == "shard":
                    self.loader.create_shards()
                    [self.set_client_data(client) for client in clients]
                else :
                    # [self.set_client_data(client, number_of_sample=1000) for client in clients[:10]]
                    # [self.set_client_data(client, number_of_sample=3000) for client in clients[10:]]
                    [self.set_client_data(client, number_of_sample=self.config.local_dataset) for client in clients]
            else:
                if self.config.loader == "shard":
                    m = max(int(self.config.IID_ratio * self.config.num_clients), 1)
                    # IID_clients = [client for client in random.sample(
                    #     clients, m)]
                    IID_clients = [client for client in clients[:m]]
                    Non_clients = set(clients) - set(IID_clients)
                    # print(len(IID_clients))
                    # print(len(Non_clients))
                    self.loader.hybird_shards(len(IID_clients), self.config.shard)
                    # [self.set_client_IID(client) for client in clients[:int(num_clients/2)]]
                    # [self.set_client_Non_IID(client) for client in clients[int(num_clients/2):]]
                    [self.set_client_IID(client) for client in IID_clients]
                    [self.set_client_Non_IID(client) for client in Non_clients]

                else :
                    pass
        else: # IID
            [self.set_client_data(client) for client in clients]

        self.clients = clients
        logging.info('Clients: {}'.format(self.clients))

        self.number_sample_list = []
        for c in self.clients:
            self.number_sample_list.append(len(c.data))
        

    def set_client_data(self, client, number_of_sample=2500):
        IID = self.config.IID
        loader = self.config.loader
        # number_of_sample = 2500
        if not IID:
            if loader == "shard":
                data = self.loader.get_partition()
            elif loader == "bias":
                data = self.loader.get_partition(number_of_sample, client.pref, self.bias)

        else: # IID
            data = self.loader.get_partition(number_of_sample)
            # print(data[0])
            # exit()

        # Send data to client
        client.set_data(data)

    def set_client_IID(self, client):
        IID = True
        data = self.loader.get_hybrid_partition(IID)
        client.set_data(data)

    def set_client_Non_IID(self, client):
        IID = False
        data = self.loader.get_hybrid_partition(IID)
        client.set_data(data)



    def run(self, round=0):
        rounds = self.config.rounds
        target_accuracy = self.config.target_accuracy
        
        logging.info('**** Round {}/{} ****'.format(round+1, rounds))
        accuracy = self.fl_round(round)
    
    def fl_round(self, curr_round):
        start_time_epochs = time.time()
        sample_clients = self.selection()
        self.configuration(sample_clients)
        self.adm_configuration(sample_clients)
        
        '''
        ADM Algorithm 1
        self.optimal_v_n, sol_list, optimal_t = block_coordinate_descent(self.parameters,
                                                                    curr_round,
                                                                    self.parameters["t"]) # sol_list는 objective function 값
        
        self.parameters["t"] = optimal_t

        if curr_round > 0 and curr_round % 20 == 0:
            self.optimal_v_n = [x - 0.1 for x in self.optimal_v_n]
        vn = [round(num, 3) for num in self.optimal_v_n]
        logging.info('v_n: {}'.format(vn))

        for client in sample_clients:
            client.adm_algorithm_1(self.optimal_v_n[client.client_id])
            # if client.client_id == 0 or client.client_id == 10:
                # client.pt_data_distribution(self.file_logger)
            client.train()
        # logging.info("D_n: {}".format(self.parameters["D_n"]))
        '''

        for client in sample_clients:
            client.train()

        reports = [client.get_report() for client in sample_clients]
        logging.info('Reports recieved: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)
        updateModel.load_weights(self.model, updated_weights)

        testset = self.loader.get_testset()
        batch_size = 1000
        testloader = updateModel.get_testloader(testset, batch_size)
        accuracy = updateModel.test(self.model, testloader)

        logging.info('Global model accuracy: {:.2f}%'.format(100 * accuracy))
        self.file_logger.info("{0}_{1}".format(curr_round, 100*accuracy))
        print()

        return accuracy

    def selection(self):
        m = max(int(self.config.frac * self.config.num_clients), 1)
        # sample_clients = [client for client in random.sample(
        #     self.clients, m)]
        sample_clients = [client for client in self.clients[:m]]
        
        return sample_clients
    
    def adm_configuration(self, sample_clients):
        curr_number_sample = []
        number_sample_list=[1000 for _ in range(10)]
        number_sample_list.extend([3000 for _ in range(10)])
        for client in sample_clients:
            curr_number_sample.append(len(client.data))
        constant_parameters = {'sigma' : 0.9, 'D_n': number_sample_list, 'Gamma': 0.4, 'local_iter': 10, 'c_n': 30,
                #   'frequency_n_GHz' : [1.5, 2, 2.5, 3], 
                  'frequency_n_GHz' : [3000], 
                  'weight_size_n_kbit' : 100,
                  'number_of_clients' : self.config.num_clients, 'bandwidth_MHz' : 1, 'channel_gain_n': 1, 
                #   'transmission_power_n' : [0.2, 0.5, 1], 
                  'transmission_power_n' : [1], 
                  'noise_W' : 1e-12,
                #   't':500}
                  't':300} # non-iid 섞어서 할때
        
        self.parameters=init_param_hetero(constant_parameters, self.config.num_clients, constant_parameters["t"])
        # logging.info("D_n: {}".format(curr_number_sample))
        

    def configuration(self, sample_clients):
        hybrid = self.config.hybrid
        if hybrid :
            # 로컬을 다 뽑는 경우에만 해당
            for client in sample_clients:
                config = self.config
                client.configure(config, model=copy.deepcopy(self.model))
                # if client.client_id <= 9: # IID
                #     client.configure_manual(config, model=copy.deepcopy(self.model))
                # else:
                #     client.configure(config, model=copy.deepcopy(self.model))

        else :
            for client in sample_clients:
                config = self.config
                client.configure(config, model=copy.deepcopy(self.model))

    def extract_client_updates(self,reports):
        baseline_weights = updateModel.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            # print(update)
            # exit(1)
            updates.append(update)
        return updates


    def aggregation(self, reports):
        return self.fedavg(reports)
    
    def fedavg(self,reports):
        # num = int((self.config.num_clients / 2))
        # contri_action = [self.config.contri / num] * num
        # contri_action_non = [(1 - self.config.contri) / num] * num
        # contri_action.extend(contri_action_non)
        
        updates = self.extract_client_updates(reports)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])
        
        # Perform weighted averaging
        avg_update = [torch.zeros(x.size()) for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            # if i >= 10:
            #     break
            # contri = contri_action[reports[i].client_id]
            # logging.info("기여도: {}".format(num_samples / total_samples))

            for j, (_, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)
                # avg_update[j] += delta * contri

        # Extract baseline model weights
        baseline_weights = updateModel.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights
