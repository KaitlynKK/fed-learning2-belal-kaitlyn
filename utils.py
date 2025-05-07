import logging
import random
import math
import dists
from torchvision import datasets, transforms

def get_data(dataset):
    if dataset == "cifar":
        return CIFAR()
    elif dataset == "mnist":
        return MNIST()

class MNIST(object):
    def __init__(self):
        self.path = "./data/mnist/"
    
    def load_data(self):
        self.trainset = datasets.MNIST(
            self.path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.testset = datasets.MNIST(
            self.path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.labels = list(self.trainset.classes)

    # Group the data by label
    def group(self):
        # Create empty dict of labels
        grouped_data = {label: []
                        for label in self.labels}  # pylint: disable=no-member
        

        # Populate grouped data dict
        for datapoint in self.trainset:  # pylint: disable=all
            # print(datapoint) # tensor, label로 구성됨
            
            _, label = datapoint  # Extract label
            # print(label)
            label = self.labels[label]
            # print(label)
            # exit(1)
            
            grouped_data[label].append(  # pylint: disable=no-member
                datapoint)

        self.trainset = grouped_data  # Overwrite trainset with grouped data

    
    def generate(self):
        self.trainset_size = len(self.trainset)
        self.group()

        return self.trainset  

class CIFAR(object):
    def __init__(self):
        self.path = "./data/cifar/"
    
    def load_data(self):
        self.trainset = datasets.CIFAR10(
            self.path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.testset = datasets.CIFAR10(
            self.path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.labels = list(self.trainset.classes)

    # Group the data by label
    def group(self):
        # Create empty dict of labels
        grouped_data = {label: []
                        for label in self.labels}  # pylint: disable=no-member
        

        # Populate grouped data dict
        for datapoint in self.trainset:  # pylint: disable=all
            # print(datapoint) # tensor, label로 구성됨
            
            _, label = datapoint  # Extract label
            # print(label)
            label = self.labels[label]
            # print(label)
            # exit(1)
            

            grouped_data[label].append(  # pylint: disable=no-member
                datapoint)

        self.trainset = grouped_data  # Overwrite trainset with grouped data

    
    def generate(self):
        self.trainset_size = len(self.trainset)
        self.group()

        return self.trainset


class Loader(object):
    """Load and pass IID data partitions."""

    def __init__(self, config, generator):
        # Get data from generator
        self.config = config
        self.trainset = generator.trainset
        self.testset = generator.testset
        self.labels = generator.labels
        self.trainset_size = generator.trainset_size
        random.seed(self.config.seed)

        # Store used data seperately
        self.used = {label: [] for label in self.labels}
        self.used['testset'] = []

    def extract(self, label, n):
        # n = 600 / 10 = 60
        # print(label)
        # print(len(self.trainset[label]))
        # print(self.trainset[label][0])
        # exit()
        
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]  # Extract data
            self.used[label].extend(extracted)  # Move data to used

            del self.trainset[label][:n]  # Remove from trainset
            return extracted
        else:
            logging.warning('Insufficient data in label: {}'.format(label))
            logging.warning('Dumping used data for reuse')

            # Unmark data as used
            for label_ in self.labels:
                self.trainset[label_].extend(self.used[label_])
                self.used[label_] = []

            # Extract replenished data
            return self.extract(label, n)

    def get_partition(self, partition_size):
        # Get an partition uniform across all labels
        # IID인 경우 get_partition
        # 예, partition_size : 600

        # Use uniform distribution
        dist = dists.uniform(partition_size, len(self.labels))
        # print(dist)
        # exit()
        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))
        # print(len(partition))
        # print(partition[0])
        # exit()
        
        # Shuffle data partition
        random.shuffle(partition)

        return partition

    def get_testset(self):
        # Return the entire testset
        return self.testset
    
class BiasLoader(Loader):
    """Load and pass 'preference bias' data partitions."""

    def get_partition(self, partition_size, pref, bia):
        # Get a non-uniform partition with a preference bias
        # print("일단 여기")
        # exit(1)

        # Extract bias configuration from config
        bias = bia
        secondary = False
        
        # print(bias)
        # print(secondary)
        # exit(1)

       # Calculate sizes of majorty and minority portions
        majority = int(partition_size * bias)
        minority = partition_size - majority

        # Calculate number of minor labels
        len_minor_labels = len(self.labels) - 1

        if secondary:
                # Distribute to random secondary label
            dist = [0] * len_minor_labels
            dist[random.randint(0, len_minor_labels - 1)] = minority
        else:
            # Distribute among all minority labels
            # print(minority) # 120
            # print(len_minor_labels) # 9
            dist = dists.uniform(minority, len_minor_labels)

        # Add majority data to distribution
        dist.insert(self.labels.index(pref), majority)

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition
    
class ShardLoader(Loader):
    """Load and pass 'shard' data partitions."""

    def hybird_shards(self, num_IID_clients, non_shard):
        IID_client_shard = 10
        Non_IID_client_shard = non_shard
        # Non_IID_client_shard = 2
        
        IID_num_client = int(num_IID_clients)
        Non_IID_num_client = self.config.num_clients - IID_num_client

        IID_total = int(IID_num_client * IID_client_shard) # IID가 가지는 shard 총 개수
        
        IID_shard_size = int((self.trainset_size / self.config.num_clients) / IID_client_shard) # shard 사이즈
        # IID_shard_size = 600
        # IID shard : 250
        # IID_shard_size = int((10000) / IID_client_shard) # shard 사이즈

        Non_IID_total = int(Non_IID_num_client * Non_IID_client_shard) # 20
        Non_IID_shard_size = int((self.trainset_size / self.config.num_clients) / Non_IID_client_shard)

        self.IID_data = [[] for _ in range(IID_num_client)]
        
        for i in range(IID_num_client):
            partition = []
            for _, label in enumerate(self.labels):
                # random.shuffle(self.trainset[label])
                extracted = self.trainset[label][:IID_shard_size]
                self.used[label].extend(extracted)

                del self.trainset[label][:IID_shard_size]
                partition.extend(extracted)
            self.IID_data[i].extend(partition)

        '''    
        data = []
        for _, items in self.trainset.items():
            data.extend(items)

        shards = [data[(i * Non_IID_shard_size):((i + 1) * Non_IID_shard_size)]
                  for i in range(Non_IID_total)]
        '''
        
        # IID 는 다 다른 데이터셋
        # Non-IID는 어느정도는 겹치는 데이터셋
        number_shard = math.ceil((Non_IID_num_client * Non_IID_client_shard / 10)) # 10은 class 개수
        # number_shard = int(non_shard)

        shards = []
        for _, label in enumerate(self.labels):
            random.shuffle(self.trainset[label])
            shards_ = [self.trainset[label][:Non_IID_shard_size] for i in range(number_shard)]
            shards.extend(shards_)

        random.shuffle(shards)
        self.shards = shards
        self.used = []
        

    def create_shards(self):
        # Extract shard configuration from config
        # per_client : 한 사람당 shard 몇개 가질지 ?
        per_client = self.config.shard

        # Determine correct total shards, shard size
        total = self.config.num_clients * per_client
        shard_size = int(self.trainset_size / total)

        data = []  # Flatten data
        for _, items in self.trainset.items():
            data.extend(items)

        shards = [data[(i * shard_size):((i + 1) * shard_size)]
                  for i in range(total)]
        random.shuffle(shards)

        self.shards = shards
        
        self.used = []

        logging.info('Created {} shards of size {}'.format(
            len(shards), shard_size))

    def extract_shard(self):
        shard = self.shards[0]
        self.used.append(shard)
        del self.shards[0]
        return shard

    def get_partition(self):
        # Get a partition shard

        # Extract number of shards per client
        per_client = self.config.shard

        # Create data partition
        partition = []
        for i in range(per_client):
            partition.extend(self.extract_shard())

        # Shuffle data partition
        random.shuffle(partition)

        return partition

    def get_hybrid_partition(self, IID):
        # Get a partition shard

        if not IID:
            return self.get_partition()
        else:
            data = self.IID_data[0]
            random.shuffle(data)
            del self.IID_data[0]
            return data