import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', type=str, default='INFO',
                        help='Log messages level.')
    parser.add_argument('-d', '--dataset', type=str, default='cifar',
                        help='the name of dataset')

    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--local_dataset", type=int, default=2500, help="Total number of local dataset of each client")
    parser.add_argument("--IID_ratio", type=float, default=1.0)
    parser.add_argument("--frac", type=float, default=1)
    parser.add_argument("--IID", type=int, default=1, help='0 : non-IID / 1 : IID')
    parser.add_argument("--shard", type=int, default=2)
    parser.add_argument("--hybrid", type=int, default=0, help='0 : only one category  / 1 : non-IID & IID')
    parser.add_argument("--loader", type=str, default="bias", help='shard / bias')
    parser.add_argument("--seed", type=int, default=24, help='random seed')
    parser.add_argument("--contri", type=float, default=0.5, help='contribution')

    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--target_accuracy", default=0.999)
    parser.add_argument("--local_ep", type=int, default=10, help='local epoch')
    parser.add_argument("--local_bs", type=int, default=16, help='local batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    
    parser.add_argument("--model", type=str, default="cnn")

    args = parser.parse_args()

    return args
