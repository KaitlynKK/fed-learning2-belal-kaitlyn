import logging
import server
import os
from options import args_parser

def file_generate():
    file_logger = logging.getLogger("File")
    file_logger.setLevel(logging.DEBUG)
    formatter1 = logging.Formatter('%(message)s')
    log_dir = "logs"

    log_dir = log_dir  + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    # log_f_name = log_dir + 'test_' + str(run_num) + '.log'
    log_f_name = log_dir + 'test_GUI.log'
    file_handler = logging.FileHandler(log_f_name)
    
    file_handler.setFormatter(formatter1)
    file_logger.addHandler(file_handler)

    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
    
    return file_logger, log_f_name

def server_init(params):
    args = args_parser()
    # print(args)
    req_rounds = int(params['rounds'])
    req_clients = int(params['clients'])
    req_dataset = params['dataset']
    req_IID = int(params['data_distribution'])
    
    args.rounds = req_rounds
    args.num_clients = req_clients
    args.dataset = req_dataset
    args.IID = req_IID
    

    file_logger = logging.getLogger("File")
    file_logger.setLevel(logging.DEBUG)
    formatter1 = logging.Formatter('%(message)s')
    log_dir = "logs"

    log_dir = log_dir  + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    # log_f_name = log_dir + 'test_' + str(run_num) + '.log'
    # log_f_name = log_dir + 'test_GUI.log'
    log_f_name = log_dir + '[' + str(args.rounds) + "]rounds_" + "[" + \
        str(args.num_clients) + "]clients_[" + str(args.dataset) + "]_[" + str(args.IID) + "]IID.log"
    file_handler = logging.FileHandler(log_f_name)
    
    file_handler.setFormatter(formatter1)
    file_logger.addHandler(file_handler)

    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
    
    fl_server = server.Server(args, file_logger)
    return fl_server, log_f_name

def server_boot(fl_server):
    fl_server.boot()

def server_run(fl_server, r):
    fl_server.run(r)

def main(params, file_logger):
    args = args_parser()
    # print(args)
    req_rounds = int(params['rounds'])
    req_clients = int(params['clients'])
    req_dataset = params['dataset']
    
    args.rounds = req_rounds
    args.local_ep = req_clients
    args.dataset = req_dataset
    
    fl_server = server.Server(args, file_logger)
    # fl_server = server.Server(args)
    fl_server.boot()
    # exit(1)
    fl_server.run()

if __name__ == "__main__":
    main()