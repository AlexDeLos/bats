import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser(prog="SNN script",
                                     description="Run a SNN with residual connections on the MNIST dataset.")
    parser.add_argument("snn", choices=["SNN"], default="SNN")
    parser.add_argument("--use_residual", action="store_true", default=True)
    # if file is moved in another directory level relative to the root (currently in root/utils/src), this needs to be changed
    # root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # parser.add_argument("-s", "--save_model", action="store_true", default=True)
    # # parser.add_argument("-m", "--model_name", default=''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)]))
    # parser.add_argument("-p", "--plot", action="store_true", default=True)
    # parser.add_argument("-o", "--optimizer", default="Adam")
    # parser.add_argument("-c", "--criterion", default="MSELoss")
    # parser.add_argument("-b", "--batch_size", default=32, type=int)
    # parser.add_argument("-n", "--n_epochs", default=200, type=int)
    # parser.add_argument("-l", "--learning_rate", default=1e-3, type=float)
    # parser.add_argument("-w", "--weight_decay", default=0.05, type=float)
    # parser.add_argument("--mixed_loss_weight", default=0.1, type=float)
    # parser.add_argument("--n_hidden_gnn", default=2, type=int)
    # parser.add_argument("--gnn_hidden_dim", default=32, type=int)
    # parser.add_argument("--n_hidden_lin", default=2, type=int)
    # parser.add_argument("--lin_hidden_dim", default=8, type=int)
    # parser.add_argument("--patience", default=40, type=int)
    # parser.add_argument("--plot_node_error", action="store_true", default=False)
    # parser.add_argument("--normalize", action="store_true", default=False)
    # parser.add_argument("--no_linear", action="store_true", default=False)
    # parser.add_argument("--loss_type", choices=['standard', 'physics', 'mixed'], default='standard')
    # parser.add_argument("--value_mode", choices=['all', 'missing', 'voltage'], default='all')
    # parser.add_argument("--pretrain", action="store_true", default=False)
    # parser.add_argument("--from_checkpoint", default=None, type=str) 

    args = parser.parse_args()
    return args
