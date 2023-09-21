import os
import argparse
from solver import Solver, SolverCustom
from data_loader import get_loader, TestDataset
from torch.backends import cudnn
import mlflow

def str2bool(v):
    return v.lower() in 'true'


def main(config):
    # For fast training.
    cudnn.benchmark = True

    print("Directories:", os.listdir(os.getcwd()))
    # Create needed directories
    if config.where_exec == "slurm":
        train_data_dir = os.path.join(config.gnrl_data_dir, "data/mc/train")
        test_data_dir = os.path.join(config.gnrl_data_dir, "data/mc/test")
        wav_dir = os.path.join(config.gnrl_data_dir, "data/wav16")
    elif config.where_exec == "local":
        train_data_dir = os.path.join("E:/TFM_EN_ESTE_DISCO_DURO/TFM_project/", "data/mc/train")
        test_data_dir = os.path.join("E:/TFM_EN_ESTE_DISCO_DURO/TFM_project/", "data/mc/test")
        wav_dir = os.path.join("E:/TFM_EN_ESTE_DISCO_DURO/TFM_project/", "data/wav16")

    output_directory = os.path.join(config.gnrl_data_dir, "output")
    os.makedirs(output_directory, exist_ok=True)


    log_dir = os.path.join(output_directory, "logs")
    model_save_dir = os.path.join(output_directory, "models")
    sample_dir = os.path.join(output_directory, "samples")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # MlFlow Parameters
    mlruns_folder = os.path.join(output_directory, "mlruns")# "./mlruns"
    mlflow_experiment_name = "[19_09_2023] 1st Attempt Slurm"
    mlflow_run_name = "_".join([f"{key}_{value}".replace(":", "_")  for key, value in vars(config).items() if "dir" not in key and "speakers" not in key ]).split("_resume_iters")[0]#str(config)
    mlflow.set_tracking_uri(mlruns_folder)
    experiment = mlflow.set_experiment(mlflow_experiment_name)
    mlflow.start_run(run_name=mlflow_run_name)

    # Log Parameters
    for key, value in vars(config).items():
        mlflow.log_param(key, value)

    # Create savedir subdirectories for current run
    config.log_dir = os.path.join(log_dir, mlflow_run_name)
    config.model_save_dir = os.path.join(model_save_dir, mlflow_run_name)
    config.sample_dir = os.path.join(sample_dir, mlflow_run_name)

    print(os.getcwd())
    print(train_data_dir)
    print(test_data_dir)
    print(wav_dir)

    # Create directories if not exist.
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    # TODO: remove hard coding of 'test' speakers
    src_spk = config.speakers[0]
    trg_spk = config.speakers[1]

    # Data loader.
    train_loader = get_loader(config.speakers, train_data_dir, config.batch_size, 'train',
                              num_workers=config.num_workers, preload_data=config.preload_data )
    # TODO: currently only used to output a sample whilst training
    test_loader = TestDataset(config.speakers, test_data_dir, wav_dir, src_spk=src_spk, trg_spk=trg_spk)


    # Solver for training and testing StarGAN.
    if config.preload_data:
        solver = SolverCustom(train_loader, test_loader, config)
    else:
        solver = Solver(train_loader, test_loader, config)

    # Train
    if config.mode == 'train':
        solver.train()

    # elif config.mode == 'test':
    #     solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=5, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=5, help='weight for id mapping loss')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')

    # Training configuration.
    parser.add_argument('--preload_data', type=bool, default=True, help='preload data on RAM')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Directories.
    parser.add_argument('--gnrl_data_dir', type=str, default='/workspace/NASFolder')
    parser.add_argument('--where_exec', type=str, default='local') # "slurm", "local"
    parser.add_argument('--speakers', type=str, nargs='+', required=False, help='Speaker dir names.',
                        default= ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251'])

    # Step size.
    parser.add_argument('--log_step', type=int, default=1) #10
    parser.add_argument('--sample_step', type=int, default=10000) #10000
    parser.add_argument('--model_save_step', type=int, default=10000) #10000
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()

    # no. of spks
    config.num_speakers = len(config.speakers)

    if len(config.speakers) < 2:
        raise RuntimeError("Need at least 2 speakers to convert audio.")

    print(config)
    main(config)
