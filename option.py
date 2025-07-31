import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='SSR', help='SSR, ASR')

# Degradation specifications for LF SSR
parser.add_argument('--blur_kernel', type=int, default=21, help='size of blur kernels')
parser.add_argument('--blur_type', type=str, default='iso_gaussian', help='blur types (iso_gaussian | aniso_gaussian)')
parser.add_argument('--noise', type=float, default=75.0, help='noise level')
parser.add_argument('--sig_min', type=float, default=0.2, help='minimum sigma of isotropic Gaussian blurs (training setting)')
parser.add_argument('--sig_max', type=float, default=4.0, help='maximum sigma of isotropic Gaussian blurs (training setting)')
parser.add_argument('--sig', type=float, default=0.0, help='specific sigma of isotropic Gaussian blurs (test setting)')
parser.add_argument('--lambda_min', type=float, default=0.2, help='minimum value for the eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--lambda_max', type=float, default=4.0, help='maximum value for the eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--lambda_1', type=float, default=0.0, help='one eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--lambda_2', type=float, default=0.0, help='another eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--theta', type=float, default=0.0, help='rotation angle of anisotropic Gaussian blurs [0, 180]')


parser.add_argument('--model_name', type=str, default='EPIT_SSR', help="model name")
parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
parser.add_argument("--path_pre_pth", type=str, default='./pth/', help="path for pre model ckpt")
parser.add_argument('--path_for_train', type=str, default='./data_for_training/')
parser.add_argument('--path_for_test', type=str, default='./data_for_test/')
parser.add_argument('--path_log', type=str, default='./log/')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--epoch', default=80, type=int, help='Epoch to run [default: 50]')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=4, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )

args = parser.parse_args()

