import argparse

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)

# about dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='')
parser.add_argument('--dataset', type=str, default='esol')
parser.add_argument('--num_workers', type=int, default=8)

# about training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_scale', type=float, default=1)
parser.add_argument('--decay', type=float, default=0)

# about molecule GNN
parser.add_argument('--gnn_type', type=str, default='gin')
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.5)
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')
parser.add_argument('--gnn_lr_scale', type=float, default=1)
parser.add_argument('--model_3d', type=str, default='schnet', choices=['schnet'])

# for AttributeMask
parser.add_argument('--mask_rate', type=float, default=0.15)
parser.add_argument('--mask_edge', type=int, default=0)

# for ContextPred
parser.add_argument('--csize', type=int, default=3)
parser.add_argument('--contextpred_neg_samples', type=int, default=1)

# for SchNet
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--num_interactions', type=int, default=6)
parser.add_argument('--num_gaussians', type=int, default=51)
parser.add_argument('--cutoff', type=float, default=10)
parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add'])
parser.add_argument('--schnet_lr_scale', type=float, default=1)

# for 2D-3D Contrastive CL
parser.add_argument('--CL_neg_samples', type=int, default=1)
parser.add_argument('--CL_similarity_metric', type=str, default='InfoNCE_dot_prod',
                    choices=['InfoNCE_dot_prod', 'EBM_dot_prod'])
parser.add_argument('--T', type=float, default=0.1)
parser.add_argument('--normalize', dest='normalize', action='store_true')
parser.add_argument('--no_normalize', dest='normalize', action='store_false')
parser.add_argument('--SSL_masking_ratio', type=float, default=0)
# This is for generative SSL.
parser.add_argument('--AE_model', type=str, default='AE', choices=['AE', 'VAE'])
parser.set_defaults(AE_model='AE')

# for 2D-3D AutoEncoder
parser.add_argument('--AE_loss', type=str, default='l2', choices=['l1', 'l2', 'cosine'])
parser.add_argument('--detach_target', dest='detach_target', action='store_true')
parser.add_argument('--no_detach_target', dest='detach_target', action='store_false')
parser.set_defaults(detach_target=True)

# for 2D-3D Variational AutoEncoder
parser.add_argument('--beta', type=float, default=1)

# for 2D-3D Contrastive CL and AE/VAE
parser.add_argument('--alpha_1', type=float, default=1)
parser.add_argument('--alpha_2', type=float, default=1)

# for 2D SSL and 3D-2D SSL
parser.add_argument('--SSL_2D_mode', type=str, default='AM')
parser.add_argument('--alpha_3', type=float, default=0.1)
parser.add_argument('--gamma_joao', type=float, default=0.1)
parser.add_argument('--gamma_joaov2', type=float, default=0.1)

# about if we would print out eval metric for training data
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=True)

# about loading and saving
parser.add_argument('--input_model_file', type=str, default='')
parser.add_argument('--output_model_dir', type=str, default='')

# verbosity
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no_verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)





parser.add_argument('--mlr', type=float, default=1e-3,
                    help='Initial learning rate. Default is 5e-4.')#2e-5  1e-3  3e-4
parser.add_argument('--jingtiaolr', type=float, default=3e-4,
                    help='Initial learning rate. Default is 5e-4.')
#

parser.add_argument('--mdropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability). Default is 0.5.')
parser.add_argument('--mattdropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability). Default is 0.5.')

parser.add_argument('--mweight_decay', type=float, default=5e-4,   #2e-4
                    help='Weight decay (L2 loss on parameters) Default is 5e-4.')#7e-4 1e-9
#
# parser.add_argument('--batch', type=int, default=256, ##
#                     help='Batch size. Default is 256.')
# #
# parser.add_argument('--epochs', type=int, default=100,
#                     help='Number of epochs to train. Default is 30.')#
#
# parser.add_argument('--network_ratio', type=float, default=1,
#                     help='Remain links in network. Default is 1')
#
# parser.add_argument('--loss_ratio1', type=float, default=1,
#                     help='Ratio of task1. Default is 1')
# ###
# parser.add_argument('--loss_ratio2', type=float, default=0.05,
#                     help='Ratio of task2. Default is 0.1')
# ##
# parser.add_argument('--loss_ratio3', type=float, default=0.1,
#                     help='Ratio of task3. Default is 0.1')
# #
# # GCN parameters#
# parser.add_argument('--dimensions', type=int, default=128,
#                     help='dimensions of feature. Default is 128.')
#
# parser.add_argument('--hidden1', default=64,
#                     help='Number of hidden units for encoding layer 1 for CSGNN. Default is 64.')
# #
# parser.add_argument('--hidden2', default=32,
#                     help='Number of hidden units for encoding layer 2 for CSGNN. Default is 32.')
#
# parser.add_argument('--decoder1', default=512,
#                     help='Number of hidden units for decoding layer 1 for CSGNN. Default is 512.')
# parser.add_argument('--zhongzi', default=0,
#                     help='Number of zhongzi.')
parser.add_argument('--pre', default=True,
                    help='pretrain or not.')
parser.add_argument('--modeltype', default='longadd',
                    help='model type.')
parser.add_argument('--traintype', default='freeze',
                    help='train type.')
parser.add_argument('--result_file', type=str, default='result.txt')
parser.add_argument('--pretrained', action="store_true", default=False)
parser.add_argument('--finetuned', action="store_true", default=False)

parser.add_argument('--pretrained_file', type=str, default='pretrained_model/pre-trained_M2UMol.pt')


args = parser.parse_args()
print('arguments\t', args)
