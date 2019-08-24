import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--numWorker', type=int, default=4,
                        help='Number of workers to create')



# Main arguments
main_arg = parser.add_argument_group("Main")
modes = ["grey", "scale", "train", "predict"]  # valid execution modes
main_arg.add_argument("--mode", type=str,
                      choices=modes,
                      help="Client mode options")

# Greyscale mode arguments
grey_arg = parser.add_argument_group("Grey")
grey_arg.add_argument('--gsrc', type=str, default='../',
                        help='the directory to sample files from')
grey_arg.add_argument('--gout', type=str, default='../',
                        help='the directory to save the generated images to')
grey_arg.add_argument('--gext', type=str, default='png',
                        help='the extension of images to rescale.')

# Image scale mode arguements
scale_arg = parser.add_argument_group("Scale")
scale_arg.add_argument('--scale', type=int, default=2,
                        help='the pooling-scale factor to apply.')
scale_arg.add_argument('--ssrc', type=str, default='../',
                        help='the directory to sample files from')
scale_arg.add_argument('--sout', type=str, default='../gen',
                        help='the directory to save the generated images to')
scale_arg.add_argument('--sext', type=str, default='png',
                        help='the extension of images to rescale.')

# Training mode arguments
train_arg = parser.add_argument_group("Training")
train_arg.add_argument("--data_dir", type=str,
                       default="./data",
                       help="Training data directory")

train_arg.add_argument("--learn_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=10,
                       help="Size of each training batch")

train_arg.add_argument("--num_epoch", type=int,
                       default=25,
                       help="Number of training epochs")

train_arg.add_argument("--val_intv", type=int,
                       default=1000,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=1000,
                       help="Report interval")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")

train_arg.add_argument("--resume", type=bool,
                       default=True,
                       help="If true, resume training from existing \
                       checkpoint.")

# Model configuration
model_arg = parser.add_argument_group("Model")

model_arg.add_argument("--l2_reg", type=float,
                       default=1e-4,
                       help="L2 Regularization strength")


def get():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()
