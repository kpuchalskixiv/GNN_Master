import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="sample",
    help="dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample",
)
parser.add_argument("--batchSize", type=int, default=100, help="input batch size")
parser.add_argument("--hiddenSize", type=int, default=100, help="hidden state size")
parser.add_argument(
    "--epoch", type=int, default=30, help="the number of epochs to train for"
)
parser.add_argument(
    "--lr", type=float, default=0.001, help="learning rate"
)  # [0.001, 0.0005, 0.0001]
parser.add_argument("--lr-dc", type=float, default=0.1, help="learning rate decay rate")
parser.add_argument(
    "--lr-dc-step",
    type=int,
    default=3,
    help="the number of steps after which the learning rate decay",
)
parser.add_argument(
    "--lr-milestones",
    type=int,
    default=[2, 5, 8],
    help="Shedule of steps after which the learning rate decay",
    nargs="+",
)
parser.add_argument(
    "--l2", type=float, default=1e-5, help="l2 penalty"
)  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument("--step", type=int, default=1, help="gnn propogation steps")
parser.add_argument(
    "--patience",
    type=int,
    default=10,
    help="the number of epoch to wait before early stop ",
)
parser.add_argument(
    "--nonhybrid", action="store_true", help="only use the global preference to predict"
)
parser.add_argument(
    "--validation",
    action="store_true",
    help="[Depraceted] - always use validation. For testing refer to included notebooks.",
)
parser.add_argument(
    "--valid-portion",
    type=float,
    default=0.1,
    help="split the portion of training set as validation set",
)
parser.add_argument(
    "--pretrained-embeddings",
    action="store_true",
    help="initialize embeddings using word2vec",
)
parser.add_argument(
    "--unfreeze-epoch",
    type=int,
    default=0,
    help="epoch in which to unfreeze the embeddings layer",
)
parser.add_argument(
    "--gmm",
    default=[],
    nargs="*",
    type=int,
    help="train GM on validation dataset after training",
)
parser.add_argument(
    "--weight-init",
    type=str,
    default="normal",
    help="Type of torch.nn.init wieght initilization to use",
)
parser.add_argument(
    "--augment-matrix",
    action="store_true",
    help="Use version of SRGNN with modified adjacency matrix",
)
parser.add_argument(
    "--augment-clusters",
    action="store_true",
    help="[Depraceted] - use augment-alg instead! Use clusters from GMM to modify adjacency matrix",
)
parser.add_argument(
    "--augment-old-run-id",
    type=str,
    default="",
    help="Full ID of an old run, to use embeddings from",
)
parser.add_argument(
    "--augment-clip",
    type=float,
    default=0,
    help="Max value at which to clip adjacency matrix",
)
parser.add_argument(
    "--augment-normalize",
    action="store_true",
    help="Normalize adjacency matrix as in basic approach",
)
parser.add_argument(
    "--augment-raw",
    action="store_true",
    help="[Depraceted] Raw distances in adjacency matrix",
)
parser.add_argument(
    "--augment-p",
    type=float,
    default=1.0,
    help="Probability of matrix augmentation occuring",
)
parser.add_argument(
    "--augment-noise-p",
    type=float,
    default=0.0,
    help="Probability of matrix augmentation occuring",
)
parser.add_argument(
    "--augment-mean",
    type=float,
    default=0.01,
    help="Mean of gausian noise to inject into A",
)
parser.add_argument(
    "--augment-std",
    type=float,
    default=0.0,
    help="StandardDeviation of gausian noise to inject into A. Value equal to 0 corresponds to no noise injected",
)
parser.add_argument(
    "--augment-categories",
    action="store_true",
    help="[Depraceted] - use augment-alg instead! Use basic categories to modify adjacency matrix",
)
parser.add_argument(
    "--augment-nogmm",
    type=int,
    default=16,
    help="Number of gausoids used in GMM algorithm",
)
parser.add_argument(
    "--augment-gmm-init",
    type=str,
    default="k-means++",
    help="initialization of gausoids used in GMM algorithm",
)
parser.add_argument(
    "--augment-gmm-covariance",
    type=str,
    default="full",
    help="initialization of gausoids used in GMM algorithm",
)
parser.add_argument(
    "--augment-gmm-tol",
    type=float,
    default=1e-3,
    help="initialization of gausoids used in GMM algorithm",
)
parser.add_argument(
    "--lr-scheduler",
    type=str,
    default="step",
    help="Learning Rate scheduler to use",
)
parser.add_argument(
    "--augment-prenormalize-distances",
    action="store_true",
    help="Use basic categories to modify adjacency matrix",
)
parser.add_argument(
    "--augment-alg",
    type=str,
    default="gmm",
    choices=["gmm", "kmeans", "categories", "raw"],
    help="Augment with clusters distance based on GMM, KMeans or basic dataset item categories, eventually with raw item2item distance.",
)
