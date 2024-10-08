from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'noise',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

######## S1-S4 Space ########
#### cifar10 s1 - s4

init_pt_s1_C10_0 = Genotype(
    normal=[["dil_conv_3x3", 0], ["dil_conv_5x5", 1], ["sep_conv_3x3", 1], ["dil_conv_3x3", 2], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 0], ["dil_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["avg_pool_3x3", 0], ["dil_conv_3x3", 1], ["avg_pool_3x3", 1], ["skip_connect", 2], ["sep_conv_3x3", 1],
            ["dil_conv_3x3", 2], ["dil_conv_5x5", 2], ["dil_conv_5x5", 4]], reduce_concat=range(2, 6))
init_pt_s1_C10_2 = Genotype(
    normal=[["skip_connect", 0], ["dil_conv_5x5", 1], ["sep_conv_3x3", 1], ["dil_conv_3x3", 2], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 0], ["dil_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["max_pool_3x3", 0], ["dil_conv_3x3", 1], ["max_pool_3x3", 0], ["avg_pool_3x3", 1], ["sep_conv_3x3", 1],
            ["dil_conv_5x5", 3], ["dil_conv_5x5", 3], ["dil_conv_5x5", 4]], reduce_concat=range(2, 6))
init_pt_s2_C10_0 = Genotype(
    normal=[["sep_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["skip_connect", 0], ["skip_connect", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s2_C10_2 = Genotype(
    normal=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["skip_connect", 0],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 1], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["skip_connect", 0], ["skip_connect", 1]], reduce_concat=range(2, 6))
init_pt_s3_C10_0 = Genotype(
    normal=[["sep_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["skip_connect", 0], ["skip_connect", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s3_C10_2 = Genotype(
    normal=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["skip_connect", 0],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 1], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s4_C10_0 = Genotype(
    normal=[["sep_conv_3x3", 0], ["noise", 1], ["noise", 1], ["sep_conv_3x3", 2], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 2], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s4_C10_2 = Genotype(
    normal=[["sep_conv_3x3", 0], ["noise", 1], ["sep_conv_3x3", 1], ["noise", 2], ["sep_conv_3x3", 2],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 2], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["noise", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))

#### cifar100 s1 - s4
init_pt_s1_C100_0 = Genotype(
    normal=[["dil_conv_3x3", 0], ["dil_conv_5x5", 1], ["sep_conv_3x3", 1], ["dil_conv_3x3", 2], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 0], ["dil_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["avg_pool_3x3", 0], ["dil_conv_3x3", 1], ["avg_pool_3x3", 0], ["dil_conv_5x5", 2], ["sep_conv_3x3", 1],
            ["dil_conv_3x3", 2], ["avg_pool_3x3", 1], ["dil_conv_5x5", 2]], reduce_concat=range(2, 6))
init_pt_s1_C100_2 = Genotype(
    normal=[["dil_conv_3x3", 0], ["dil_conv_5x5", 1], ["dil_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 0], ["dil_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["avg_pool_3x3", 0], ["dil_conv_3x3", 1], ["avg_pool_3x3", 1], ["dil_conv_5x5", 2], ["sep_conv_3x3", 1],
            ["dil_conv_5x5", 3], ["dil_conv_5x5", 3], ["dil_conv_5x5", 4]], reduce_concat=range(2, 6))
init_pt_s2_C100_0 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 2], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 1], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["skip_connect", 0], ["skip_connect", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s2_C100_2 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 1], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["skip_connect", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s3_C100_0 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["skip_connect", 1], ["skip_connect", 0],
            ["skip_connect", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s3_C100_2 = Genotype(
    normal=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["skip_connect", 0], ["sep_conv_3x3", 2], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 1], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["skip_connect", 0], ["skip_connect", 1], ["sep_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s4_C100_0 = Genotype(
    normal=[["sep_conv_3x3", 0], ["noise", 1], ["sep_conv_3x3", 1], ["noise", 2], ["sep_conv_3x3", 2],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 2], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s4_C100_2 = Genotype(
    normal=[["noise", 0], ["sep_conv_3x3", 1], ["noise", 0], ["sep_conv_3x3", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 2], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))

#### svhn s1 - s4
init_pt_s1_svhn_0 = Genotype(
    normal=[["dil_conv_3x3", 0], ["dil_conv_5x5", 1], ["sep_conv_3x3", 1], ["dil_conv_3x3", 2], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 0], ["dil_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["avg_pool_3x3", 0], ["dil_conv_3x3", 1], ["avg_pool_3x3", 1], ["dil_conv_5x5", 2], ["sep_conv_3x3", 1],
            ["dil_conv_5x5", 3], ["dil_conv_5x5", 2], ["dil_conv_5x5", 4]], reduce_concat=range(2, 6))
init_pt_s1_svhn_2 = Genotype(
    normal=[["dil_conv_3x3", 0], ["dil_conv_5x5", 1], ["dil_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 0], ["dil_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["max_pool_3x3", 0], ["dil_conv_3x3", 1], ["max_pool_3x3", 0], ["dil_conv_5x5", 2], ["sep_conv_3x3", 1],
            ["dil_conv_5x5", 3], ["avg_pool_3x3", 0], ["dil_conv_5x5", 3]], reduce_concat=range(2, 6))
init_pt_s2_svhn_0 = Genotype(
    normal=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["skip_connect", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s2_svhn_2 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["skip_connect", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["skip_connect", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s3_svhn_0 = Genotype(
    normal=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s3_svhn_2 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 1], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s4_svhn_0 = Genotype(
    normal=[["noise", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["noise", 2], ["sep_conv_3x3", 2],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 2], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1],
            ["noise", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s4_svhn_2 = Genotype(
    normal=[["sep_conv_3x3", 0], ["noise", 1], ["noise", 0], ["sep_conv_3x3", 2], ["sep_conv_3x3", 2],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))

######## DARTS Space ########

####init-100-N10
init_pt_s5_C10_0_100_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 2], ["sep_conv_5x5", 2], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 1],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_1_100_N10 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 1],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 0], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_2_100_N10 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 0], ["sep_conv_5x5", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_3_100_N10 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))

####global op gready
global_pt_s5_C10_0_100_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 1], ["sep_conv_5x5", 4]], reduce_concat=range(2, 6))
global_pt_s5_C10_1_100_N10 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 1],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 0], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 1], ["sep_conv_3x3", 2], ["sep_conv_3x3", 1],
            ["sep_conv_5x5", 3], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
global_pt_s5_C10_2_100_N10 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["skip_connect", 0], ["skip_connect", 1], ["sep_conv_5x5", 0],
            ["dil_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
global_pt_s5_C10_3_100_N10 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 0], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["dil_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))

####2500_sample
sample_2500_0 = Genotype(
    normal=[["dil_conv_5x5", 0], ["dil_conv_5x5", 1], ["dil_conv_5x5", 0], ["skip_connect", 1], ["sep_conv_3x3", 2],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 2], ["dil_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["dil_conv_3x3", 0], ["dil_conv_3x3", 1], ["dil_conv_5x5", 1], ["sep_conv_5x5", 2], ["skip_connect", 0],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 0], ["dil_conv_5x5", 3]], reduce_concat=range(2, 6))
sample_2500_1 = Genotype(
    normal=[["skip_connect", 0], ["skip_connect", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 2], ["dil_conv_5x5", 1],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 1], ["dil_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["dil_conv_3x3", 1], ["sep_conv_5x5", 0], ["avg_pool_3x3", 2], ["dil_conv_5x5", 1],
            ["dil_conv_5x5", 2], ["dil_conv_5x5", 0], ["sep_conv_3x3", 4]], reduce_concat=range(2, 6))
sample_2500_2 = Genotype(
    normal=[["dil_conv_5x5", 0], ["dil_conv_3x3", 1], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 2], ["sep_conv_5x5", 3]], normal_concat=range(2, 6),
    reduce=[["dil_conv_3x3", 0], ["avg_pool_3x3", 1], ["sep_conv_5x5", 0], ["skip_connect", 1], ["sep_conv_3x3", 0],
            ["dil_conv_5x5", 2], ["dil_conv_5x5", 0], ["dil_conv_5x5", 2]], reduce_concat=range(2, 6))
sample_2500_3 = Genotype(
    normal=[["sep_conv_3x3", 0], ["dil_conv_3x3", 1], ["sep_conv_5x5", 1], ["dil_conv_3x3", 2], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 2], ["dil_conv_3x3", 0], ["dil_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["skip_connect", 0], ["dil_conv_3x3", 1], ["dil_conv_5x5", 0], ["max_pool_3x3", 1], ["avg_pool_3x3", 0],
            ["max_pool_3x3", 1], ["avg_pool_3x3", 1], ["skip_connect", 3]], reduce_concat=range(2, 6))

####20000_sample
sample_20000_0 = Genotype(
    normal=[["skip_connect", 0], ["dil_conv_3x3", 1], ["sep_conv_5x5", 1], ["skip_connect", 2], ["dil_conv_5x5", 0],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 2], ["dil_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["skip_connect", 0], ["sep_conv_5x5", 1], ["dil_conv_5x5", 0], ["skip_connect", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 2], ["dil_conv_5x5", 0], ["sep_conv_5x5", 4]], reduce_concat=range(2, 6))
sample_20000_1 = Genotype(
    normal=[["skip_connect", 0], ["skip_connect", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 2], ["dil_conv_5x5", 1],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 1], ["dil_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["dil_conv_3x3", 1], ["sep_conv_5x5", 0], ["avg_pool_3x3", 2], ["dil_conv_5x5", 1],
            ["dil_conv_5x5", 2], ["dil_conv_5x5", 0], ["sep_conv_3x3", 4]], reduce_concat=range(2, 6))
sample_20000_2 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 2], ["dil_conv_5x5", 0],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 1], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["dil_conv_5x5", 0], ["dil_conv_3x3", 1], ["skip_connect", 0], ["max_pool_3x3", 1], ["sep_conv_5x5", 0],
            ["dil_conv_5x5", 1], ["sep_conv_3x3", 1], ["dil_conv_3x3", 3]], reduce_concat=range(2, 6))
sample_20000_3 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["skip_connect", 0], ["dil_conv_3x3", 2], ["dil_conv_3x3", 1],
            ["sep_conv_5x5", 3], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["dil_conv_5x5", 1], ["dil_conv_5x5", 0], ["dil_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 2], ["dil_conv_3x3", 1], ["sep_conv_3x3", 2]], reduce_concat=range(2, 6))

####50000_sample 
sample_50000_0 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["sep_conv_3x3", 0],
            ["skip_connect", 2], ["sep_conv_3x3", 0], ["sep_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["max_pool_3x3", 0], ["sep_conv_5x5", 1], ["avg_pool_3x3", 0], ["dil_conv_3x3", 1], ["sep_conv_5x5", 0],
            ["dil_conv_3x3", 1], ["dil_conv_5x5", 0], ["max_pool_3x3", 1]], reduce_concat=range(2, 6))
sample_50000_1 = Genotype(
    normal=[["dil_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 1],
            ["sep_conv_5x5", 3], ["sep_conv_5x5", 1], ["dil_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["dil_conv_3x3", 0], ["dil_conv_5x5", 1], ["sep_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_5x5", 0],
            ["max_pool_3x3", 1], ["dil_conv_3x3", 1], ["dil_conv_5x5", 2]], reduce_concat=range(2, 6))
sample_50000_2 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 2], ["dil_conv_5x5", 0],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 1], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["dil_conv_5x5", 0], ["dil_conv_3x3", 1], ["skip_connect", 0], ["max_pool_3x3", 1], ["sep_conv_5x5", 0],
            ["dil_conv_5x5", 1], ["sep_conv_3x3", 1], ["dil_conv_3x3", 3]], reduce_concat=range(2, 6))
sample_50000_3 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["skip_connect", 0], ["dil_conv_3x3", 2], ["dil_conv_3x3", 1],
            ["sep_conv_5x5", 3], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["dil_conv_5x5", 1], ["dil_conv_5x5", 0], ["dil_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 2], ["dil_conv_3x3", 1], ["sep_conv_3x3", 2]], reduce_concat=range(2, 6))

#### random
random_max_0 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 3], ["sep_conv_5x5", 1], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
random_max_1 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 2], ["sep_conv_5x5", 1], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
random_max_2 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 3], ["sep_conv_5x5", 1], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
random_max_3 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 2], ["sep_conv_5x5", 1], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))

#### ImageNet-1k
init_pt_s5_in_0_100_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["sep_conv_5x5", 2],
            ["sep_conv_5x5", 3], ["sep_conv_3x3", 0], ["sep_conv_5x5", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["dil_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["skip_connect", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_in_1_100_N10 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 3], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["avg_pool_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_in_2_100_N10 = Genotype(
    normal=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 3], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["skip_connect", 0],
            ["dil_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_in_3_100_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 2], ["sep_conv_5x5", 2],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 0], ["sep_conv_5x5", 3]], normal_concat=range(2, 6),
    reduce=[["dil_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))

####N1
init_pt_s5_C10_0_N1 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_1_N1 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2]], normal_concat=range(2, 6),
    reduce=[["skip_connect", 0], ["sep_conv_5x5", 1], ["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_2_N1 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 0], ["sep_conv_5x5", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["skip_connect", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_3_N1 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))

####N5 

#####V1
init_pt_s5_C10_0_1_N5 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_1_1_N5 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 1],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 1], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_2_1_N5 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 1], ["sep_conv_3x3", 2], ["sep_conv_5x5", 2],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["skip_connect", 1], ["sep_conv_3x3", 0], ["dil_conv_3x3", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_3_1_N5 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))

#####V10
init_pt_s5_C10_0_10_N5 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 2], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_1_10_N5 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1],
            ["sep_conv_5x5", 3], ["sep_conv_5x5", 1], ["sep_conv_5x5", 4]], reduce_concat=range(2, 6))
init_pt_s5_C10_2_10_N5 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 1], ["sep_conv_3x3", 2], ["sep_conv_5x5", 2],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["skip_connect", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_3_10_N5 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))

#####V100
init_pt_s5_C10_0_100_N5 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 1],
            ["sep_conv_3x3", 2], ["sep_conv_5x5", 2], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 1],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_1_100_N5 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 1],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 1], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_2_100_N5 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 1], ["sep_conv_3x3", 2], ["sep_conv_3x3", 1],
            ["sep_conv_5x5", 3], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_3_100_N5 = Genotype(
    normal=[["skip_connect", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 0], ["sep_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))

####N10

#####V1
init_pt_s5_C10_0_1_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_1_1_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 1],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_2_1_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 1], ["sep_conv_3x3", 2], ["sep_conv_5x5", 2],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 0], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["skip_connect", 1], ["sep_conv_3x3", 0], ["dil_conv_3x3", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_3_1_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 3], ["sep_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))

#####V10
init_pt_s5_C10_0_10_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_1_10_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 1],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_2_10_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 2], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 2], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["skip_connect", 0], ["skip_connect", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
init_pt_s5_C10_3_10_N10 = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 1],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6),
    reduce=[["dil_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))

# fisher
cf10_fisher = Genotype(
    normal=[["avg_pool_3x3", 0], ["avg_pool_3x3", 1], ["avg_pool_3x3", 0], ["dil_conv_3x3", 1], ["avg_pool_3x3", 0],
            ["skip_connect", 2], ["sep_conv_5x5", 0], ["dil_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["max_pool_3x3", 0], ["max_pool_3x3", 2], ["sep_conv_3x3", 0],
            ["dil_conv_5x5", 3], ["sep_conv_5x5", 0], ["sep_conv_3x3", 2]], reduce_concat=range(2, 6))
# grasp
cf10_grasp = Genotype(
    normal=[["avg_pool_3x3", 0], ["avg_pool_3x3", 1], ["skip_connect", 0], ["sep_conv_5x5", 1], ["dil_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["sep_conv_5x5", 2], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["skip_connect", 1], ["avg_pool_3x3", 0], ["skip_connect", 1], ["sep_conv_5x5", 0],
            ["skip_connect", 1], ["max_pool_3x3", 1], ["sep_conv_3x3", 3]], reduce_concat=range(2, 6))
# jacob_cov
cf10_jacob_cov = Genotype(
    normal=[["max_pool_3x3", 0], ["dil_conv_3x3", 1], ["dil_conv_3x3", 0], ["sep_conv_3x3", 2], ["dil_conv_3x3", 0],
            ["sep_conv_3x3", 3], ["sep_conv_5x5", 0], ["dil_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["max_pool_3x3", 1], ["max_pool_3x3", 0], ["avg_pool_3x3", 1], ["sep_conv_3x3", 1],
            ["sep_conv_5x5", 3], ["dil_conv_3x3", 1], ["sep_conv_3x3", 4]], reduce_concat=range(2, 6))
# meco
cf10_meco = Genotype(
    normal=[["dil_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["skip_connect", 1], ["dil_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["sep_conv_3x3", 2], ["sep_conv_5x5", 3]], normal_concat=range(2, 6),
    reduce=[["dil_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["dil_conv_5x5", 0],
            ["dil_conv_5x5", 1], ["dil_conv_5x5", 1], ["sep_conv_3x3", 4]], reduce_concat=range(2, 6))
# synflow
cf10_synflow = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
# zico
cf10_zico = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["skip_connect", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
# cf10_zico= Genotype(normal=  [["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_3x3", 1]], normal_concat=range(2, 6), reduce=  [["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["skip_connect", 0], ["sep_conv_3x3", 2]], reduce_concat=range(2, 6))
# snip
cf10_snip = Genotype(
    normal=[["sep_conv_3x3", 0], ["avg_pool_3x3", 1], ["dil_conv_5x5", 0], ["sep_conv_5x5", 1], ["dil_conv_3x3", 1],
            ["sep_conv_3x3", 3], ["sep_conv_3x3", 2], ["sep_conv_5x5", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["avg_pool_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["skip_connect", 1], ["dil_conv_3x3", 0], ["sep_conv_3x3", 4]], reduce_concat=range(2, 6))
# ntk
cf10_ntk = Genotype(
    normal=[["dil_conv_5x5", 0], ["dil_conv_3x3", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0],
            ["dil_conv_5x5", 2], ["dil_conv_5x5", 0], ["sep_conv_5x5", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["dil_conv_5x5", 2],
            ["sep_conv_5x5", 3], ["sep_conv_5x5", 0], ["dil_conv_3x3", 1]], reduce_concat=range(2, 6))

# fisher
cf100_fisher = Genotype(
    normal=[["sep_conv_3x3", 0], ["max_pool_3x3", 1], ["sep_conv_5x5", 0], ["max_pool_3x3", 1], ["dil_conv_3x3", 1],
            ["skip_connect", 3], ["dil_conv_5x5", 0], ["skip_connect", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_3x3", 1], ["dil_conv_3x3", 2], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["max_pool_3x3", 1], ["sep_conv_3x3", 4]], reduce_concat=range(2, 6))
# grasp
cf100_grasp = Genotype(
    normal=[["max_pool_3x3", 0], ["avg_pool_3x3", 1], ["avg_pool_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2],
            ["sep_conv_3x3", 3], ["avg_pool_3x3", 0], ["sep_conv_3x3", 4]], normal_concat=range(2, 6),
    reduce=[["max_pool_3x3", 0], ["sep_conv_3x3", 1], ["dil_conv_3x3", 0], ["dil_conv_3x3", 2], ["skip_connect", 0],
            ["dil_conv_3x3", 1], ["dil_conv_3x3", 1], ["sep_conv_3x3", 2]], reduce_concat=range(2, 6))
# jacob_cov
cf100_jacob_cov = Genotype(
    normal=[["max_pool_3x3", 0], ["avg_pool_3x3", 1], ["dil_conv_3x3", 0], ["dil_conv_5x5", 1], ["avg_pool_3x3", 0],
            ["avg_pool_3x3", 3], ["dil_conv_5x5", 1], ["dil_conv_5x5", 4]], normal_concat=range(2, 6),
    reduce=[["skip_connect", 0], ["sep_conv_5x5", 1], ["avg_pool_3x3", 0], ["skip_connect", 2], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 1], ["dil_conv_3x3", 0], ["dil_conv_5x5", 1]], reduce_concat=range(2, 6))
# meco
cf100_meco = Genotype(
    normal=[["dil_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["dil_conv_5x5", 2], ["sep_conv_5x5", 2],
            ["sep_conv_3x3", 3], ["dil_conv_5x5", 0], ["sep_conv_3x3", 2]], normal_concat=range(2, 6),
    reduce=[["avg_pool_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["dil_conv_5x5", 2], ["sep_conv_3x3", 0],
            ["sep_conv_3x3", 3], ["dil_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
# snip
cf100_snip = Genotype(
    normal=[["sep_conv_5x5", 0], ["skip_connect", 1], ["sep_conv_3x3", 1], ["sep_conv_5x5", 2], ["skip_connect", 0],
            ["sep_conv_3x3", 2], ["dil_conv_3x3", 0], ["max_pool_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["dil_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["skip_connect", 2], ["skip_connect", 0],
            ["skip_connect", 2], ["dil_conv_5x5", 1], ["sep_conv_5x5", 2]], reduce_concat=range(2, 6))
# synflow
cf100_synflow = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 1], ["sep_conv_5x5", 2], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_5x5", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))
# zico
cf100_zico = Genotype(
    normal=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0],
            ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_3x3", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_3x3", 1], ["sep_conv_5x5", 0], ["dil_conv_5x5", 1], ["sep_conv_5x5", 0],
            ["sep_conv_3x3", 2], ["sep_conv_3x3", 0], ["sep_conv_3x3", 1]], reduce_concat=range(2, 6))
# ntk
cf100_ntk = Genotype(
    normal=[["dil_conv_5x5", 0], ["dil_conv_5x5", 1], ["dil_conv_5x5", 0], ["sep_conv_5x5", 1], ["dil_conv_3x3", 0],
            ["dil_conv_5x5", 3], ["sep_conv_3x3", 0], ["sep_conv_5x5", 3]], normal_concat=range(2, 6),
    reduce=[["sep_conv_5x5", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 0], ["sep_conv_5x5", 1], ["sep_conv_3x3", 1],
            ["sep_conv_3x3", 2], ["dil_conv_5x5", 0], ["sep_conv_5x5", 1]], reduce_concat=range(2, 6))

TE_NAS = Genotype(
    normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1),
            ('sep_conv_3x3', 3), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2),
            ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('max_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5])

