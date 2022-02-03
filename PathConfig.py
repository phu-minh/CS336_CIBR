from os.path import dirname, join, realpath


class FilePaths:
    root = dirname(realpath(__file__))

    dataset = join(dirname(root), 'Dataset')
    upload = join(dirname(root), 'Upload')
    result = join(dirname(root), 'Result')

    train_set = join(dataset, 'train')
    test_set = join(dataset, 'test')
    groundtruth = join(dataset, 'groundtruth')

    compute_ap_exe = join(groundtruth, 'compute_ap')
    
    features_db = join(root, 'output/features.hdf5')
    bovw_db = join(root, 'output/bovw.hdf5')
    codebook = join(root, 'output/vocab.cpickle')
    idf = join(root, 'output/idf.cpickle')
    query = join(root, 'output')
