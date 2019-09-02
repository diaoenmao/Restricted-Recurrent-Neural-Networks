import datasets


def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    data_name_head = data_name.split('_')[0]
    root = './data/{}'.format(data_name)
    if data_name_head in ['PennTreebank', 'WikiText2', 'WikiText103']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', download=True)'.format(data_name_head))
        dataset['validation'] = eval('datasets.{}(root=root, split=\'validation\', download=True)'.format(data_name_head))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', download=True)'.format(data_name_head))
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset
