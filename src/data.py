import datasets


def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name == 'WikiText2':
        dataset = datasets.WikiText2(root, download=True)
    elif data_name == 'WikiText103':
        dataset = datasets.WikiText103(root, download=True)
    elif data_name == 'PennTreebank':
        dataset = datasets.PennTreebank(root, download=True)
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset
