import os
import zipfile
from abc import abstractmethod

import torch
from torch.utils.data import Dataset
from utils import makedir_exist_ok, save, load
from .utils import download_url


class Vocab:
    def __init__(self):
        self.symbol_to_index = {}
        self.index_to_symbol = []
        self.symbol_counts = {}

    def add(self, symbol):
        if symbol not in self.symbol_to_index:
            self.index_to_symbol.append(symbol)
            self.symbol_to_index[symbol] = len(self.index_to_symbol) - 1
        return

    def delete(self, symbol):
        if symbol in self.symbol_to_index:
            self.index_to_symbol.remove(symbol)
            self.symbol_to_index.pop(symbol, None)
        return

    def count(self, input):
        if isinstance(input, int):
            count = self.symbol_counts[self.index_to_symbol[input]]
        elif isinstance(input, str):
            count = self.symbol_counts[input]
        else:
            raise ValueError('wrong input data type')
        return count

    def symbol_counts(self):
        counts = sorted(self.symbol_counts.items(), key=lambda x: x[1], reverse=True)
        return counts

    def __len__(self):
        return len(self.index_to_symbol)

    def __getitem__(self, input):
        if isinstance(input, int):
            output = self.index_to_symbol[input]
        elif isinstance(input, str):
            output = self.symbol_to_index[input]
        else:
            raise ValueError('wrong input data type')
        return output

    def __contains__(self, input):
        if isinstance(input, int):
            exist = input in self.index_to_symbol
        elif isinstance(input, str):
            exist = input in self.symbol_to_index
        else:
            raise ValueError('wrong input data type')
        return exist


class LanguageModeling(Dataset):
    def __init__(self, root, download=False):
        self.data_name = None
        self.root = os.path.expanduser(root)
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        self.data, self.vocab = load(self.data_path)

    def _check_exists(self):
        return os.path.exists(self.data_path)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.data_name + '\n'
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    @abstractmethod
    def download(self):
        raise NotImplementedError

    @property
    def data_path(self):
        raise NotImplementedError


class PennTreebank(LanguageModeling):
    data_name = 'PennTreebank'
    urls = ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
           'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
           'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']
    data_file = {'train':'train', 'validation':'valid', 'test':'test'}

    def __init__(self, root, download=False):
        super(PennTreebank, self).__init__(root, download)

    @property
    def data_path(self):
        return os.path.join(self.root, 'ptb')

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(os.path.join(self.root))
        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, root=self.root, filename=filename, md5=None)
        vocab = Vocab()
        data = {}
        for split in self.data_file:
            token_path = os.path.join(self.root, 'ptb.' + self.data_file[split] + '.txt')
            num_tokens = 0
            with open(token_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split() + [u'<eos>']
                    num_tokens += len(line)
                    for symbol in line:
                        vocab.add(symbol)
            with open(token_path, 'r', encoding='utf-8') as f:
                data[split] = torch.LongTensor(num_tokens)
                i = 0
                for line in f:
                    line = line.split() + [u'<eos>']
                    for symbol in line:
                        data[split][i] = vocab.symbol_to_index[symbol]
                        i += 1
        save([data, vocab], self.data_path)
        return


class WikiText2(LanguageModeling):
    data_name = 'WikiText2'
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    data_file = {'train':'train', 'validation':'valid', 'test':'test'}

    def __init__(self, root, download=False):
        super(WikiText2, self).__init__(root, download)

    @property
    def data_path(self):
        return os.path.join(self.root, 'wiki')

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(os.path.join(self.root))
        zipname = self.url.rpartition('/')[2]
        if not os.path.exists(os.path.join(self.root, 'wikitext-2')):
            download_url(self.url, root=self.root, filename=zipname, md5=None)
            with zipfile.ZipFile(os.path.join(self.root, zipname), "r") as f:
                f.extractall(self.root)
        vocab = Vocab()
        data = {}
        for split in self.data_file:
            token_path = os.path.join(self.root, 'wikitext-2', 'wiki.' + self.data_file[split] + '.tokens')
            num_tokens = 0
            with open(token_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split() + [u'<eos>']
                    num_tokens += len(line)
                    for symbol in line:
                        vocab.add(symbol)
            with open(token_path, 'r', encoding='utf-8') as f:
                data[split] = torch.LongTensor(num_tokens)
                i = 0
                for line in f:
                    line = line.split() + [u'<eos>']
                    for symbol in line:
                        data[split][i] = vocab.symbol_to_index[symbol]
                        i += 1
        save([data, vocab], self.data_path)
        return


class WikiText103(LanguageModeling):
    data_name = 'WikiText103'
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
    data_file = {'train':'train', 'validation':'valid', 'test':'test'}

    def __init__(self, root, download=False):
        super(WikiText103, self).__init__(root, download)

    @property
    def data_path(self):
        return os.path.join(self.root, 'wiki')

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(os.path.join(self.root))
        zipname = self.url.rpartition('/')[2]
        if not os.path.exists(os.path.join(self.root, 'wikitext-103')):
            download_url(self.url, root=self.root, filename=zipname, md5=None)
            with zipfile.ZipFile(os.path.join(self.root, zipname), "r") as f:
                f.extractall(self.root)
        vocab = Vocab()
        data = {}
        for split in self.data_file:
            token_path = os.path.join(self.root, 'wikitext-103', 'wiki.' + self.data_file[split] + '.tokens')
            num_tokens = 0
            with open(token_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split() + [u'<eos>']
                    num_tokens += len(line)
                    for symbol in line:
                        vocab.add(symbol)
            with open(token_path, 'r', encoding='utf-8') as f:
                data[split] = torch.LongTensor(num_tokens)
                i = 0
                for line in f:
                    line = line.split() + [u'<eos>']
                    for symbol in line:
                        data[split][i] = vocab.symbol_to_index[symbol]
                        i += 1
        save([data, vocab], self.data_path)
        return
