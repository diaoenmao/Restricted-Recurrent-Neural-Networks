import os
import shutil
import torch
from .folder import ImageFolder
from .utils import check_integrity, download_url

ARCHIVE_DICT = {
    'train': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'md5': '1d675b47d978889d74fa0da5fadfb00e',
    },
    'validation': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'md5': '29b22e2961454d5413ddabcf34fc5622',
    },
    'devkit': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz',
        'md5': 'fa75699e90414af021442c21a62c3abf',
    }
}

class ImageNet(ImageFolder):
    def __init__(self, root, split='train', download=False, **kwargs):
        self.root = root
        self.split = split
        self.data_name = 'ImageNet'
        if download:
            self.download()
        wnid_to_classes = self._load_meta_file()[0]
        super(ImageNet, self).__init__(self.root, **kwargs)

    def download(self):
        if not check_integrity(self.meta_file):
            tmpdir = os.path.join(self.root, 'tmp')

            archive_dict = ARCHIVE_DICT['devkit']
            download_and_extract_tar(archive_dict['url'], os.path.dirname(self.root),
                                     extract_root=tmpdir,
                                     md5=archive_dict['md5'])
            devkit_folder = _splitexts(os.path.basename(archive_dict['url']))[0]
            meta = parse_devkit(os.path.join(tmpdir, devkit_folder))
            self._save_meta_file(*meta)

            shutil.rmtree(tmpdir)

        if not os.path.isdir(os.path.dirname(self.root)):
            archive_dict = ARCHIVE_DICT[self.split]
            download_and_extract_tar(archive_dict['url'], os.path.dirname(self.root),
                                     extract_root=self.root,
                                     md5=archive_dict['md5'])

            if self.split == 'train':
                prepare_train_folder(self.root)
            elif self.split == 'validation':
                val_wnids = self._load_meta_file()[1]
                prepare_val_folder(self.root, val_wnids)
        else:
            msg = ("You set download=True, but a folder '{}' already exist in "
                   "the root directory. If you want to re-download or re-extract the "
                   "archive, delete the folder.")
            print(msg.format(self.split))

    @property
    def meta_file(self):
        return os.path.join(self.root, 'meta.bin')

    def _load_meta_file(self):
        if check_integrity(self.meta_file):
            return torch.load(self.meta_file)
        else:
            raise RuntimeError("Meta file not found or corrupted.",
                               "You can use download=True to create it.")

    def _save_meta_file(self, wnid_to_class, val_wnids):
        torch.save((wnid_to_class, val_wnids), self.meta_file)

def extract_tar(src, dest=None, gzip=None, delete=False):
    import tarfile

    if dest is None:
        dest = os.path.dirname(src)
    if gzip is None:
        gzip = src.lower().endswith('.gz')

    mode = 'r:gz' if gzip else 'r'
    with tarfile.open(src, mode) as tarfh:
        tarfh.extractall(path=dest)

    if delete:
        os.remove(src)

def download_and_extract_tar(url, download_root, extract_root=None, filename=None,
                             md5=None, **kwargs):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if filename is None:
        filename = os.path.basename(url)

    if not check_integrity(os.path.join(download_root, filename), md5):
        download_url(url, download_root, filename=filename, md5=md5)

    extract_tar(os.path.join(download_root, filename), extract_root, **kwargs)


def parse_devkit(root):
    idx_to_wnid, wnid_to_classes = parse_meta(root)
    val_idcs = parse_val_groundtruth(root)
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
    return wnid_to_classes, val_wnids


def parse_meta(devkit_root, path='data', filename='meta.mat'):
    import scipy.io as sio

    metafile = os.path.join(devkit_root, path, filename)
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes

def parse_val_groundtruth(devkit_root, path='data',
                          filename='ILSVRC2012_validation_ground_truth.txt'):
    with open(os.path.join(devkit_root, path, filename), 'r') as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]

def prepare_train_folder(folder):
    for archive in [os.path.join(folder, archive) for archive in os.listdir(folder)]:
        extract_tar(archive, os.path.splitext(archive)[0], delete=True)

def prepare_val_folder(folder, wnids):
    img_files = sorted([os.path.join(folder, file) for file in os.listdir(folder)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(folder, wnid))

    for wnid, img_file in zip(wnids, img_files):
        shutil.move(img_file, os.path.join(folder, wnid, os.path.basename(img_file)))

def _splitexts(root):
    exts = []
    ext = '.'
    while ext:
        root, ext = os.path.splitext(root)
        exts.append(ext)
    return root, ''.join(reversed(exts))
