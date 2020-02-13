import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
from .omniglot import Omniglot  
from .utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler 
from .concat import ConcatDataset 


def OmniglotLoader(root, batch_size, subfolder_name='images_background', num_workers=2,  aug=None, shuffle=True):
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    if aug==None:
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            binary_flip,
            normalize
            ])
    elif aug=='once':
        transform=transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize
        ]))
    elif aug=='ktimes':
         transform = TransformKtimes(transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.RandomAffine(degrees = (-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.2), shear = (-10, 10), fillcolor=255),
            transforms.ToTensor(),
            binary_flip,
            normalize
        ]), k=10)

    dataset = Omniglot(root=root, subfolder_name=subfolder_name, transform=transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def alphabetLoader(root, alphabet, batch_size, subfolder_name='images_evaluation', aug=None, num_workers=2, shuffle=False):
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    if aug==None:
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            binary_flip,
            normalize
            ])
    elif aug=='once':
        transform=transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize
        ]))
    elif aug=='ktimes':
         transform = TransformKtimes(transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.RandomAffine(degrees = (-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.2), shear = (-10, 10), fillcolor=255),
            transforms.ToTensor(),
            binary_flip,
            normalize
        ]), k=10)

    dataset = Omniglot(root=root, subfolder_name=subfolder_name, transform=transform)
    # Only use the images which has alphabet-name in their path name (_characters[cid])
    valid_flat_character_images = [(imgname,cid) for imgname,cid in dataset._flat_character_images if alphabet in dataset._characters[cid]]
    ndata = len(valid_flat_character_images)  # The number of data after filtering
    imgid2cid = [valid_flat_character_images[i][1] for i in range(ndata)]  # The tuple (valid_flat_character_images[i]) are (img, cid)
    cid_set = set(imgid2cid)  # The labels are not 0..c-1 here.
    cid2ncid = {cid:ncid for ncid,cid in enumerate(cid_set)}  # Create the mapping table for New cid (ncid)
    valid_characters = {cid2ncid[cid]:dataset._characters[cid] for cid in cid_set}
    for i in range(ndata):  # Convert the labels to make sure it has the value {0..c-1}
        valid_flat_character_images[i] = (valid_flat_character_images[i][0],cid2ncid[valid_flat_character_images[i][1]])
    # Apply surgery to the dataset
    dataset._flat_character_images = valid_flat_character_images
    dataset._characters = valid_characters
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    loader.num_classes = len(cid_set)
    print('=> Alphabet %s has %d characters and %d images.'%(alphabet, loader.num_classes, len(dataset)))
    return loader 

def OmniglotLoaderMix(root, alphabet, batch_size, num_workers=2,  aug=None, shuffle=False, unlabeled_batch_size=32):
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    if aug==None:
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            binary_flip,
            normalize
            ])
    elif aug=='once':
        transform=transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize
        ]))
    elif aug=='ktimes':
         transform = TransformKtimes(transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.RandomAffine(degrees = (-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.2), shear = (-10, 10), fillcolor=255),
            transforms.ToTensor(),
            binary_flip,
            normalize
        ]), k=10)

    dataset_labeled = Omniglot(root=root, subfolder_name='images_background', transform=transform)
    dataset_unlabeled = Omniglot(root=root, subfolder_name='images_evaluation', transform=transform)
    # Only use the images which has alphabet-name in their path name (_characters[cid])
    valid_flat_character_images = [(imgname, cid) for imgname,cid in dataset_unlabeled._flat_character_images if alphabet in dataset_unlabeled._characters[cid]] 
    ndata = len(valid_flat_character_images)  # The number of data after filtering
    imgid2cid = [valid_flat_character_images[i][1] for i in range(ndata)]  # The tuple (valid_flat_character_images[i]) are (img, cid)
    cid_set = set(imgid2cid)  # The labels are not 0..c-1 here.
    cid2ncid = {cid:ncid for ncid,cid in enumerate(cid_set)}  # Create the mapping table for New cid (ncid)
    valid_characters = {cid2ncid[cid]:dataset_unlabeled._characters[cid] for cid in cid_set}
    for i in range(ndata):  # Convert the labels to make sure it has the value {0..c-1}
        valid_flat_character_images[i] = (valid_flat_character_images[i][0],cid2ncid[valid_flat_character_images[i][1]])
    # Apply surgery to the dataset
    dataset_unlabeled._flat_character_images = valid_flat_character_images
    dataset_unlabeled._characters = valid_characters
    dataset= ConcatDataset((dataset_labeled, dataset_unlabeled))
    labeled_idxs = range(len(dataset_labeled)) 
    unlabeled_idxs = range(len(dataset_labeled), len(dataset_labeled)+len(dataset_unlabeled))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size) 
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    loader.labeled_length = len(dataset_labeled)
    loader.unlabeled_length = len(dataset_unlabeled)
    return loader


def alphabetData(root, alphabet, batch_size, subfolder_name='images_evaluation', aug=None):
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    if aug==None:
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            binary_flip,
            normalize
            ])
    elif aug=='once':
        transform=transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize
        ]))
    elif aug=='ktimes':
         transform = TransformKtimes(transforms.Compose([
            transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.RandomAffine(degrees = (-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.2), shear = (-10, 10), fillcolor=255),
            transforms.ToTensor(),
            binary_flip,
            normalize
        ]), k=10)

    dataset = Omniglot(root=root, subfolder_name=subfolder_name, transform=transform)
    # Only use the images which has alphabet-name in their path name (_characters[cid])
    valid_flat_character_images = [(imgname,cid) for imgname,cid in dataset._flat_character_images if alphabet in dataset._characters[cid]]
    ndata = len(valid_flat_character_images)  # The number of data after filtering
    imgid2cid = [valid_flat_character_images[i][1] for i in range(ndata)]  # The tuple (valid_flat_character_images[i]) are (img, cid)
    cid_set = set(imgid2cid)  # The labels are not 0..c-1 here.
    cid2ncid = {cid:ncid for ncid,cid in enumerate(cid_set)}  # Create the mapping table for New cid (ncid)
    valid_characters = {cid2ncid[cid]:dataset._characters[cid] for cid in cid_set}
    for i in range(ndata):  # Convert the labels to make sure it has the value {0..c-1}
        valid_flat_character_images[i] = (valid_flat_character_images[i][0],cid2ncid[valid_flat_character_images[i][1]])
    # Apply surgery to the dataset
    dataset._flat_character_images = valid_flat_character_images
    dataset._characters = valid_characters
    num_classes = len(cid_set)
    print('=> Alphabet %s has %d characters and %d images.'%(alphabet, num_classes, len(dataset)))
    return dataset, num_classes 


def alphabetLoaderMix(root, labeled_alphabet, unlabeled_alphabet, batch_size, num_workers=2,  aug=None, shuffle=False, unlabeled_batch_size=64):
    dataset_labeled, num_labeled_classes = alphabetData(root, labeled_alphabet, batch_size, subfolder_name='images_background', aug=aug)
    dataset_unlabeled, num_unlabeled_classes = alphabetData(root, unlabeled_alphabet, batch_size, subfolder_name='images_evaluation', aug=aug)
    dataset= ConcatDataset((dataset_labeled, dataset_unlabeled))
    labeled_idxs = range(len(dataset_labeled)) 
    unlabeled_idxs = range(len(dataset_labeled), len(dataset_labeled)+len(dataset_unlabeled))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size) 
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    loader.num_labeled_classes = num_labeled_classes
    loader.num_unlabeled_classes = num_unlabeled_classes
    loader.labeled_length = len(dataset_labeled)
    loader.unlabeled_length = len(dataset_unlabeled)
    return loader


omniglot_background_alphabets=[
    'Alphabet_of_the_Magi', 
    'Gujarati',
    'Anglo-Saxon_Futhorc',  
    'Hebrew',  
    'Arcadian',  
    'Inuktitut_(Canadian_Aboriginal_Syllabics)',
    'Armenian',  
    'Japanese_(hiragana)',
    'Asomtavruli_(Georgian)',                     
    'Japanese_(katakana)',
    'Balinese',                                   
    'Korean',
    'Bengali',                                    
    'Latin',
    'Blackfoot_(Canadian_Aboriginal_Syllabics)',  
    'Malay_(Jawi_-_Arabic)',
    'Braille',                                    
    'Mkhedruli_(Georgian)',
    'Burmese_(Myanmar)',                          
    'N_Ko',
    'Cyrillic',                                   
    'Ojibwe_(Canadian_Aboriginal_Syllabics)',
    'Early_Aramaic',                              
    'Sanskrit',
    'Futurama',                                   
    'Syriac_(Estrangelo)',
    'Grantha',                                    
    'Tagalog',
    'Greek',                                     
    'Tifinagh'
        ]

omniglot_evaluation_alphabets_mapping = {
    'Malayalam':'Malayalam',
     'Kannada':'Kannada',
     'Syriac':'Syriac_(Serto)',
     'Atemayar_Qelisayer':'Atemayar_Qelisayer',
     'Gurmukhi':'Gurmukhi',
     'Old_Church_Slavonic':'Old_Church_Slavonic_(Cyrillic)',
     'Manipuri':'Manipuri',
     'Atlantean':'Atlantean',
     'Sylheti':'Sylheti',
     'Mongolian':'Mongolian',
     'Aurek':'Aurek-Besh',
     'Angelic':'Angelic',
     'ULOG':'ULOG',
     'Oriya':'Oriya',
     'Avesta':'Avesta',
     'Tibetan':'Tibetan',
     'Tengwar':'Tengwar',
     'Keble':'Keble',
     'Ge_ez':'Ge_ez',
     'Glagolitic':'Glagolitic'
}
