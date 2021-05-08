from dataset.loaders import get_caltech256_loader, get_cifar_100_loader, get_cifar_10_loader, get_imagenet_loader, get_svhn_loader

def build_data_loader(dataset_name, batch_size, test_batch_size=100, root='./data', num_workers=8):
    if dataset_name == 'imagenet-mini':
        return get_imagenet_loader(root=root, batch_size=batch_size, test_batch_size=test_batch_size, num_workers=num_workers)
    elif dataset_name == 'cifar100':
        return get_cifar_100_loader(root=root, batch_size=batch_size, test_batch_size=test_batch_size, num_workers=num_workers)
    elif dataset_name == 'cifar10':
        return get_cifar_10_loader(root=root, batch_size=batch_size, test_batch_size=test_batch_size, num_workers=num_workers)
    elif dataset_name == 'svhn':
        return get_svhn_loader(root=root, batch_size=batch_size, test_batch_size=test_batch_size, num_workers=num_workers)
    else:
        raise NotImplementedError()