from euro_sat_archive import load_batch


def get_dataset(train=True, archive_dim=0):
    if archive_dim == 0:
        archive_dim = 22950 if train else 27000 - 22950

    images, labels = load_batch(archive_dim, train)
    return images, labels