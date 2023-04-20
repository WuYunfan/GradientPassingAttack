from dataset import get_dataset


def process_dataset(name):
    dataset_config = {'name': name + 'Dataset', 'path': 'data/' + name,
                      'device': 'cpu', 'split_ratio': [0.8, 0.2], 'min_inter': 15}
    dataset = get_dataset(dataset_config)
    dataset.output_dataset('data/' + name + '/time')


def main():
    process_dataset('Yelp')


if __name__ == '__main__':
    main()