import generate_augmentation_set
import generate_dataset
import generate_mnli_training_directory

if __name__ == '__main__':
    generate_dataset.main()
    generate_augmentation_set.main()
    generate_mnli_training_directory.generate_all_augmented_training_dir()