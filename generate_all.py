import os

## 所有需要配置的路径都在这里了，关于运行的方法，README.md文件中有详细的介绍，这里我已经配置好所有需要的库，和所有的路径，可以直接运行。
## 在运行之前，我们先删除所有之前运行所产生的数据集。
raw_dataset_dir = ".\\raw"
augmentation_output_dir = ".\\datasets"
mnli_dir = os.path.expanduser('D:\\Documents\\se_learning\\automate-test\\syntactic-augmentation-nli\\multinli_1.0')
origin_training_dir = ".\\MNLI_1-1000\\MNLI"
augmentation_training_set_dir = ".\\MNLI_1-1000"
path_to_bert_base_modle = "D:\\Documents\\se_learning\\automate-test\\uncased_L-12_H-768_A-12"
path_to_bert_train_code = "D:\\Documents\\se_learning\\automate-test\\bert-master"
path_to_hans = "D:\\Documents\\se_learning\\automate-test\\hans"
path_to_hans_test_set = "D:\\Documents\\se_learning\\automate-test\\hans\\berts_of_a_feather"


import generate_augmentation_set
import generate_dataset
import generate_mnli_training_directory





## 现在我们点击下面的箭头开始运行
## 可以看到，运行已经开始了。。。
if __name__ == '__main__':
    generate_dataset.main()
    generate_augmentation_set.main()
    generate_mnli_training_directory.generate_all_augmented_training_dir()