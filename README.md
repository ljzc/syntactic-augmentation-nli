# 工具运行前的准备

## 下载

- 安装tensorflow框架，如果使用bert原本的模型训练代码，则安装python2和tensorflow1.11.0
- 下载 [MultiNLI]([MultiNLI (nyu.edu)](https://cims.nyu.edu/~sbowman/multinli/)) 的json文件`multinli_1.0_train.jsonl`
- 下载[bert](https://github.com/google-research/bert)，并根据bert的说明下载预训练模型
- 下载[MNLI训练集](https://dl.fbaipublicfiles.com/glue/data/MNLI.zip)

-  下载[HANS](https://github.com/tommccoy1/hans)测试集

## 路径配置

`generate_all.py`

```python
raw_dataset_dir = ".\\raw" ## 用来放置生成的五个基本扩增集的文件目录
augmentation_output_dir = ".\\datasets" ## 用来放置生成的30个实验扩增集的目录
mnli_dir = os.path.expanduser('D:\\Documents\\se_learning\\automate-test\\syntactic-augmentation-nli\\multinli_1.0') ## 放置multinli_1.0_train.jsonl的文件夹
origin_training_dir = ".\\MNLI_1-1000\\MNLI" ## 放置原本的MNLI训练集的文件夹
augmentation_training_set_dir = ".\\MNLI_1-1000" ## 放置生成好的扩增训练集和命令行脚本的文件夹
path_to_bert_base_modle = "..\\uncased_L-12_H-768_A-12" ## 存放预训练模型的文件夹
path_to_bert_train_code = "..\\bert-master" ## bert模型训练代码的文件夹，其中由run_classifier.py
path_to_hans = "..\\hans" ## hans项目文件夹
path_to_hans_test_set = "..\\hans\\berts_of_a_feather" ## hans训练集文件夹


```

## 运行

运行generate_all.py文件，可以在设置的`augmentation_training_set_dir`目录下看到所生成的带扩增的MNLI训练集，每个子目录是一个训练集，其中包含三个脚本：

| **脚本**          | **介绍**                               |
| ----------------- | -------------------------------------- |
| train.bat         | 运行扩增后的训练集，生成训练好的模型   |
| hans_predict.bat  | 使用训练好的模型在hans测试集上进行预测 |
| hans_evaluate.bat | 评估hans测试集上预测的结果             |

依次运行这三个脚本即可完成论文中的实验。

更多内容参考[原版readme文件]([syntactic-augmentation-nli/README.md at master · Aatlantise/syntactic-augmentation-nli (github.com)](https://github.com/Aatlantise/syntactic-augmentation-nli/blob/master/README.md))

