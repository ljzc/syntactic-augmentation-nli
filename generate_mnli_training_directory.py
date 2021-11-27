import os
import shutil
import time

origin_training_dir = ".\\MNLI_1-1000\\MNLI"
output_dir = ".\\MNLI_1-1000"
augmentation_set_dir = ".\\datasets"

augmentation_types = [
    'inv_orig',
    'inv_trsf',
    'pass_orig',
    'pass_trsf',
    'chaos',
    'pass_trsf_neg',
    'pass_trsf_pos',
    'comb_orig',
    'comb_trsf',
    'comb_trsf_neg'
]

augmentation_sizes = [
    'large',
    'medium',
    'small'
]




training_bat_template = r"""
SET BERT_BASE_DIR=D:\\Documents\\se_learning\\automate-test\\uncased_L-12_H-768_A-12
SET DATA_DIR={data_dir}
SET OUTPUT_DIR={train_output_dir}
SET BERT_DIR=D:/Documents/se_learning/automate-test/bert-master

:: #Finetune BERT and evaluate on MNLI

cd %BERT_DIR%

python run_classifier.py ^
  --task_name=MNLI ^
  --do_train=true ^
  --do_eval=true ^
  --data_dir=%DATA_DIR% ^
  --vocab_file=%BERT_BASE_DIR%/vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%/bert_config.json ^
  --init_checkpoint=%BERT_BASE_DIR%/bert_model.ckpt ^
  --max_seq_length=128 ^
  --train_batch_size=2 ^
  --learning_rate=2e-5 ^
  --num_train_epochs=3.0 ^
  --output_dir=%OUTPUT_DIR%
pause
"""

hans_predict_bat_template = r"""
SET BERT_BASE_DIR=D:\\Documents\\se_learning\\automate-test\\uncased_L-12_H-768_A-12
SET HANS_DIR=D:\\Documents\\se_learning\\automate-test\\hans\\berts_of_a_feather
SET TRAINED_CLASSIFIER={train_output_dir}
SET OUTPUT_DIR={predict_output_dir}
SET BERT_DIR=D:/Documents/se_learning/automate-test/bert-master
cd %BERT_DIR%
python run_classifier.py ^
  --task_name=MNLI ^
  --do_predict=true ^
  --data_dir=%HANS_DIR% ^
  --vocab_file=%BERT_BASE_DIR%\\vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%\\bert_config.json ^
  --init_checkpoint=%TRAINED_CLASSIFIER% ^
  --max_seq_length=128 ^
  --output_dir=%OUTPUT_DIR%
pause
"""

hans_evaluate_bat_template = r"""
SET PREDICT_OUTPUT={predict_output_dir}
SET HANS_DIR=D:\\Documents\\se_learning\\automate-test\\hans


cd %HANS_DIR%\\berts_of_a_feather\\files_for_replication

python process_test_results.py %PREDICT_OUTPUT%

cd %HANS_DIR%
python evaluate_heur_output.py %PREDICT_OUTPUT%\\preds.txt

pause
"""

def generate_augmented_training_dir(aug_type, size):
    if aug_type not in augmentation_types:
        print(f'Augmentation type {aug_type} not exists! Chose in {augmentation_types}')
        return
    if size not in augmentation_sizes:
        print(f'Augmentation size {size} not exists! Chose in {augmentation_sizes}')
        return
    generated_training_dir = os.path.join(output_dir, f'{aug_type}_{size}_{int(time.time())}')
    shutil.copytree(origin_training_dir, generated_training_dir)
    train_tsv = open(os.path.join(generated_training_dir, 'train.tsv'), 'a', encoding='utf-8')
    aug_tsv = open(os.path.join(augmentation_set_dir, f'{aug_type}_{size}.tsv'), 'r', encoding='utf-8')
    train_tsv.writelines(aug_tsv.readlines())
    train_tsv.close()
    aug_tsv.close()

    data_dir = os.path.abspath(generated_training_dir)
    train_output_dir = os.path.join(data_dir, "train_output")
    predict_output_dir = os.path.join(data_dir, "predict_output")
    os.mkdir(train_output_dir)
    os.mkdir(predict_output_dir)

    train_bat = open(os.path.join(data_dir, "train.bat"), 'w', encoding='utf-8')
    hans_predict_bat = open(os.path.join(data_dir, "hans_predict.bat"), 'w', encoding='utf-8')
    hans_evaluate_bat = open(os.path.join(data_dir, "hans_evaluate.bat"), 'w', encoding='utf-8')

    train_bat.write(training_bat_template.format(data_dir=data_dir.replace("\\", "\\\\"), train_output_dir=train_output_dir.replace("\\", "\\\\")))
    hans_predict_bat.write(hans_predict_bat_template.format(train_output_dir=train_output_dir.replace("\\", "\\\\"), predict_output_dir=predict_output_dir.replace("\\", "\\\\")))
    hans_evaluate_bat.write(hans_evaluate_bat_template.format(predict_output_dir=predict_output_dir.replace("\\", "\\\\")))

    train_bat.close()
    hans_predict_bat.close()
    hans_evaluate_bat.close()

    print(f'dir <{generated_training_dir}> is successfully generated')

def generate_all_augmented_training_dir():
    for aug_type in augmentation_types:
        for aug_size in augmentation_sizes:
            generate_augmented_training_dir(aug_type, aug_size)

if __name__ == '__main__':
    generate_augmented_training_dir('inv_orig', 'small')