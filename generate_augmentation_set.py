import os.path
import random

from generate_all import raw_dataset_dir
from generate_all import augmentation_output_dir

class AugmentationSetGenerator:
    def __init__(self, base_name, augmentation_lines, large_size=1215, medium_size=405, small_size=101):
        self.base_name = base_name
        self.origin_lines = augmentation_lines
        self.large = large_size
        self.medium = medium_size
        self.small = small_size

    def generate_augmentation_set(self, output_dir=augmentation_output_dir):
        task_list = [
            {
                'file_name': os.path.join(output_dir, f"{self.base_name}_large.tsv"),
                'lines': self.large
            },
            {
                'file_name': os.path.join(output_dir, f"{self.base_name}_medium.tsv"),
                'lines': self.medium
            },
            {
                'file_name': os.path.join(output_dir, f"{self.base_name}_small.tsv"),
                'lines': self.small
            },
        ]

        for task in task_list:
            print(f"Writing {task['lines']} <{self.base_name}> examples to {task['file_name']}")

            file = open(task['file_name'], 'w', encoding='utf-8')
            random.shuffle(self.origin_lines)
            file.writelines(self.origin_lines[0:task['lines']])
            file.close()

def combine(lines_1: list, lines_2: list, limit):
    comb = lines_1 + lines_2
    random.shuffle(comb)
    return comb[0 : limit]

def main(debug=False):
    inv_orig = open(os.path.join(raw_dataset_dir , 'inv_orig.tsv'), 'r', encoding='utf-8')
    inv_trsf = open(os.path.join(raw_dataset_dir , 'inv_trsf.tsv'), 'r', encoding='utf-8')
    pass_orig = open(os.path.join(raw_dataset_dir , 'pass_orig.tsv'), 'r', encoding='utf-8')
    pass_trsf = open(os.path.join(raw_dataset_dir , 'pass_trsf.tsv'), 'r', encoding='utf-8')
    chaos = open(os.path.join(raw_dataset_dir , 'chaos.tsv'), 'r', encoding='utf-8')

    files = [
        inv_orig,
        inv_trsf,
        pass_orig,
        pass_trsf,
        chaos
    ]

    inv_orig_lines = inv_orig.readlines()
    inv_trsf_lines = inv_trsf.readlines()
    pass_orig_lines = pass_orig.readlines()
    pass_trsf_lines = pass_trsf.readlines()
    chaos_lines = chaos.readlines()
    pass_trsf_neg_lines = [pn for pn in pass_trsf_lines if pn.startswith('1')]
    pass_trsf_pos_lines = [pp for pp in pass_trsf_lines if pp.startswith('2')]
    comb_orig_lines = combine(inv_orig_lines, pass_orig_lines, len(inv_orig_lines))
    comb_trsf_lines = combine(inv_trsf_lines, pass_trsf_lines, len(inv_trsf_lines))
    comb_trsf_neg_lines = combine(inv_trsf_lines, pass_trsf_neg_lines, len(inv_trsf_lines))

    generators = [
        AugmentationSetGenerator('inv_orig', inv_orig_lines),
        AugmentationSetGenerator('inv_trsf', inv_trsf_lines),
        AugmentationSetGenerator('pass_orig', pass_orig_lines),
        AugmentationSetGenerator('pass_trsf', pass_trsf_lines),
        AugmentationSetGenerator('chaos', chaos_lines),
        AugmentationSetGenerator('pass_trsf_neg', pass_trsf_neg_lines),
        AugmentationSetGenerator('pass_trsf_pos', pass_trsf_pos_lines),
        AugmentationSetGenerator('comb_orig', comb_orig_lines),
        AugmentationSetGenerator('comb_trsf', comb_trsf_lines),
        AugmentationSetGenerator('comb_trsf_neg', comb_trsf_neg_lines)
    ]

    for generator in generators:
        generator.generate_augmentation_set()

    for file in files:
        file.close()

if __name__ == '__main__':
    main()
