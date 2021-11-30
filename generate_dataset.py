import unicodecsv as csv
import json
import os

import nltk
from pattern.text import en
import spacy
import random


from generate_all import raw_dataset_dir as output_dir
from generate_all import mnli_dir as mnli_dir

mnli_train = os.path.join(mnli_dir, 'multinli_1.0_train.jsonl')
mnli_headers = ['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse',
                'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'gold_label']

nlp = spacy.load('en_core_web_sm')
ner = nlp.create_pipe('ner')
parser = nlp.create_pipe('parser')
lemmatizer = nltk.stem.WordNetLemmatizer()


def lower_first(s):
    return s[0].lower() + s[1:]


def upper_first(s):
    return s[0].upper() + s[1:]


def tsv(file):
    return csv.writer(file, delimiter='\t', encoding='utf-8')


def mnli_row(writer, i, premise, hypothesis, label):
    row = [str(i)] + ['ba'] * 7 + [premise, hypothesis, 'ba', label]
    writer.writerow(row)


r"""
    用来获取动词短语中的动词
"""


def get_vp_head(vp):
    head = None
    if vp.label() == 'VP':
        while True: ## 这个循环用来跳过所有的情态动词和be动词，找到真正的实义动词。
            nested_vps = [x for x in vp[1:] if x.label() == 'VP']
            if len(nested_vps) == 0:
                break
            vp = nested_vps[0]
        if vp[0].label().startswith('VB'):
            head = vp[0][0].lower()

    return head, vp[0].label()


r"""
    用来将动词短语变成被动形式
"""


def passivize_vp(vp, subj_num=en.SINGULAR):
    head = None
    flat = vp.flatten()
    if vp.label() == 'VP':
        nesters = []
        while True:
            nesters.append(vp[0][0])
            nested_vps = [x for x in vp[1:] if x.label() == 'VP']
            if len(nested_vps) == 0:
                break
            vp = nested_vps[0]
        label = vp[0].label()
        if label.startswith('VB'):
            head = vp[0][0].lower()
            if len(nesters) > 1:  # 如果动词短语中的动词多于1个说明可能存在情态动词，也就是类似于can do，must do等等，这时be动词在被动形式中应该使用原型
                passivizer = 'be'
            elif label in ['VBP', 'VB', 'VBZ']:  # 如果此时第一个动词时原型或者第三人称单数形式，说明该句的时态为一般现在时，这是需要根据宾语的单复数形式来决定被动句中be动词的形式
                # 'VB' here (not nested) is a POS tag error
                passivizer = 'are' if subj_num == en.PLURAL else 'is'
            elif label == 'VBD' or label == 'VBN':  # 如果该词为过去式或者过去分词（标准情况下二者相同）表示当前句子的时态为过去时（之前已经排除了所有的被动语态和过去完成时）
                # 'VBN' here (not nested) is a POS tag error
                passivizer = 'were' if subj_num == en.PLURAL else 'was'
                # Alternatively, figure out the number of the subject
                # to decide whether it's was or were
            else:
                passivizer = 'is'
            vbn = en.conjugate(head, 'ppart')  # 将动词变成过去分词

            return f'{passivizer} {vbn} by'


r"""
@param np: 名词短语（可能是，也可能不是）
"""


def get_np_head(np):
    head = None
    if np.label() == 'NP' and np[0].label() == 'DT':
        # 要满足这个传入的组成部分是一个名词短语，并且这个名词短语的第一个词是一个限定词
        head_candidates = [x for x in np[1:] if x.label().startswith('NN')]
        # NN开头的代表名词，包括名词原型、复数、专有名词、专有名词复数
        if len(head_candidates) == 1:  # 只接受含有一个名词的名词短语
            # > 1: Complex noun phrases unlikely to be useful
            # 0: Pronominal subjects like "many"
            head = lemmatizer.lemmatize(head_candidates[0][0])
    return head


r"""
用来获取名词的形式（单数或复数）
"""


def get_np_number(np):
    number = None
    if np[0].label() == 'NP':  # 名词短语可能会由一个名词短语和一个介词短语来构成，这一步会取出前面的名词短语而保留之前的名词短语
        np = np[0]
    head_candidates = [x for x in np if x.label().startswith('NN')]  # 对于极其复杂的名词短语，这个方法将会返回none
    if len(head_candidates) == 1:
        label = head_candidates[0].label()
        number = en.PLURAL if label == 'NNS' else en.SINGULAR
    elif len(head_candidates) > 1:
        number = en.PLURAL
    return number


def get_chaos_idx_list(lines, limit=3201):
    idx_list = list(range(0, len(lines)))
    random.shuffle(idx_list)
    return idx_list[0:limit]


def get_shuffled_sentence(sentence: str):
    words = sentence.split(" ")
    random.shuffle(words)
    return ' '.join(words)


def get_random_label():
    return random.choice(("entailment", "neutral", "contradiction"))


r"""
获取时态
"""


def get_tense(vp):
    lookup = en.tenses(vp)

    if len(lookup) == 0:
        if vp[-2:]:
            tense = en.PAST
        else:
            tense = en.PRESENT
    else:
        if en.tenses(vp)[0][0] == u'past':
            tense = en.PAST
        else:
            tense = en.PRESENT

    return tense


def write_out_chaos_lines(lines, w_chaos):
    chaos_idx = get_chaos_idx_list(lines)
    for i, line_idx in enumerate(chaos_idx):
        j = json.loads(lines[line_idx])
        mnli_row(w_chaos, 1000000 + i,
                 get_shuffled_sentence(j['sentence1']),
                 get_shuffled_sentence(j['sentence2']),
                 get_random_label())


def generate_chaos_tsv(lines, file_path=None):
    if file_path is None:
        file_path = os.path.join(output_dir, 'chaos.tsv')
    file = open(file_path, 'wb')
    w_chaos = tsv(file)
    write_out_chaos_lines(lines, w_chaos)
    file.close()


def find_vp(sentence):
    vp = None
    k = 1

    while (sentence[k].label() not in (u'VP', u'SBAR', u'ADJP')) and (k < len(sentence) - 1):
        k += 1

    if k != len(sentence) - 1:  # 排除没有谓语（动词短语）的句子
        vp = sentence[k]
    # iterate through top level branches to find VP
    return vp


class SentenceConvertor:
    r"""
    关于语法树的结构：
    将英文表示成下面这种上下文无关文法：
    S表示句子， → 表示改写（rewrite），NP表示名词词组，VP表示动词词组。以上表达式意为“句子可改写为（或定义为）名词词组加上动词词组。” 语类规则的特点之一是形式化。用一系列符号代替语类、关系和特征。
    下面是各种语类规则的表达式：
    S→NP VP
    NP→D (A) N （D代表限定词，A代表形容词，括号代表可以有可无，N表名词）
    VP→V NP
    VP→V PP （PP代表介词短语）
    VP→V S' （S'代表从句）
    PP→P NP （P代表介词）
    S→NP I VP （I代表助动词或动词的形态变化）
    S'→ C S （S'代表超句，C代表标句词，如that）
    语类规则的特点之二是演绎法。例如从表达式S→NP VP作如下推导：
    S→NP VP（意为:句子由名词词组和动词词组构成）
    NP→D (A ) N （意为: 名词词组由限定词，或许还有形容词，和名词构成）
    VP→V NP（意为: 动词词组由动词和名词词组构成）

    https://www.douban.com/note/557031585/



    POS tag list:

    CC  coordinating conjunction
    CD  cardinal digit
    DT  determiner
    EX  existential there (like: "there is" ... think of it like "there exists")
    FW  foreign word
    IN  preposition/subordinating conjunction
    JJ  adjective   'big'
    JJR adjective, comparative  'bigger'
    JJS adjective, superlative  'biggest'
    LS  list marker 1)
    MD  modal   could, will
    NN  noun, singular 'desk'
    NNS noun plural 'desks'
    NNP proper noun, singular   'Harrison'
    NNPS    proper noun, plural 'Americans'
    PDT predeterminer   'all the kids'
    POS possessive ending   parent's
    PRP personal pronoun    I, he, she
    PRP$    possessive pronoun  my, his, hers
    RB  adverb  very, silently,
    RBR adverb, comparative better
    RBS adverb, superlative best
    RP  particle    give up
    TO  to  go 'to' the store.
    UH  interjection    errrrrrrrm
    VB  verb, base form take
    VBD verb, past tense    took
    VBG verb, gerund/present participle taking
    VBN verb, past participle   taken
    VBP verb, sing. present, non-3d take
    VBZ verb, 3rd person sing. present  takes
    WDT wh-determiner   which
    WP  wh-pronoun  who, what
    WP$ possessive wh-pronoun   whose
    WRB wh-abverb   where, when
    """
    def __init__(self, sentence):
        self.sentence_valid = True
        self.sentence = sentence

        if len(self.sentence) < 2:  # Not a full NP + VP sentence
            self.sentence_valid = False
            return

        self.subj_head = get_np_head(self.sentence[0])

        if self.subj_head is None:  # 排除主语不符合要求的句子
            self.sentence_valid = False
            return

        self.subject_number = get_np_number(self.sentence[0])
        self.vp = find_vp(self.sentence)

        if self.vp is None:
            self.sentence_valid = False
            return

        self.vp_head = get_vp_head(self.vp)

        if self.vp_head[0] is None:  # 排除动词短语不和规则的情况
            self.sentence_valid = False
            return

        self.subj = ' '.join(self.sentence[0].flatten())

        arguments = tuple(x.label() for x in self.sentence[1][1:])

        if (arguments != ('NP',) or
                # arguments代表的应该是动词之后的部分，要保留的句子中这个部分应该只有一个名词短语,
                # 可能排除的情况有：宾语从句、介宾短语、情态动词加动词词组、被动语态、完成时、虚拟语气等等、由直接宾语和间接宾语两个宾语组成的句子
                en.lemma(self.vp_head[0]) in ['be', 'have']):  # 这里是为了排除类似于he is ...或者he has ...的句子
            self.sentence_valid = False
            return

        self.direct_object = ' '.join(sentence[1][1].flatten())
        self.object_number = get_np_number(sentence[1][1])

        if self.object_number is None:  # Personal pronoun, very complex NP, or parse error
            self.sentence_valid = False
            return

    def get_subjobj_rev_hyp(self):
        tense = get_tense(self.vp_head[0])
        return ' '.join([
            upper_first(self.direct_object),
            # keep tense
            en.conjugate(self.vp_head[0], number=self.object_number, tense=tense),
            lower_first(self.subj)]) + '.'

    def get_passive_hyp_same_meaning(self):
        return ' '.join([
            upper_first(self.direct_object),
            passivize_vp(self.vp, self.object_number),
            lower_first(self.subj)]) + '.'

    def get_passive_hyp_inverted(self):
        return ' '.join([
            self.subj,
            passivize_vp(self.vp, self.subject_number),
            self.direct_object]) + '.'

    def is_sentence_valid(self):
        return self.sentence_valid




def main(debug=False):
    files = [
        open(os.path.join(output_dir,'inv_orig.tsv'), 'wb'),
        open(os.path.join(output_dir , 'inv_trsf.tsv'), 'wb'),
        open(os.path.join(output_dir , 'pass_orig.tsv'), 'wb'),
        open(os.path.join(output_dir , 'pass_trsf.tsv'), 'wb')
    ]
    w_inv_orig = tsv(files[0])
    w_inv_trsf = tsv(files[1])
    w_pass_orig = tsv(files[2])
    w_pass_trsf = tsv(files[3])

    lines = open(mnli_train).readlines()
    n = 0

    generate_chaos_tsv(lines)

    for i, line in enumerate(lines):
        j = json.loads(line)
        if i % 10000 == 0:
            print('%d out of %d' % (i, len(lines)))

        if debug and i == 10000:
            break

        if j['genre'] == 'telephone':  ## 排除文本类型为telephone的用例
            continue

        tree = nltk.tree.Tree.fromstring(j['sentence2_parse'])  ## 通过nltk库来解析得到一颗语法树

        sentences = [x for x in tree.subtrees() if x.label() == 'S']  ## 得到主句或者从句子树

        for s in sentences[:1]:  # 只取第一个句子
            convertor = SentenceConvertor(s)
            if not convertor.is_sentence_valid():
                continue

            subjobj_rev_hyp = convertor.get_subjobj_rev_hyp()

            passive_hyp_same_meaning = convertor.get_passive_hyp_same_meaning()

            passive_hyp_inverted = convertor.get_passive_hyp_inverted()

            if j['gold_label'] == 'entailment':  # 仅仅选择原先标签为entailment的样本进行主宾交换
                mnli_row(w_inv_orig, 1000000 + n,
                         j['sentence1'], subjobj_rev_hyp, 'neutral')

            mnli_row(w_inv_trsf, 1000000 + n,
                     j['sentence2'], subjobj_rev_hyp, 'neutral')

            mnli_row(w_pass_orig, 1000000 + n,
                     j['sentence1'], passive_hyp_same_meaning,
                     j['gold_label'])

            mnli_row(w_pass_trsf, 1000000 + n,
                     j['sentence2'], passive_hyp_inverted, 'neutral')
            mnli_row(w_pass_trsf, 2000000 + n,
                     j['sentence2'], passive_hyp_same_meaning, 'entailment')

            n += 1

    for file in files:
        file.close()


if __name__ == '__main__':
    main()
    ## generate_augmentation_set.main()
