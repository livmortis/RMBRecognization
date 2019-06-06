# coding=utf8
#########################################################################
# File Name: map_word_to_index.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: Fri 18 May 2018 03:30:26 PM CST
#########################################################################
'''
此代码用于将所有文字映射到index上，有两种方式
    1. 映射每一个英文单词为一个index
    2. 映射每一个英文字母为一个index
'''

import os
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import json
from collections import OrderedDict

exist_dict = False

def map_word_to_index(train_word_file, word_index_json, word_count_json, index_label_json, alphabet_to_index=True):
    with open(train_word_file, 'r') as f:
        # csvlabels = f.read().strip().decode('utf8')
        csvlabels = f.read().strip()

      
    '''xzy统计字母出现次数'''
    word_count_dict = { }
    for line in csvlabels.split('\n')[1:]:
        line = line.strip()
        # image, sentence = line.strip().split('.png,')
        image, label = line.strip().split('.jpg,')
        # label = label.strip('"')
        label = label.strip()   #xzy
        for w in label:         #xzy w为每个字母
            word_count_dict[w] = word_count_dict.get(w,0) + 1
    print ('一共有{:d}种字符，共{:d}个'.format(len(word_count_dict), sum(word_count_dict.values())))
    
    #xzy 首次创建“标签字母--编号”字典
    if not exist_dict :
      word_sorted = sorted(word_count_dict.keys(), key=lambda k:word_count_dict[k], reverse=True)
      word_index_dict = { w:i for i,w in enumerate(word_sorted) }
      with open(word_index_json, 'w') as f:
          f.write(json.dumps(word_index_dict, indent=4, ensure_ascii=False))
    #xzy 读取已有的“标签字母--编号”字典
    else:
      word_index_dict = json.load(open(word_index_json))



    with open(word_count_json, 'w') as f:   #word_count_json  统计每个字母出现的次数
        f.write(json.dumps(word_count_dict, indent=4, ensure_ascii=False))

        
        
        
    '''xzy统计'''    
    image_label_dict = OrderedDict()
    for line in csvlabels.split('\n')[1:]:
        line = line.strip()
        # image, sentence = line.strip().split('.png,')
        image, rmblabel = line.strip().split('.jpg,')
        # rmblabel = rmblabel.strip('"')
        rmblabel = rmblabel.strip()#xzy

        # 换掉部分相似符号      xzy废弃
        # for c in u"　 ":
        #     rmblabel = rmblabel.replace(c, '')
        # replace_words = [
        #         u'(（',
        #         u')）',
        #         u',，',
        #         u"´'′", 
        #         u"″＂“",
        #         u"．.",
        #         u"—-"
        #         ]
        # for words in replace_words:
        #     for w in words[:-1]:
        #         rmblabel = rmblabel.replace(w, words[-1])

        index_list = []
        for w in rmblabel:
            index_list.append(str(word_index_dict[w]))
        # image_label_dict[image + '.png'] = ' '.join(index_list)
        image_label_dict[image + '.jpg'] = ' '.join(index_list)
    with open(index_label_json, 'w') as f:
        f.write(json.dumps(image_label_dict, indent=4))


def main():

    # 映射字母为index
    # train_word_file = '../../files/train.csv'   #已有的“1个图片----N个字母”原始标签
    train_word_file = '../../../../../../dataset_formal/classify_data/train_id_label.csv'   #已有的“1个图片----N个字母”原始标签
    # word_index_json = '../../files/alphabet_index_dict.json'    #要创建的/已有的 “1个字母---1个索引” 的字典
    word_index_json = '../../RMBfiles/RMB_index_dict.json'    #要创建的/已有的 “1个字母---1个索引” 的字典
    # word_count_json = '../../files/alphabet_count_dict.json'    #要创建的字数统计字典
    word_count_json = '../../RMBfiles/RMB_count_dict.json'    #要创建的字数统计字典
    # index_label_json = '../../files/train_alphabet.json'    #要创建的 “1个图片----N个字母标签的索引” 的新标签。
    index_label_json = '../../RMBfiles/RMB_train_index_label.json'    #要创建的 “1个图片----N个字母标签的索引” 的新标签。
    map_word_to_index(train_word_file, word_index_json, word_count_json, index_label_json, True)

if __name__ == '__main__':
    main()
