#!/usr/bin/env python
# -*- coding:utf-8 -*-
import openpyxl
import os
import numpy as np


def create_class_list(file_dir, file_type, number=None):
    """

    """
    class_name_set = set()  # 创建空集合必须用set()

    if file_type is 'xlsx':
        for cur_folder, sub_folders, sub_files in os.walk(file_dir):
            for file in sub_files:
                file_path = os.path.join(cur_folder, file)
                wb = openpyxl.load_workbook(file_path)
                sheet = wb.worksheets[0]
                class_list = []
                for i in range(1, number+1):
                    class_name = sheet['A'+str(i)]
                    class_list.append(class_name.value)
                class_name_set.update(class_list)

    if file_type is 'txt':
        for cur_folder, sub_folders, sub_files in os.walk(file_dir):
            for file in sub_files:
                if file.endswith('txt'):
                    txt_path = os.path.join(cur_folder, file)
                    f = open(txt_path)
                    class_list = f.readlines()
                    # 删除标注中的换行符 制表符 空格和背景类
                    for label in class_list[:]:  # 使用class_list[:] 复制一份进行循环，避免删除对象和循环对象是同一个
                        if label.startswith('pack'):
                            class_list.remove(label)  # 使用remove删除指定元素，
                            continue
                        if label.endswith('\n'):
                            i = class_list.index(label)
                            label = label.strip('\n')
                            if '\t' in label:
                                label = label.split('\t')[0]
                            label = label.replace(' ', '')
                            class_list[i] = label
                    f.close()
                    class_name_set.update(class_list)  # 更新集合

    return class_name_set


def create_label_map(class_list, save_path):
    """
    create a .pdtxt file from [label_list]
    :param file_path: a file name of .pdtxt file
    :param label_list:
    :return:
    """
    base_str = "item {}\n  id: {}\n  name: '{}'\n{}\n"
    string = []
    for idx, label in enumerate(class_list):
        string.append(base_str.format('{', idx + 1, label, '}'))
    write_text_file(save_path, string)


def write_text_file(file_name, *file_lists, split=' ', mode='w', encoding='utf-8'):
    """
    将多个字符串列表写入文本文件中
    """
    check_consistent_length(*file_lists)
    with open(file_name, mode, encoding=encoding) as file:
        for strings in zip(*file_lists):
            string = split.join(strings)  # '-'.join(['1','2','3']) -->'1-2-3'
            file.write(string+'\n')


def check_consistent_length(*arrays):
    """
    检查所有数组的第一维的长度是否相等

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """
    lengths = [num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of samples: %r" % [int(l) for l in lengths])


def num_samples(x):
    """
    返回类似数组的数据数量
    Parameters
    ----------
    x: array-like
    Returns
    -------
    """
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


if __name__ == '__main__':
    class_set = create_class_list('D:\\Picture\\Nestle\\Nestle_original', 'txt')
    class_list = sorted(list(class_set))
    create_label_map(class_list, 'label_map.pbtxt')
