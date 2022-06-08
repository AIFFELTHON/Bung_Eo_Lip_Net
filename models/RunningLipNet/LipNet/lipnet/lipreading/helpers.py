# -*- coding: utf-8 -*-

def text_to_labels(text):
    ret = []
    for char in text:
        if char >= u'가' and char <= u'힣':
            ret.append(ord(char)-ord(u'가'))
        elif char == ' ':
            ret.append(11172)
    return ret

def labels_to_text(labels):
    # 26 is space, 가 = 44032 / 힣 = 55203
    text = ''
    for c in labels:
        if c >= 0 and c < 11172:
            c += ord(u'가')
            text += chr(c)
        elif c == 11172:
            text += ' '
    return text