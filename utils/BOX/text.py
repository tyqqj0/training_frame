# -*- CODING: UTF-8 -*-
# @time 2023/12/2 19:20
# @Author tyqqj
# @File text.py
# @
# @Aim 

def print_line(up_or_down, len=65):
    if up_or_down == 'up' or up_or_down == 0:
        print('=' * len)
        print('.' * len)
    elif up_or_down == 'down' or up_or_down == 1:
        print('.' * len)
        print('=' * len)
    else:
        print('Invalid input')
        raise ValueError


def text_in_box(text, length=65, center=True):
    # Split the text into lines that are at most `length` characters long
    lines = [text[i:i + length] for i in range(0, len(text), length)]

    # Create the box border, with a width of `length` characters
    up_border = '┏' + '━' * (length + 2) + '┓'
    down_border = '┗' + '━' * (length + 2) + '┛'
    # Create the box contents
    contents = '\n'.join(['┃ ' + (line.center(length) if center else line.ljust(length)) + ' ┃' for line in lines])

    # Combine the border and contents to create the final box
    box = '\n'.join([up_border, contents, down_border])

    return box
