import requests
from bs4 import BeautifulSoup
import pandas as pd


# start_url = 'https://www.iseecars.com/used-cars/used-tesla-for-sale'
# download_page  = requests.get(start_url)
# print(download_page)
#
#
# start_url = 'https://en.wikipedia.org/wiki/Tesla,_Inc.'
#
# downloaded_html = requests.get(start_url)
#
# soup = BeautifulSoup(downloaded_html.text, features="lxml")
#
# with open('downloaded.html', 'w') as file:
#     file.write(soup.prettify())
#
# full_table = soup.select('table.wikitable tbody')[0]
# print(full_table)
#
# table_head = full_table.select('tr th')
#
# table_columns = []
# for element in table_head:
#     column_label = element.get_text(separator=" ", strip = True)
#     table_columns.append(column_label)
#     print(column_label)
#
# print('------------')
# print(table_columns)
#
#
# table_rows = full_table.select('tr')
#
# table_data = []
# for index, element in enumerate(table_rows):
#     if index>0:
#         row_list = []
#         values = element.select('td')
#         for value in values:
#             row_list.append(value.text)
#         table_data.append(row_list)
#
# print(table_data)
#
# table_rows = full_table.select('tr')
#
# table_data = []
# for index, element in enumerate(table_rows):
#     if index>0:
#         row_list = []
#         values = element.select('td')
#         for value in values:
#             row_list.append(value.text.strip())
#         table_data.append(row_list)
#
# print(table_data)
#
# df = pd.DataFrame(table_data, columns=table_columns)
# print(df)


expr_tree = ["*", "+", "-", "a", "b", "c", "d"]
iterator = iter(expr_tree)
print(iterator)
print(next(iterator))
for item in iterator:
    print(item)

def _is_perfect_length(sequence):
    n = len(sequence)
    return ((n+1) & n==0) and (n !=0)


class LevelOrderIterator:

    def __init__(self, sequence):
        if not _is_perfect_length(sequence):
            raise ValueError(f"Sequence of length {len(sequence)} does not represent "
                             f"a perfect binary tree 2n-1 ")
        self.sequence = sequence
        self.index = 0

    def __next__(self):
        if self.index >=len(self.sequence):
            raise StopIteration
        result = self.sequence[self.index]
        self.index +=1
        return result

    def __iter__(self):
        return self



def __left_child(index):
    return 2 * index +1

def __right_child(index):
    return 2 * index +2

class PreOrderIterator:

    def __init__(self, sequence):
        if not _is_perfect_length(sequence):
            raise ValueError(f"Sequence of length {len(sequence)} does not represent "
                             f"a perfect binary tree 2n-1 ")
        self.sequence = sequence
        self.stack = [0]

    def __next__(self):
        if len(self.stack) == 0:
            raise StopIteration
        index = self.stack.pop()
        result = self.sequence[index]

        right_child_index = __right_child(index)
        if right_child_index < len(self.sequence):
            self.stack.append(right_child_index)

        left_child_index = __left_child(index)
        if left_child_index < len(self.sequence):
            self.stack.append(left_child_index)

        return result






expr_tree = ["*", "+", "-", "a", "b", "c", "d"]

iterator = LevelOrderIterator(expr_tree)
print(next(iterator))

print(" ".join(iterator))

