import os

def create_pos_n_neg():
    for file_type in ['pos']:
        for img in os.listdir(file_type):
            if (file_type == 'neg'):
                line = file_type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)
            elif (file_type == 'pos'):
                line = file_type + '/' + img + ' 1 0 0 500 500\n'
                with open('info.txt', 'a') as f:
                    f.write(line)

if __name__ == '__main__':
    create_pos_n_neg()
    print('正样本描述文件info.txt已生成')