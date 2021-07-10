import os

def create_pos_n_neg():
    for file_type in ['neg']: #此处修改pos或neg即可生成正负样本的描述文件，neg是生成正样本描述文件bg.txt
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
    print('负样本描述文件bg.txt已生成')