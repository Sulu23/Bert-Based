
def make_tsv(file1, file2):
    """Transforms data files into true tsv files"""

    with open('data/tsv_train.conll', 'x') as outfile1:
        for line in file1:
            if line == '\n' or line.startswith('# sent_enum ='):
                pass
            else:
                outfile1.writelines(line)

    with open('data/tsv_dev.conll', 'x') as outfile2:
        for line in file2:
            if line == '\n' or line.startswith('# sent_enum = '):
                pass
            else:
                outfile2.writelines(line)


def main():
    filepath1 = 'data/train.conll'
    filepath2 = 'data/dev.conll'

    file1 = open(filepath1, 'r')
    file2 = open(filepath2, 'r')

    make_tsv(file1, file2)

    file1.close()
    file2.close()


if __name__ == '__main__':
    main()
