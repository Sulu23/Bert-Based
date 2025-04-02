def make_tsv(file):
    with open('data/tsv_train.conll', 'x') as outfile:
        for line in file:
            if line == '\n' or line.startswith('# sent_enum ='):
                pass
            else:
                outfile.writelines(line)


def main():
    filepath = 'data/train.conll'

    file = open(filepath, 'r')

    make_tsv(file)

    file.close()


if __name__ == '__main__':
    main()
