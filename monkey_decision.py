import random

cats = [0, 1, 2, 3, 4, 5]

with open('result/foooooooo.csv', 'w') as w:
    w.write('Id,Category\n')
    for i in range(1, 1001):
        cat = random.choice(cats)
        w.write('{},{}\n'.format(i, cat))
