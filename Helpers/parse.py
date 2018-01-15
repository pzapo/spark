# Parser
import json


def parse(lines):
    block = []
    while lines:

        if lines[0].startswith('If'):
            bl = ' '.join(lines.pop(0).split()[1:]).replace('(', '').replace(')', '')
            block.append({'name': bl, 'children': parse(lines)})

            if lines[0].startswith('Else'):
                be = ' '.join(lines.pop(0).split()[1:]).replace('(', '').replace(')', '')
                block.append({'name': be, 'children': parse(lines)})
        elif not lines[0].startswith(('If', 'Else')):
            block2 = lines.pop(0)
            block.append({'name': block2})
        else:
            break
    return block


# Convert Tree to JSON
def tree_json(tree):
    data = []
    for line in tree.splitlines():
        if line.strip():
            line = line.strip()
            data.append(line)
        else:
            break
        if not line: break
    res = []
    res.append({'name': 'Root', 'children': parse(data[1:])})
    # print(res[0])
    with open('../data/structure.json', 'w') as outfile:
        json.dump(res[0], outfile)
    print('Conversion Success !')


