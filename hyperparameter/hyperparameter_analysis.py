import re
import pandas as pd


def main():
    with open('../logs/hyperparameter/erap4.txt') as f:
        text = f.read()
    reg = 'Hit ratio: (.*?), Parameters: ({.*?})'
    columns = ['hr']
    results = []
    while True:
        m = re.search(reg, text)
        if m is None:
            break
        text = text[m.end():]
        hr = float(m.group(1))
        params = eval(m.group(2))

        if len(columns) == 1:
            for key in params.keys():
                columns.append(key)

        result = [hr]
        for key in columns[1:]:
            result.append(params[key])
        results.append(result)

    df = pd.DataFrame(results, columns=columns)
    for key in columns[1:]:
        print(df.groupby(key).agg(['mean', 'max'])['hr'])


if __name__ == '__main__':
    main()