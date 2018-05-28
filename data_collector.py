import pandas as pd

import requests
import datetime


def collector(freq, symbols):

    main_url = 'https://min-api.cryptocompare.com/data/histo{}'.format(freq)
    first = True

    for symbol in symbols:
        url = main_url + '?fsym={}'.format(symbol) + '&tsym=EUR' + '&limit=2000' + '&aggregate=1'
        response = requests.get(url)
        inter = response.json()['Data']
        inter = pd.DataFrame(inter)[['close', 'time']]
        inter.columns = [symbol, 'time']
        inter['time'] = inter['time'].map(datetime.datetime.fromtimestamp)

        if first:
            data = inter
            first = False
        else:
            data = pd.merge(data, inter, on='time')

    return data.set_index('time')