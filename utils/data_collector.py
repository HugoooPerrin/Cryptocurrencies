
import pandas as pd
import pandas_datareader.data as web

import requests
import datetime


def download_crypto(freq, symbols, priceNumber):
    first = True
    main_url = 'https://min-api.cryptocompare.com/data/histo{}'.format(freq)

    for symbol in symbols:
        print('Downloading {}'.format(symbol), end='... ')
        url = main_url + '?fsym={}'.format(symbol) + '&tsym=EUR' + '&limit={}'.format(priceNumber) + '&aggregate=1'
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

        print('done')

    return data.sort_values('time', ascending=False).set_index('time')



def download_data(symbols, start, end):
    first = True

    for symbol in symbols:
        try:
            print('Downloading {}'.format(symbol), end='... ')
            inter = web.DataReader(symbol, 'morningstar', start, end)[['Close']]
            inter.columns = [symbol]
            inter['Date'] = inter.index.get_level_values('Date')
            inter.reset_index(drop=True, inplace=True)

            if first:
                price = inter.copy(True)
                del inter
                first = False
            else:
                price = pd.merge(price, inter, on='Date')
                del inter
            print('done')

        except:
            print("failed ! ")

    return price.sort_values('Date', ascending=False).set_index('Date')
