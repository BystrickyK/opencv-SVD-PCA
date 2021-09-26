import pandas as pd
import requests
import shutil

df = pd.read_csv('FEC_dataset/faceexp-comparison-data-train-public.csv',
                 error_bad_lines=False)

datapath = 'datapath/'
for i, row in df.loc[:50000].iterrows():
    print("#{}".format(i))
    link = row.iloc[0]
    r = requests.get(link, stream=True)

    if r.status_code == 200:
        filename = link.split("/")[-1]
        r.raw.decode_content = True
        with open(datapath + filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
            print('.')
        print("\tRequest successful: {}".format(r.status_code))
    else:
        print("\tRequest unsuccessful: {}".format(r.status_code))
