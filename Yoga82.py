# To download images from Yoga 82 dataset - https://arxiv.org/pdf/2004.10362v1.pdf
import urllib.request, urllib.error
import urllib.parse
import requests

#Input .txt file consisting of URLs
input_txt = "SomeRandomFile.txt"
filepath = "C:/Users/U S Matharu/Desktop/yoga_dataset_links/"+input_txt

# Headers to pass HTTP Errors like 403
hdr = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}


with open(filepath) as fp:
    while True:
        line = fp.readline().split('\t')
        if line == ['']:
            break
        else:
            url = line[1]
            req = urllib.request.Request(url, headers=hdr)
            try:
                file_name = r'E:/Images/'+line[0]
                response = urllib.request.urlretrieve(url,file_name)
                print("Downloading",file_name)
            except urllib.error.HTTPError as e:
                # Trying to pass secured connection
                try:
                    if 'https' in file_name:
                            file_name = file_name.replace('https','http')
                            response = urllib.request.urlretrieve(url,file_name)
                            print("Downloading",file_name)

                except urllib.error.HTTPError as e:
                    try:
                        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
                            data = response.read() # a `bytes` object
                            out_file.write(data)
                            print("Downloading",file_name)
                    except urllib.error.HTTPError as e:
                        print('HTTPError: {}'.format(e.code))
            except urllib.error.URLError as e:
                print('URLError: {}'.format(e.reason))
