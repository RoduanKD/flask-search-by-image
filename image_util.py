import sys, os
import requests
from multiprocessing import Pool, cpu_count
from functools import partial
from io import BytesIO
import shutil


def delete_directory_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



def download_images_parallel(url, filePath):
    broken_images = []
    try:
        file_name = url.split("/")[-1]
        response = requests.get(url,stream=True)
        if response.status_code == 200:
        # This command below will allow us to write the data to a file as binary:
            with open(file_name, 'wb') as f:
                for chunk in response:
                    f.write(chunk)
                    
            # print(" Downloaded {} ".format(file_name))
        else:
        # We will write all of the images back to the broken_images list:
            broken_images.append(url)
            
               
    except Exception as e:
        print(e)
        

def download_images_parallel_starting_point(images_urls_list=[]):
    os.chdir(os.getcwd()+"/static/img")
    print("dir changed to /img")
    # filePath = os.path.dirname(os.path.abspath(__file__))
    # filePath = os.path.dirname(os.path.abspath(os.getcwd()))
    filePath = os.getcwdb()
    print("filesPath fot downloading images is %s " % filePath)
    # sys.path.append(filePath) # why do you need this?
    # urls = [
    #     'https://sempioneer.com/wp-content/uploads/2020/05/dataframe-300x84.png',
    #          'https://sempioneer.com/wp-content/uploads/2020/05/json_format_data-300x72.png',
    #          'https://homepages.cae.wisc.edu/~ece533/images/arctichare.png',
    #          'https://homepages.cae.wisc.edu/~ece533/images/baboon.png',
    #          'https://homepages.cae.wisc.edu/~ece533/images/monarch.png',
    #          'https://homepages.cae.wisc.edu/~ece533/images/serrano.png',
    #          'https://homepages.cae.wisc.edu/~ece533/images/zelda.png',
    #          'https://homepages.cae.wisc.edu/~ece533/images/tulips.png',
    #          'https://homepages.cae.wisc.edu/~ece533/images/goldhill.png',
    #          'https://homepages.cae.wisc.edu/~ece533/images/cat.png'
             
    #        ]

    print("There are {} CPUs on this machine ".format(cpu_count()))
    pool = Pool(cpu_count())
    download_func = partial(download_images_parallel, filePath = filePath)
    results = pool.map(download_func, images_urls_list)
    pool.close()
    pool.join()
    #delete photos after done downloading and training
    # delete_directory_files(filePath)