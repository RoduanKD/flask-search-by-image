import os


#root
STATIC_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+'/static'

#inference_queue
INFERENCE_QUEUE_DIR = STATIC_ROOT_DIR+'/inference_queue'

#Uploaded
UPLOADED  = STATIC_ROOT_DIR+'/uploaded'

# trained 
TRAINED = STATIC_ROOT_DIR+'/trained'

#featured
FEATURED = STATIC_ROOT_DIR+'/feature'

#cached images
CACHE_DIR = STATIC_ROOT_DIR+'/cache'