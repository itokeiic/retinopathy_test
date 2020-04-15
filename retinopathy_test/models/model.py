# -*- coding: utf-8 -*-
"""
Model description
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import pkg_resources
# import project config.py
import retinopathy_test.config as cfg
import retinopathy_test.models.run_prediction as runpred #ki: comment out to avoid tensorflow import
import os
import shutil
import tempfile
import zipfile
import retinopathy_test.models.retinopathy_main as retimain
# import retinopathy_test.models.models-master.official as official
import absl
from absl import flags
from absl import app as absl_app
from official.utils.logs import logger

import tensorflow as tf
#from official.utils.flags import core as flags_core
import deepaas
from pkg_resources import parse_version
import subprocess
import time
from webargs import fields, validate, ValidationError
from aiohttp.web import HTTPBadRequest
import json
import mimetypes 
## Authorization
from flaat import Flaat
flaat = Flaat()

# Switch for debugging in this script
debug_model = True 

def _catch_error(f):
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap

def _fields_to_dict(fields_in):
    """
    Function to convert mashmallow fields to dict()
    """
    dict_out = {}

    for key, val in fields_in.items():
        param = {}
        param['default'] = val.missing
        param['type'] = type(val.missing)
        if key == 'files' or key == 'urls':
            param['type'] = str

        val_help = val.metadata['description']
        if 'enum' in val.metadata.keys():
            val_help = "{}. Choices: {}".format(val_help, 
                                                val.metadata['enum'])
        param['help'] = val_help

        try:
            val_req = val.required
        except:
            val_req = False
        param['required'] = val_req

        dict_out[key] = param
    return dict_out

class EmptyObj():
    pass

def rclone_copy(src_path, dest_path, cmd='copy',):
    '''
    Wrapper around rclone to copy files
    :param src_path: path of what to copy. in the case of "copyurl" path at the remote
    :param dest_path: path where to copy
    :param cmd: how to copy, "copy" or "copyurl"
    :return: output message and a possible error
    '''

    if cmd == 'copy':
        command = (['rclone', 'copy', '--progress', src_path, dest_path])
    elif cmd == 'copyurl':
        src_path = '/' + src_path.lstrip('/')
        src_dir, src_file = os.path.split(src_path)
        remote_link = cfg.Retina_RemotePublic + src_dir + '&files=' + src_file
        print("[INFO] Trying to download {} from {}".format(src_file,
                                                            remote_link))
        command = (['rclone', 'copyurl', remote_link, dest_path])
    else:
        message = "[ERROR] Wrong 'cmd' value! Allowed 'copy', 'copyurl', received: " + cmd
        raise Exception(message)

    try:
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()
    except OSError as e:
        output, error = None, e
    return output, error
    
#def get_metadata():

    #module = __name__.split('.', 1)

    #pkg = pkg_resources.get_distribution(module[0])
    #meta = {
        #'Name': None,
        #'Version': None,
        #'Summary': None,
        #'Home-page': None,
        #'Author': None,
        #'Author-email': None,
        #'License': None,
    #}

    #for line in pkg.get_metadata_lines("PKG-INFO"):
        #for par in meta:
            #if line.startswith(par+":"):
                #_, value = line.split(": ", 1)
                #meta[par] = value

    #return meta
def get_metadata():
    """
    Function to read metadata
    """

    module = __name__.split('.', 1)

    try:
        pkg = pkg_resources.get_distribution(module[0])
    except pkg_resources.RequirementParseError:
        # if called from CLI, try to get pkg from the path
        distros = list(pkg_resources.find_distributions(cfg.BASE_DIR, 
                                                        only=True))
        if len(distros) == 1:
            pkg = distros[0]
    except Exception as e:
        raise HTTPBadRequest(reason=e)

    # deserialize key-word arguments
    train_args = _fields_to_dict(get_train_args())
    # make 'type' JSON serializable
    for key, val in train_args.items():
        train_args[key]['type'] = str(val['type'])

    predict_args = _fields_to_dict(get_predict_args())
    # make 'type' JSON serializable
    for key, val in predict_args.items():
        predict_args[key]['type'] = str(val['type'])

    meta = {
        'name' : None,
        'version' : None,
        'summary' : None,
        'home-page' : None,
        'author' : None,
        'author-email' : None,
        'license' : None,
        'help-train' : train_args,
        'help-predict' : predict_args
    }
    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower() # to avoid inconsistency due to letter cases
        for par in meta:
            if line_low.startswith(par.lower() + ":"):
                _, value = line.split(": ", 1)
                meta[par] = value
                
    return meta


        
#def predict(**args):

    #if (not any([args['urls'], args['files']]) or
            #all([args['urls'], args['files']])):
        #raise Exception("You must provide either 'url' or 'data' in the payload")

    #if args['files']:
        #args['files'] = [args['files']]  # patch until list is available
        #return predict_data(args)
    #elif args['urls']:
        #args['urls'] = [args['urls']]  # patch until list is available
        #return predict_url(args)
        
def predict(**kwargs):

    print("predict(**kwargs) - kwargs: %s" % (kwargs)) if debug_model else ''

    if (not any([kwargs['urls'], kwargs['files']]) or
            all([kwargs['urls'], kwargs['files']])):
        raise Exception("You must provide either 'url' or 'data' in the payload")

    if kwargs['files']:
        kwargs['files'] = [kwargs['files']]  # patch until list is available
        return predict_data(kwargs)
    elif kwargs['urls']:
        kwargs['urls'] = [kwargs['urls']]  # patch until list is available
        return predict_url(kwargs)

    
def predict_file(img_path, trained_graph):
    """
    Function to make prediction on a local file
    """
    print ('[DEBUG] image_path: ', img_path)
    model_dir = os.path.join(cfg.Retina_LocalModelsServe, trained_graph)
    print ('[DEBUG] model_dir: ', model_dir)
    
    trained_graph_file = trained_graph + ".zip"
    store_zip_path = os.path.join(cfg.Retina_LocalModelsServe, trained_graph_file)

    if not os.path.exists(model_dir):
        remote_src_path = os.path.join('models', trained_graph_file)
        print("[INFO] Graph {} is not found.".format(trained_graph))
        output, error = rclone_copy(src_path=remote_src_path,
                                    dest_path=store_zip_path,
                                    cmd='copyurl')
        if error:
            message = "[ERROR] graph was not properly copied. rclone returned: "
            message = message + error
            raise Exception(message)

        # if .zip is present locally, de-archive it
        if os.path.exists(store_zip_path):
            print("[INFO] {}.zip was downloaded. Unzipping...".format(trained_graph))
            data_zip = zipfile.ZipFile(store_zip_path, 'r')
            data_zip.extractall(cfg.Retina_LocalModelsServe)
            data_zip.close()
            # remove downloaded zip-file
            if os.path.exists(model_dir):
                os.remove(store_zip_path)

    results=runpred.predict_image(model_dir, img_path)
    print ('[DEBUG] results: %s'%results)
    return results


def predict_data(*args):
    """
    Function to make prediction on an uploaded file
    """
    #deepaas_ver_cut = parse_version('0.5.0')
    #imgs = []
    #filenames = []
    
    #deepaas_ver = parse_version(deepaas.__version__)
    #print("[INFO] deepaas_version: %s" % deepaas_ver)
    #predict_debug = False
    #if predict_debug:
        #print('[DEBUG] predict_data - args: %s' % args)
        #print('[DEBUG] predict_data - kwargs: %s' % kwargs)
    #if deepaas_ver >= deepaas_ver_cut:
        #for arg in args:
            #imgs.append(arg['files'])
            #trained_graph = str(yaml.safe_load(arg.trained_graph))
    #else:
        #imgs = args[0]

    #if not isinstance(imgs, list):
        #imgs = [imgs]
            
    #for image in imgs:
        #if deepaas_ver >= deepaas_ver_cut:
            #f = open("/tmp/%s" % image[0].filename, "w+")
            #image[0].save(f.name)
        #else:
            #f = tempfile.NamedTemporaryFile(delete=False)
            #f.write(image)
        #f.close()
        #filenames.append(f.name)
        #print("Stored tmp file at: {} \t Size: {}".format(f.name,
        #os.path.getsize(f.name)))
        
    print("predict_data(*args) - args: %s" % (args)) if debug_model else ''

    files = []

    for arg in args:
        file_objs = arg['files']
        for f in file_objs:
            files.append(f.filename)
            if debug_model:
                print("file_obj: name: {}, filename: {}, content_type: {}".format(
                                                               f.name,
                                                               f.filename,
                                                               f.content_type))
                print("File for prediction is at: {} \t Size: {}".format(
                                                  f.filename,
                                                  os.path.getsize(f.filename)))
        trained_graph = arg['trained_graph']

    prediction = []
    try:
        #for imgfile in filenames:
        for imgfile in files:    
            prediction.append(str(predict_file(imgfile, trained_graph)))
            #prediction.append(predict_file(imgfile, trained_graph))
            print("image: ", imgfile)
    except Exception as e:
        raise e
    finally:
        #for imgfile in filenames:
        for imgfile in files:
            os.remove(imgfile)

    return prediction

    #print (img_path)
    #model_dir = os.path.join(cfg.BASE_DIR, 'models','retinopathy_serve')
    #model_dir = os.path.join('.','retinopathy_serve')
    #model_dir+='/'
    
    #print(model_dir)
    #run_prediction.predict_image(model_dir,img_path)
    
    #message = 'Not implemented in the model (predict_data hello!)'
    #return runpred.predict_image(model_dir,img_path)


def predict_url(*args):
    """
    Function to make prediction on a URL
    """    
    message = 'Not implemented in the model (predict_url)'
    message = {"Error": message}
    return message

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)

@flaat.login_required() # Require only authorized people to do training
#def train(train_args):
def train(**kwargs):
    """
    Train network (transfer learning)
    Parameters
    ----------
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.train
    """
    print("train(**kwargs) - kwargs: %s" % (kwargs)) if debug_model else ''
    run_results = { "status": "ok",
                    "sys_info": [],
                    "training": [],
                  }


    # Check if necessary local directories exist:
    if not os.path.exists(cfg.Retina_LocalDataRecords):
        print("[INFO] %s is not found locally, creating..." % 
              cfg.Retina_LocalDataRecords)
        os.makedirs(cfg.Retina_LocalDataRecords)
    if not os.path.exists(cfg.Retina_LocalModelsServe):
        print("[INFO] %s is not found locally, creating..." % 
              cfg.Retina_LocalModelsServe)
        os.makedirs(cfg.Retina_LocalModelsServe)  

    # use the schema
    schema = cfg.TrainArgsSchema()
    # deserialize key-word arguments
    train_args = schema.load(kwargs)

    # Take parameters defined via deepaas by a user
    #train_epochs = yaml.safe_load(train_args.train_epochs)
    #batch_size = yaml.safe_load(train_args.batch_size)
    #num_gpus = yaml.safe_load(train_args.num_gpus)
    train_epochs = train_args['train_epochs']
    batch_size = train_args['batch_size']
    num_gpus = train_args['num_gpus']
    upload_back = train_args['upload_back']
    if debug_model:
        print("train_args:", train_args)
        print(type(train_args['train_epochs']), type(train_args['batch_size']))
        print("Number of GPUs:", train_args['num_gpus'], num_gpus)

    # from deep-nextcloud into the container
    # data_origin = 'rshare:/records_short/'
    training_data = os.path.join(cfg.Retina_LocalDataRecords, 
                                 cfg.Retina_TrainingData)
    validation_data = os.path.join(cfg.Retina_LocalDataRecords, 
                                   cfg.Retina_ValidationData)
    e1=time.time()
    # check if retinopathy_tr.tfrecord or retinopathy_va.tfrecord exist locally,
    # if not -> download them from the RemoteStorage
    #if not (os.path.exists(training_data) or os.path.exists(validation_data)):
        ## Retina_RemoteDataRecords and Retina_LocalDataRecords are defined in config.py #vk
        #print("[INFO] Either %s or %s NOT found locally, download them from %s" % 
              #(training_data, validation_data, cfg.Retina_RemoteDataRecords))
        #output, error = rclone_copy(cfg.Retina_RemoteDataRecords, cfg.Retina_LocalDataRecords)
        #if error:
            #message = "[ERROR] training data not copied. rclone returned: " + error
            #raise Exception(message)
            
        
    download_time=time.time()-e1
    time.sleep(60)
 
    #FLAGS=flags.FLAGS
    #for name in list(FLAGS):
    #    print (name)

    #flags.DEFINE_string('listen-ip', '0.0.0.0', 'port')
    #flags.FLAGS.unparse_flags()
    #for name in list(FLAGS):
    #    print (name)
    #    #if name == 'listen-ip':
    #        #delattr(flags.FLAGS, name)

    e2=time.time()
    ### mimic retinopathy_main.py main()
    # we first delete all the FLAGS
    FLAGS = flags.FLAGS
    #FLAGS.unparse_flags()
    for name in list(FLAGS):
        delattr(FLAGS, name)

    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # define default FLAGS for retinopathy_main and _run_loop
    retimain.define_retinopathy_flags(batch_size=str(batch_size),
                                      train_epochs=str(train_epochs),
                                      num_gpus=str(num_gpus))

    # build list of FLAG names and parse them via FLAGS(list)(IMPORTANT!) #vk
    flag_names = []
    for name in FLAGS:
        flag_names.append(name)

    # According to the docs, actual parsing happens by either calling
    # FLAGS(list_of_arguments) or by app.run()
    FLAGS(flag_names)
    # call actual training with the set flags
    with logger.benchmark_context(flags.FLAGS):
        graph_zip_path = retimain.run_retinopathy(flags.FLAGS)
        
    ### depricated: direct call of the script via subprocess
    #training_script=os.path.join(cfg.BASE_DIR,
    #                          'retinopathy_test',
    #                          'models',
    #                          'retinopathy_main.py')
    ## alternative (??) retinopathy_main_short.py
    #print(training_script)
    #code = subprocess.call(["python", training_script,
    #                                  '--batch_size', str(batch_size), 
    #                                  '--train_epochs', str(train_epochs)])
    #    
    ## check_output allows to catch output/printed informaion
    #graph_zip_path = subprocess.check_output(["python", training_script,
    #                                        '--batch_size', str(batch_size),
    #                                        '--train_epochs', str(train_epochs)])
    ###

    try:
        graph_zip_path = graph_zip_path.decode()
    except (UnicodeDecodeError, AttributeError):
        pass
    graph_zip_path = graph_zip_path.rstrip()

    print("[INFO] Call of the training script returned: ", graph_zip_path)
    training_time=time.time()-e2
    time.sleep(60)

    e3=time.time()
    # Retina_LocalModelsServe and Retina_RemoteModelsUpload are defined in config.py #vk
    # upload_back = yaml.safe_load(train_args.upload_back)
    if(upload_back and os.path.exists(graph_zip_path)):
        graph_zip_dir, graph_zip_name = os.path.split(graph_zip_path)
        print("[INFO] Uploading {} to {} ...".format(graph_zip_name, 
                                                 cfg.Retina_RemoteModelsUpload))
        output, error = rclone_copy(graph_zip_path,
                                    os.path.join(cfg.Retina_RemoteModelsUpload, 
                                                 graph_zip_name))
        if error:
            print("[ERROR] rclone returned: {}".format(error))
        else:
            # if there is no error, remove zip file and the graph directory
            savedmodel_dir, _ = os.path.splitext(graph_zip_name) # split name, ext
            savedmodel_path = os.path.join(graph_zip_dir, savedmodel_dir)
            ## Try to remove tree, if it exists
            print("[INFO] Uploaded, deleting local {} and {}...".format(graph_zip_path,
                                                                     savedmodel_path))
            os.remove(graph_zip_path)          # remove zipped file
            if os.path.exists(savedmodel_path):
                shutil.rmtree(savedmodel_path) # remove corresponding directory
            else:
                print("[INFO] Saved model path, {}, doesn't exitst!".format(
                                                              savedmodel_path)) 
    else:
        print("[INFO] Created zip file of the graph, %s, was NOT uploaded!" % graph_zip_path)

    upload_time=time.time()-e3
    training_data_path = os.path.join(cfg.Retina_LocalDataRecords,
                                      'retinopathy_tr.tfrecords')
    #message = 'Not implemented in the model (train)'
    message = 'Training finished! download time: %f, training time: %f, upload time: %f, training file size: %s'%(download_time,training_time,upload_time,file_size(training_data_path))
    return message

def get_train_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    https://marshmallow.readthedocs.io/en/latest/api_reference.html#module-marshmallow.fields
    :param kwargs:
    :return:
    """

    #train_args = cfg.train_args

    ## convert default values and possible 'choices' into strings
    ##for key, val in train_args.items():
        ##val['default'] = str(val['default']) #yaml.safe_dump(val['default']) #json.dumps(val['default'])
        ##if 'choices' in val:
            ##val['choices'] = [str(item) for item in val['choices']]

    #return train_args
    return cfg.TrainArgsSchema().fields

# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
#def get_test_args():
def get_predict_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    
    #predict_args = cfg.predict_args

    ## convert default values and possible 'choices' into strings
    ##for key, val in predict_args.items():
        ##val['default'] = str(val['default'])  # yaml.safe_dump(val['default']) #json.dumps(val['default'])
        ##if 'choices' in val:
            ##val['choices'] = [str(item) for item in val['choices']]
        ##print(val['default'], type(val['default']))

    #return predict_args
    return cfg.PredictArgsSchema().fields

