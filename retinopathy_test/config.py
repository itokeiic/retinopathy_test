# -*- coding: utf-8 -*-
from os import path
#import os
import tensorflow as tf
from webargs import fields, validate, ValidationError
from marshmallow import Schema, INCLUDE
# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
# Retina_RemoteStorage = 'rshare:/deep-oc-apps/retinopathy_test'
Retina_RemoteStorage = 'rshare:/retinopathy_test'
Retina_RemotePublic = 'https://nc.deep-hybrid-datacloud.eu/s/yNsYpACAFbFS8Rp/download?path='
Retina_RemoteDataRecords = path.join(Retina_RemoteStorage, 'data', 'records')
Retina_RemoteUnsplitRecords = 'rshare:/records'
Retina_RemoteModelsUpload = path.join(Retina_RemoteStorage, 'models')
Retina_LocalDataRecords = path.join(BASE_DIR, 'data', 'records')
Retina_LocalModels = path.join(BASE_DIR, 'models')
Retina_LocalModelsServe = path.join(Retina_LocalModels, 'retinopathy_serve')
Retina_TrainingData = "retinopathy_tr.tfrecords"
Retina_ValidationData = "retinopathy_va.tfrecords"

def gpus_must_exist(val):
    if val > 0:
        if not tf.test.is_gpu_available():
            raise ValidationError("GPUs does not exist.")
        
#train_args = { 'train_epochs': {'default': 10,
                              #'help': 'Number of epochs to train on',
                              #'required': False
                             #},
               #'batch_size': {'default': 16,
                              #'help': 'Global Batch size',
                              #'required': False
                              #},
               #'num_gpus':   {'default': 1 if tf.test.is_gpu_available() else 0,
                              #'help': 'Number of GPUs to use, if available (0 = CPU)',
                              #'required': False
                             #},
               #'upload_back': {'default': False,
                               #'choices': [False, True],
                               #'help': 'Either upload a trained graph back to the remote storage (True) or not (False, default)',
                               #'required': False
                              #},
#}
train_args = { 'train_epochs': fields.Int(missing=10,
                              description='Number of epochs to train on',
                              required = False
                             ),
               'batch_size': fields.Int(missing=16,
                              description='Global Batch size',
                              required=False
                             ),
               'num_gpus':   fields.Int(missing=1,
                              validate=gpus_must_exist,
                              description='Number of GPUs to use, if available (0 = CPU)',
                              required=False
                             ),
               'upload_back': fields.Bool(missing=False,
                               enum=[False, True],
                               description='Either upload a trained graph back to the remote storage (True) or not (False, default)',
                               required=False
                              ),
}

#predict_args = {'trained_graph':   {'default': '1540408813_cpu',
                             #'choices': ['1540408813_cpu', '1533577729_gpu'],
                             #'help': 'Pre-trained graph to use',
                             #'required': False
                           #},

#}
predict_args = {'trained_graph': fields.Str(missing='1540408813_cpu',
                             enum=['1540408813_cpu', '1533577729_gpu'],
                             description='Pre-trained graph to use',
                             required=False
                           ),
                'files': fields.Field(
                            required=False,
                            missing=None,
                            type="file",
                            data_key="data",
                            location="form",
                            description="Select the image you want to classify."
                           ),
                'urls': fields.Url(
                            required=False,
                            missing=None,
                            description="Select an URL of the image you want to classify."
                           )


}
                
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # supports extra parameters

    trained_graph = fields.Str(
        required=False,
        missing='1540408813_cpu',
        enum=['1540408813_cpu', '1533577729_gpu'],
        description="Pre-trained graph to use"
    )

    files = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="data",
        location="form",
        description="Select the image you want to classify."
    )

    urls = fields.Url(
        required=False,
        missing=None,
        description="Select an URL of the image you want to classify."
    )


# class / place to describe arguments for train()
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # supports extra parameters

    train_epochs = fields.Integer(
        required=False,
        missing=10,
        description="Number of training epochs")

    batch_size = fields.Integer(
        missing=16,
        description='Global Batch size',
        required=False)

    num_gpus =  fields.Integer(
        missing=1,
        validate=gpus_must_exist,
        description='Number of GPUs to use, if available (0 = CPU)',
        required=False)
    
    upload_back = fields.Boolean(
        missing=False,
        enum=[False, True],
        description='Either upload a trained graph back to the remote storage (True) or not (False, default)',
        required=False)
