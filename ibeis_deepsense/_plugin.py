from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject
from ibeis_deepsense import docker_control  # NOQA
import utool as ut
import dtool as dt
from PIL import Image
from io import BytesIO
import base64
import requests
from ibeis.constants import ANNOTATION_TABLE
from ibeis.web.apis_engine import ensure_uuid_list

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)
register_preproc_annot = controller_inject.register_preprocs['annot']

u"""
Interfacing with the ACR from python is a headache, so for now we will assume that
the docker image has already been downloaded. Command:
docker pull wildme.azurecr.io/ibeis/deepsense
"""


BACKEND_URL = None


def _ibeis_plugin_deepsense_check_container(url):
    ut.embed()


docker_control.docker_register_config(None, 'deepsense', 'wildme.azurecr.io/ibeis/deepsense:latest', run_args={'_internal_port': 5000, '_external_suggested_port': 5000}, container_check_func=_ibeis_plugin_deepsense_check_container)


@register_ibs_method
def ibeis_plugin_deepsense_ensure_backend(ibs):
    global BACKEND_URL
    # make sure that the container is online using docker_control functions
    if BACKEND_URL is None:
        BACKEND_URL = ibs.docker_ensure('deepsense')
    return BACKEND_URL


@register_ibs_method
@register_api('/api/plugin/deepsense/identify/', methods=['GET'])
def ibeis_plugin_deepsense_identify(ibs, annot_uuid, use_depc=True):
    r"""
    Run the Kaggle winning Right-whale deepsense.ai ID algorithm

    Args:
        ibs         (IBEISController): IBEIS controller object
        annot_uuid  (uuid): Annotation for ID

    CommandLine:
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_identify
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_identify:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis_deepsense
        >>> from ibeis_deepsense._plugin import _ibeis_plugin_deepsense_rank  # NOQA
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> from os.path import abspath, exists, join, dirname, split, splitext
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> local_path = dirname(abspath(ibeis_deepsense.__file__))
        >>> image_path = abspath(join(local_path, '..', 'example-images'))
        >>> assert exists(image_path)
        >>> gid_list = ibs.import_folder(image_path, ensure_loadable=False, ensure_exif=False)
        >>> uri_list = ibs.get_image_uris_original(gid_list)
        >>> image_id_list = [int(splitext(split(uri)[1])[0]) for uri in uri_list]
        >>> aid_list = ibs.use_images_as_annotations(gid_list)
        >>> uuid_list = ibs.get_annot_uuids(aid_list)
        >>> rank_list = []
        >>> for image_id, annot_uuid in zip(image_id_list, uuid_list):
        >>>     resp_json = ibs.ibeis_plugin_deepsense_identify(annot_uuid, use_depc=False)
        >>>     rank, score = _ibeis_plugin_deepsense_rank(resp_json, image_id)
        >>>     print('[instant] for whale id = %s, got rank %d with score %0.04f' % (image_id, rank, score, ))
        >>>     rank_list.append(rank)
        >>> response_list = ibs.depc_annot.get('DeepsenseIdentification', aid_list, 'response')
        >>> rank_list_cache = []
        >>> for image_id, resp_json in zip(image_id_list, response_list):
        >>>     rank, score = _ibeis_plugin_deepsense_rank(resp_json, image_id)
        >>>     print('[cache] for whale id = %s, got rank %d with score %0.04f' % (image_id, rank, score, ))
        >>>     rank_list_cache.append(rank)
        >>> assert rank_list == rank_list_cache
        >>> result = rank_list
        [-1, 0, 0, 3, -1]
    """
    annot_uuid_list = [annot_uuid]
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)
    annot_uuid_list = ensure_uuid_list(annot_uuid_list)
    # Ensure annotations
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    aid = aid_list[0]

    if use_depc:
        response_list = ibs.depc_annot.get('DeepsenseIdentification', [aid], 'response')
        response = response_list[0]
    else:
        response = ibs.ibeis_plugin_deepsense_identify_aid(aid)
    return response


@register_ibs_method
def ibeis_plugin_deepsense_identify_aid(ibs, aid):
    url = ibs.ibeis_plugin_deepsense_ensure_backend()

    image_path = ibs.get_annot_chip_fpath(aid)
    pil_image = Image.open(image_path)
    byte_buffer = BytesIO()
    pil_image.save(byte_buffer, format="JPEG")
    b64_image = base64.b64encode(byte_buffer.getvalue()).decode("utf-8")

    data = {
        'image': b64_image,
        'configuration': {
            'top_n': 100,
            'threshold': 0.0,
        }
    }
    response = requests.post('http://%s/api/classify' % (url), json=data)
    assert response.status_code == 200
    return response.json()



class DeepsenseIdentificationConfig(dt.Config):  # NOQA
    _param_info_list = []


@register_preproc_annot(
    tablename='DeepsenseIdentification', parents=[ANNOTATION_TABLE],
    colnames=['response'], coltypes=[dict],
    configclass=DeepsenseIdentificationConfig,
    fname='deepsense',
    chunksize=4)
def ibeis_plugin_deepsense_identify_depc(depc, aid_list, config):
    # The doctest for ibeis_plugin_deepsense_identify also covers this func
    ibs = depc.controller
    for aid in aid_list:
        response = ibs.ibeis_plugin_deepsense_identify_aid(aid)
        yield (response, )


def _ibeis_plugin_deepsense_rank(response_json, whale_id):
    ids = response_json['identification']
    for index, result in enumerate(ids):
        if result['whale_id'] == whale_id:
            return (index, result['probability'])
    return (-1, -1)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_deepsense._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
