from __future__ import absolute_import, division, print_function
from os.path import abspath, exists, join, dirname, split, splitext
import ibeis
from ibeis.control import controller_inject, docker_control
from ibeis.constants import ANNOTATION_TABLE
from ibeis.web.apis_engine import ensure_uuid_list
import ibeis.constants as const
import utool as ut
import dtool as dt
import vtool as vt
import numpy as np
import base64
import requests
from PIL import Image, ImageDraw
from io import BytesIO


(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)
register_preproc_annot = controller_inject.register_preprocs['annot']

u"""
Interfacing with the ACR from python is a headache, so for now we will assume that
the docker image has already been downloaded. Command:

docker pull wildme.azurecr.io/ibeis/deepsense:latest

"""


DIM_SIZE = 2000
BACKEND_URL = None


INDIVIDUAL_MAP_FPATH = 'https://cthulhu.dyn.wildme.io/public/random/deepsense.flukebook.v0.csv'
ID_MAP = None


def _ibeis_plugin_deepsense_check_container(url):
    endpoints = {
        'api/alignment' : ['POST'],
        'api/keypoints' : ['POST'],
        'api/classify'  : ['POST'],
    }
    flag_list = []
    endpoint_list = list(endpoints.keys())
    for endpoint in endpoint_list:
        print('Checking endpoint %r against url %r' % (endpoint, url, ))
        flag = False
        required_methods = set(endpoints[endpoint])
        supported_methods = None
        url_ = 'http://%s/%s' % (url, endpoint, )

        try:
            response = requests.options(url_, timeout=1)
        except:
            response = None

        if response is not None and response.status_code:
            headers = response.headers
            allow = headers.get('Allow', '')
            supported_methods_ = [method.strip().upper() for method in allow.split(',')]
            supported_methods = set(supported_methods_)
            if len(required_methods - supported_methods) == 0:
                flag = True
        if not flag:
            args = (endpoint, )
            print('[ibeis_deepsense - FAILED CONTAINER ENSURE CHECK] Endpoint %r failed the check' % args)
            print('\tRequired Methods:  %r' % (required_methods, ))
            print('\tSupported Methods: %r' % (supported_methods, ))
        print('\tFlag: %r' % (flag, ))
        flag_list.append(flag)
    supported = np.all(flag_list)
    return supported


docker_control.docker_register_config(None, 'deepsense', 'wildme.azurecr.io/ibeis/deepsense:latest', run_args={'_internal_port': 5000, '_external_suggested_port': 5000}, container_check_func=_ibeis_plugin_deepsense_check_container)
# next two lines for comparing containers side-by-side
docker_control.docker_register_config(None, 'deepsense2', 'wildme.azurecr.io/ibeis/deepsense:app2', run_args={'_internal_port': 5000, '_external_suggested_port': 5000}, container_check_func=_ibeis_plugin_deepsense_check_container)
docker_control.docker_register_config(None, 'deepsense5', 'wildme.azurecr.io/ibeis/deepsense:app5', run_args={'_internal_port': 5000, '_external_suggested_port': 5000}, container_check_func=_ibeis_plugin_deepsense_check_container)


@register_ibs_method
def _ibeis_plugin_deepsense_init_testdb(ibs):
    local_path = dirname(abspath(__file__))
    image_path = abspath(join(local_path, '..', 'example-images'))
    assert exists(image_path)
    gid_list = ibs.import_folder(image_path, ensure_loadable=False, ensure_exif=False)
    uri_list = ibs.get_image_uris_original(gid_list)
    annot_name_list = [splitext(split(uri)[1])[0] for uri in uri_list]
    aid_list = ibs.use_images_as_annotations(gid_list)
    ibs.set_annot_names(aid_list, annot_name_list)
    return gid_list, aid_list


@register_ibs_method
def _ibeis_plugin_deepsense_rank(ibs, response_json, desired_name):
    ids = response_json['identification']
    for index, result in enumerate(ids):
        whale_id = result['whale_id']
        flukebook_id = result.get('flukebook_id', whale_id)
        probability = result['probability']
        name = str(flukebook_id)
        if name == desired_name:
            return (index, probability)
    return (-1, -1)


# This method converts from the ibeis/Flukebook individual UUIDs to the Deepsense/
# NEAQ IDs used by the deepsense container.
@register_ibs_method
def ibeis_plugin_deepsense_id_to_flukebook(ibs, deepsense_id):
    id_dict = ibs.ibeis_plugin_deepsense_ensure_id_map()
    if deepsense_id not in id_dict:
        # print warning bc we're missing a deepsense_id from our deepsense-flukebook map
        # print('[WARNING]: deepsense id %s is missing from the deepsense-flukebook ID map .csv' % deepsense_id)
        return str(deepsense_id)
    ans = id_dict[deepsense_id]
    return ans


@register_ibs_method
def ibeis_plugin_deepsense_ensure_backend(ibs, container_name='deepsense', **kwargs):
    global BACKEND_URL
    # make sure that the container is online using docker_control functions
    if BACKEND_URL is None:
        BACKEND_URLS = ibs.docker_ensure(container_name)
        if len(BACKEND_URLS) == 0:
            raise RuntimeError('Could not ensure container')
        elif len(BACKEND_URLS) == 1:
            BACKEND_URL = BACKEND_URLS[0]
        else:
            BACKEND_URL = BACKEND_URLS[0]
            args = (BACKEND_URLS, BACKEND_URL, )
            print('[WARNING] Multiple BACKEND_URLS:\n\tFound: %r\n\tUsing: %r' % args)
    return BACKEND_URL


@register_ibs_method
def ibeis_plugin_deepsense_ensure_id_map(ibs, container_name='deepsense'):
    global ID_MAP
    # make sure that the container is online using docker_control functions
    if ID_MAP is None:
        fpath = ut.grab_file_url(INDIVIDUAL_MAP_FPATH, appname='ibeis_deepsense', check_hash=True)
        csv_obj = ut.CSV.from_fpath(fpath, binary=False)
        ID_MAP = dict_from_csv(csv_obj)
    return ID_MAP


def dict_from_csv(csv_obj):
    import uuid
    id_dict = {}
    row_list = csv_obj.row_data
    row_list = row_list[1:]  # skip header row
    for row in row_list:
        deepsense_id = row[0]
        try:
            deepsense_id = int(deepsense_id)
        except:
            raise ValueError('Unable to cast provided Deepsense id %s to an int' % deepsense_id)
        assert deepsense_id not in id_dict, 'Deepsense-to-Flukebook id map contains two entries for deepsense ID %s' % deepsense_id

        flukebook_id = row[1]
        try:
            uuid.UUID(flukebook_id)
        except:
            raise ValueError('Unable to cast provided Flukebook id %s to a UUID' % flukebook_id)
        id_dict[deepsense_id] = flukebook_id
    return id_dict


@register_ibs_method
@register_api('/api/plugin/deepsense/identify/', methods=['GET'])
def ibeis_plugin_deepsense_identify(ibs, annot_uuid, use_depc=True, config={}, **kwargs):
    r"""
    Run the Kaggle winning Right-whale deepsense.ai ID algorithm

    Args:
        ibs         (IBEISController): IBEIS controller object
        annot_uuid  (uuid): Annotation for ID

    CommandLine:
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_identify
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_identify:0

    Example0:
        >>> # DISABLE_DOCTEST
        >>> import ibeis_deepsense
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> container_name = ut.get_argval('--container', default='deepsense')
        >>> print('Using container %s' % container_name)
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list, aid_list = ibs._ibeis_plugin_deepsense_init_testdb()
        >>> annot_uuid_list = ibs.get_annot_uuids(aid_list)
        >>> annot_name_list = ibs.get_annot_names(aid_list)
        >>> rank_list = []
        >>> score_list = []
        >>> for annot_uuid, annot_name in zip(annot_uuid_list, annot_name_list):
        >>>     resp_json = ibs.ibeis_plugin_deepsense_identify(annot_uuid, use_depc=False, container_name=container_name)
        >>>     rank, score = ibs._ibeis_plugin_deepsense_rank(resp_json, annot_name)
        >>>     print('[instant] for whale id = %s, got rank %d with score %0.04f' % (annot_name, rank, score, ))
        >>>     rank_list.append(rank)
        >>>     score_list.append('%0.04f' % score)
        >>> response_list = ibs.depc_annot.get('DeepsenseIdentification', aid_list, 'response')
        >>> rank_list_cache = []
        >>> score_list_cache = []
        >>> for annot_name, resp_json in zip(annot_name_list, response_list):
        >>>     rank, score = ibs._ibeis_plugin_deepsense_rank(resp_json, annot_name)
        >>>     print('[cache] for whale id = %s, got rank %d with score %0.04f' % (annot_name, rank, score, ))
        >>>     rank_list_cache.append(rank)
        >>>     score_list_cache.append('%0.04f' % score)
        >>> assert rank_list == rank_list_cache
        >>> # assert score_list == score_list_cache
        >>> result = (rank_list, score_list)
        ([0, -1, -1, 0], ['0.9052', '-1.0000', '-1.0000', '0.6986'])

    Example1:
        >>> # DISABLE_DOCTEST
        >>> import ibeis_deepsense
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> container_name = ut.get_argval('--container', default='deepsense')
        >>> print('Using container %s' % container_name)
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list, aid_list_ = ibs._ibeis_plugin_deepsense_init_testdb()
        >>> aid = aid_list_[3]
        >>> aid_list = [aid] * 10
        >>> annot_uuid_list = ibs.get_annot_uuids(aid_list)
        >>> annot_name_list = ibs.get_annot_names(aid_list)
        >>> rank_list = []
        >>> score_list = []
        >>> for annot_uuid, annot_name in zip(annot_uuid_list, annot_name_list):
        >>>     resp_json = ibs.ibeis_plugin_deepsense_identify(annot_uuid, use_depc=False, container_name=container_name)
        >>>     rank, score = ibs._ibeis_plugin_deepsense_rank(resp_json, annot_name)
        >>>     print('[instant] for whale id = %s, got rank %d with score %0.04f' % (annot_name, rank, score, ))
        >>>     rank_list.append(rank)
        >>>     score_list.append(score)
        >>> rank_list = np.array(rank_list)
        >>> score_list = np.array(score_list)
        >>> print(np.min(rank_list))
        >>> print(np.max(rank_list))
        >>> print(np.mean(rank_list))
        >>> print(np.std(rank_list))
        >>> print(np.min(score_list))
        >>> print(np.max(score_list))
        >>> print(np.mean(score_list))
        >>> print(np.std(score_list))
        >>> result = (rank_list, score_list)
        ([0, -1, -1, 0], ['0.9052', '-1.0000', '-1.0000', '0.6986'])
    """
    aid = aid_from_annot_uuid(ibs, annot_uuid)

    if use_depc:
        response_list = ibs.depc_annot.get('DeepsenseIdentification', [aid], 'response', config=config)
        response = response_list[0]
    else:
        response = ibs.ibeis_plugin_deepsense_identify_aid(aid, config=config, **kwargs)
    return response


def aid_from_annot_uuid(ibs, annot_uuid):
    annot_uuid_list = [annot_uuid]
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)
    annot_uuid_list = ensure_uuid_list(annot_uuid_list)
    # Ensure annotations
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    aid = aid_list[0]
    return aid


def get_b64_image(ibs, aid, **kwargs):
    image_path = deepsense_annot_chip_fpath(ibs, aid, **kwargs)
    pil_image = Image.open(image_path)
    byte_buffer = BytesIO()
    pil_image.save(byte_buffer, format="JPEG")
    b64_image = base64.b64encode(byte_buffer.getvalue()).decode("utf-8")
    return b64_image


@register_ibs_method
def ibeis_plugin_deepsense_identify_aid(ibs, aid, config={}, **kwargs):
    url = ibs.ibeis_plugin_deepsense_ensure_backend(**kwargs)
    b64_image = get_b64_image(ibs, aid, **config)
    data = {
        'image': b64_image,
        'configuration': {
            'top_n': 100,
            'threshold': 0.0,
        }
    }
    url = 'http://%s/api/classify' % (url)
    print('Sending identify to %s' % url)
    response = requests.post(url, json=data, timeout=120)
    assert response.status_code == 200
    response = response.json()
    response = update_response_with_flukebook_ids(ibs, response)
    return response


@register_ibs_method
def ibeis_plugin_deepsense_align_aid(ibs, aid, config={}, **kwargs):
    url = ibs.ibeis_plugin_deepsense_ensure_backend(**kwargs)
    b64_image = get_b64_image(ibs, aid, **config)
    data = {
        'image': b64_image,
    }
    url = 'http://%s/api/alignment' % (url)
    print('Sending alignment to %s' % url)
    response = requests.post(url, json=data, timeout=120)
    assert response.status_code == 200
    return response.json()


@register_ibs_method
def ibeis_plugin_deepsense_keypoint_aid(ibs, aid, alignment_result, config={}, **kwargs):
    url = ibs.ibeis_plugin_deepsense_ensure_backend(**kwargs)
    b64_image = get_b64_image(ibs, aid, **config)
    data = alignment_result.copy()
    data['image'] = b64_image
    url = 'http://%s/api/keypoints' % (url)
    print('Sending keypoints to %s' % url)
    response = requests.post(url, json=data, timeout=120)
    assert response.status_code == 200
    return response.json()


@register_ibs_method
@register_api('/api/plugin/deepsense/align/', methods=['GET'])
def ibeis_plugin_deepsense_align(ibs, annot_uuid, use_depc=True, config={}, **kwargs):
    r"""
    Run the Kaggle winning Right-whale deepsense.ai ID algorithm

    Args:
        ibs         (IBEISController): IBEIS controller object
        annot_uuid  (uuid): Annotation for ID

    CommandLine:
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_align
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_align:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis_deepsense
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> container_name = ut.get_argval('--container', default='deepsense')
        >>> print('Using container %s' % container_name)
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list, aid_list = ibs._ibeis_plugin_deepsense_init_testdb()
        >>> annot_uuid_list = ibs.get_annot_uuids(aid_list)
        >>> aligns_list = []
        >>> for annot_uuid in annot_uuid_list:
        >>>     resp_json = ibs.ibeis_plugin_deepsense_align(annot_uuid, use_depc=False, container_name=container_name)
        >>>     aligns_list.append(resp_json)
        >>> aligns_list_cache = ibs.depc_annot.get('DeepsenseAlignment', aid_list, 'response')
        >>> assert aligns_list == aligns_list_cache
        >>> result = aligns_list_cache
        [{'localization': {'bbox1': {'x': 994, 'y': 612}, 'bbox2': {'x': 1511, 'y': 1160}}}, {'localization': {'bbox1': {'x': 0, 'y': 408}, 'bbox2': {'x': 1128, 'y': 727}}}, {'localization': {'bbox1': {'x': 2376, 'y': 404}, 'bbox2': {'x': 3681, 'y': 1069}}}, {'localization': {'bbox1': {'x': 822, 'y': 408}, 'bbox2': {'x': 1358, 'y': 956}}}]
    """
    aid = aid_from_annot_uuid(ibs, annot_uuid)

    if use_depc:
        response_list = ibs.depc_annot.get('DeepsenseAlignment', [aid], 'response', config=config)
        response = response_list[0]
    else:
        response = ibs.ibeis_plugin_deepsense_align_aid(aid, config=config, **kwargs)
    return response


@register_ibs_method
@register_api('/api/plugin/deepsense/keypoint/', methods=['GET'])
def ibeis_plugin_deepsense_keypoint(ibs, annot_uuid, use_depc=True, config={}, **kwargs):
    r"""
    Run the Kaggle winning Right-whale deepsense.ai ID algorithm

    Args:
        ibs         (IBEISController): IBEIS controller object
        annot_uuid  (uuid): Annotation for ID

    CommandLine:
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_keypoint
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_keypoint:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis_deepsense
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> container_name = ut.get_argval('--container', default='deepsense')
        >>> print('Using container %s' % container_name)
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list, aid_list = ibs._ibeis_plugin_deepsense_init_testdb()
        >>> annot_uuid_list = ibs.get_annot_uuids(aid_list)
        >>> viewpoint_list = []
        >>> for annot_uuid in annot_uuid_list:
        >>>     resp_json = ibs.ibeis_plugin_deepsense_keypoint(annot_uuid, use_depc=False, container_name=container_name)
        >>>     viewpoint_list.append(resp_json)
        >>> viewpoint_list_cache = ibs.depc_annot.get('DeepsenseKeypoint', aid_list, 'response')
        >>> assert viewpoint_list == viewpoint_list_cache
        >>> result = viewpoint_list_cache
        [{'keypoints': {'blowhead': {'x': 1357, 'y': 963}, 'bonnet': {'x': 1151, 'y': 804}, 'angle': -142.33743653326957}}, {'keypoints': {'blowhead': {'x': 0, 'y': 724}, 'bonnet': {'x': 757, 'y': 477}, 'angle': -18.070882049942213}}, {'keypoints': {'blowhead': {'x': 3497, 'y': 404}, 'bonnet': {'x': 2875, 'y': 518}, 'angle': -190.38588712124752}}, {'keypoints': {'blowhead': {'x': 1098, 'y': 784}, 'bonnet': {'x': 1115, 'y': 523}, 'angle': -86.27335507676072}}]

    """
    aid = aid_from_annot_uuid(ibs, annot_uuid)

    if use_depc:
        # TODO: depc version
        response_list = ibs.depc_annot.get('DeepsenseKeypoint', [aid], 'response')
        response = response_list[0]
    else:
        alignment = ibs.ibeis_plugin_deepsense_align_aid(aid, config=config, **kwargs)
        response  = ibs.ibeis_plugin_deepsense_keypoint_aid(aid, alignment, config=config, **kwargs)
    return response


def deepsense_annot_chip_fpath(ibs, aid, dim_size=DIM_SIZE, **kwargs):
    config = {
        'dim_size': dim_size,
        'resize_dim': 'area',
        'ext': '.png',
    }
    fpath = ibs.get_annot_chip_fpath(aid, ensure=True, config2_=config)
    return fpath


@register_ibs_method
def ibeis_plugin_deepsense_illustration(ibs, annot_uuid, output=False, config={}, **kwargs):
    r"""
    Run the illustration examples

    Args:
        ibs         (IBEISController): IBEIS controller object
        annot_uuid  (uuid): Annotation for ID

    CommandLine:
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_illustration
        python -m ibeis_deepsense._plugin --test-ibeis_plugin_deepsense_illustration:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis_deepsense
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> container_name = ut.get_argval('--container', default='deepsense')
        >>> print('Using container %s' % container_name)
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list, aid_list = ibs._ibeis_plugin_deepsense_init_testdb()
        >>> annot_uuid_list = ibs.get_annot_uuids(aid_list)
        >>> for annot_uuid in annot_uuid_list:
        >>>     output_filepath_list = ibs.ibeis_plugin_deepsense_illustration(annot_uuid)
    """
    alignment = ibs.ibeis_plugin_deepsense_align(annot_uuid, config=config)
    keypoints = ibs.ibeis_plugin_deepsense_keypoint(annot_uuid, config=config)
    aid = aid_from_annot_uuid(ibs, annot_uuid)
    image_path = deepsense_annot_chip_fpath(ibs, aid, **config)
    # TODO write this func
    #image_path = ibs.get_deepsense_chip_fpath(aid)
    pil_img = Image.open(image_path)
    # draw a red box based on alignment on pil_image
    draw = ImageDraw.Draw(pil_img)
    # draw.rectangle(((0, 00), (100, 100)), fill="black")
    draw.rectangle(
        (
            (alignment['localization']['bbox1']['x'], alignment['localization']['bbox1']['y']),
            (alignment['localization']['bbox2']['x'], alignment['localization']['bbox2']['y']),
        ),
        outline='red',
        width=5,
    )

    blowhead = (keypoints['keypoints']['blowhead']['x'], keypoints['keypoints']['blowhead']['y'])
    blowhead_btm, blowhead_top = bounding_box_at_centerpoint(blowhead)
    draw.ellipse( (blowhead_btm, blowhead_top), outline="green", width=5)

    bonnet = (keypoints['keypoints']['bonnet']['x'], keypoints['keypoints']['bonnet']['y'])
    bonnet_btm, bonnet_top = bounding_box_at_centerpoint(bonnet)
    draw.ellipse( (bonnet_btm, bonnet_top), outline="blue", width=5)

    if output:
        local_path = dirname(abspath(__file__))
        output_path = abspath(join(local_path, '..', '_output'))
        ut.ensuredir(output_path)
        output_filepath_fmtstr = join(output_path, 'illustration-%s.png')
        output_filepath = output_filepath_fmtstr % (annot_uuid, )
        print('Writing to %s' % (output_filepath, ))
        pil_img.save(output_filepath)

    return pil_img


@register_ibs_method
def ibeis_plugin_deepsense_passport(ibs, annot_uuid, output=False, config={}, **kwargs):
    keypoints = ibs.ibeis_plugin_deepsense_keypoint(annot_uuid, config=config)
    aid = aid_from_annot_uuid(ibs, annot_uuid)
    image_path = deepsense_annot_chip_fpath(ibs, aid, **config)
    # TODO write this func
    #image_path = ibs.get_deepsense_chip_fpath(aid)
    pil_img = Image.open(image_path)

    # add padding on all sides of the image to prevent cutoff
    orig_size_np = np.array(pil_img.size)
    new_size = tuple(orig_size_np * 3)
    canvas = Image.new("RGB", new_size)
    canvas.paste(pil_img, pil_img.size)

    # get new coords of the blowhead and bonnet to use for rotation
    blowhead_np = np.array((keypoints['keypoints']['blowhead']['x'], keypoints['keypoints']['blowhead']['y']))
    blowhead_np += orig_size_np
    bonnet_np = np.array((keypoints['keypoints']['bonnet']['x'], keypoints['keypoints']['bonnet']['y']))
    bonnet_np += orig_size_np
    bonnet = tuple(bonnet_np)

    # rotate along the whale's axis
    angle = keypoints['keypoints']['angle']
    angle -= 90.0  # deepsense is left-aligned by default, we prefer top-aligned
    # translate coords are the difference from the blowhold to the center of the image
    blowhole = bonnet_np
    center = orig_size_np * 1.5
    translate = tuple(center - blowhole)
    canvas = canvas.rotate(angle, center=bonnet, translate=translate, resample=Image.NEAREST)

    # crop down to a square around the keypoints
    axis_line = blowhead_np - bonnet_np
    unit_size = np.hypot(axis_line[0], axis_line[1])
    crop_1 = center - np.array((unit_size, 1.5 * unit_size))
    crop_2 = center + np.array((unit_size, 0.5 * unit_size))
    # PIL.Image.crop needs a 4-tuple of ints for the crop function
    crop_box = tuple(np.concatenate((crop_1, crop_2)).astype(int))
    canvas = canvas.crop(crop_box)

    # resize the image to standard
    square_size = 1000
    canvas = canvas.resize((square_size, square_size), resample=Image.LANCZOS)
    # now draw ellipses on the blowhole and bonnet.
    # because of the rotation, centering, and now resizing, we know these will always be in the exact same pixel location
    draw = ImageDraw.Draw(canvas)
    bonnet_coords = bounding_box_at_centerpoint((square_size / 2, square_size / 4))
    draw.ellipse( bonnet_coords, outline="green", width=2)
    blowhole_coords = bounding_box_at_centerpoint((square_size / 2, square_size * 3 / 4))
    draw.ellipse( blowhole_coords, outline="blue", width=2)

    if output:
        local_path = dirname(abspath(__file__))
        output_path = abspath(join(local_path, '..', '_output'))
        ut.ensuredir(output_path)
        output_filepath_fmtstr = join(output_path, 'passport-%s.png')
        output_filepath = output_filepath_fmtstr % (annot_uuid, )
        print('Writing to %s' % (output_filepath, ))
        canvas.save(output_filepath)

    return canvas


def bounding_box_at_centerpoint(point, radius=15):
    point_less = tuple(coord - radius for coord in point)
    point_more = tuple(coord + radius for coord in point)
    return (point_less, point_more)


def update_response_with_flukebook_ids(ibs, response):
    for score_dict in response['identification']:
        deepsense_id = score_dict['whale_id']
        flukebook_id = ibs.ibeis_plugin_deepsense_id_to_flukebook(deepsense_id)
        score_dict['flukebook_id'] = flukebook_id
    return response


class DeepsenseIdentificationConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('dim_size', DIM_SIZE),
    ]


@register_preproc_annot(
    tablename='DeepsenseIdentification', parents=[ANNOTATION_TABLE],
    colnames=['response'], coltypes=[dict],
    configclass=DeepsenseIdentificationConfig,
    fname='deepsense',
    chunksize=4)
def ibeis_plugin_deepsense_identify_deepsense_ids_depc(depc, aid_list, config):
    # The doctest for ibeis_plugin_deepsense_identify_deepsense_ids also covers this func
    ibs = depc.controller
    for aid in aid_list:
        response = ibs.ibeis_plugin_deepsense_identify_aid(aid, config=config)
        yield (response, )


class DeepsenseAlignmentConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('dim_size', DIM_SIZE),
    ]


@register_preproc_annot(
    tablename='DeepsenseAlignment', parents=[ANNOTATION_TABLE],
    colnames=['response'], coltypes=[dict],
    configclass=DeepsenseAlignmentConfig,
    fname='deepsense',
    chunksize=128)
def ibeis_plugin_deepsense_align_deepsense_ids_depc(depc, aid_list, config):
    # The doctest for ibeis_plugin_deepsense_identify_deepsense_ids also covers this func
    ibs = depc.controller
    for aid in aid_list:
        response = ibs.ibeis_plugin_deepsense_align_aid(aid, config=config)
        yield (response, )


class DeepsenseKeypointsConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('dim_size', DIM_SIZE),
    ]


@register_preproc_annot(
    tablename='DeepsenseKeypoint', parents=['DeepsenseAlignment'],
    colnames=['response'], coltypes=[dict],
    configclass=DeepsenseKeypointsConfig,
    fname='deepsense',
    chunksize=128)
def ibeis_plugin_deepsense_keypoint_deepsense_ids_depc(depc, alignment_rowids, config):
    # The doctest for ibeis_plugin_deepsense_identify_deepsense_ids also covers this func
    ibs = depc.controller
    alignments = depc.get_native('DeepsenseAlignment', alignment_rowids, 'response')
    aid_list = depc.get_ancestor_rowids('DeepsenseAlignment', alignment_rowids)
    for alignment, aid in zip(alignments, aid_list):
        response = ibs.ibeis_plugin_deepsense_keypoint_aid(aid, alignment, config=config)
        yield (response, )


class DeepsenseIllustrationConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('dim_size', DIM_SIZE),
        ut.ParamInfo('ext', '.jpg')
    ]


def pil_image_load(absolute_path):
    pil_img = Image.open(absolute_path)
    return pil_img


def pil_image_write(absolute_path, pil_img):
    pil_img.save(absolute_path)


@register_preproc_annot(
    tablename='DeepsenseIllustration', parents=[ANNOTATION_TABLE],
    colnames=['image'], coltypes=[('extern', pil_image_load, pil_image_write)],
    configclass=DeepsenseIllustrationConfig,
    fname='deepsense',
    chunksize=128)
def ibeis_plugin_deepsense_illustrate_deepsense_ids_depc(depc, aid_list, config):
    # The doctest for ibeis_plugin_deepsense_identify_deepsense_ids also covers this func
    ibs = depc.controller
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    for annot_uuid in annot_uuid_list:
        response = ibs.ibeis_plugin_deepsense_illustration(annot_uuid, config=config)
        yield (response, )


class DeepsensePassportConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('dim_size', DIM_SIZE),
        ut.ParamInfo('ext', '.jpg')
    ]


@register_preproc_annot(
    tablename='DeepsensePassport', parents=[ANNOTATION_TABLE],
    colnames=['image'], coltypes=[('extern', pil_image_load, pil_image_write)],
    configclass=DeepsensePassportConfig,
    fname='deepsense',
    chunksize=128)
def ibeis_plugin_deepsense_passport_deepsense_ids_depc(depc, aid_list, config):
    # The doctest for ibeis_plugin_deepsense_identify_deepsense_ids also covers this func
    ibs = depc.controller
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    for annot_uuid in annot_uuid_list:
        response = ibs.ibeis_plugin_deepsense_passport(annot_uuid, config=config)
        yield (response, )


def get_match_results(depc, qaid_list, daid_list, score_list, config):
    """ converts table results into format for ipython notebook """
    #qaid_list, daid_list = request.get_parent_rowids()
    #score_list = request.score_list
    #config = request.config

    unique_qaids, groupxs = ut.group_indices(qaid_list)
    #grouped_qaids_list = ut.apply_grouping(qaid_list, groupxs)
    grouped_daids = ut.apply_grouping(daid_list, groupxs)
    grouped_scores = ut.apply_grouping(score_list, groupxs)

    ibs = depc.controller
    unique_qnids = ibs.get_annot_nids(unique_qaids)

    # scores
    _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_scores)
    for qaid, qnid, daids, scores in _iter:
        dnids = ibs.get_annot_nids(daids)

        # Remove distance to self
        annot_scores = np.array(scores)
        daid_list_ = np.array(daids)
        dnid_list_ = np.array(dnids)

        is_valid = (daid_list_ != qaid)
        daid_list_ = daid_list_.compress(is_valid)
        dnid_list_ = dnid_list_.compress(is_valid)
        annot_scores = annot_scores.compress(is_valid)

        # Hacked in version of creating an annot match object
        match_result = ibeis.AnnotMatch()
        match_result.qaid = qaid
        match_result.qnid = qnid
        match_result.daid_list = daid_list_
        match_result.dnid_list = dnid_list_
        match_result._update_daid_index()
        match_result._update_unique_nid_index()

        grouped_annot_scores = vt.apply_grouping(annot_scores, match_result.name_groupxs)
        name_scores = np.array([np.sum(dists) for dists in grouped_annot_scores])
        match_result.set_cannonical_name_score(annot_scores, name_scores)
        yield match_result


class DeepsenseConfig(dt.Config):  # NOQA
    """
    CommandLine:
        python -m ibeis_deepsense._plugin --test-DeepsenseConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_deepsense._plugin import *  # NOQA
        >>> config = DeepsenseConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        Deepsense(dim_size=2000)
    """
    def get_param_info_list(self):
        return [
            ut.ParamInfo('dim_size', DIM_SIZE),
        ]


class DeepsenseRequest(dt.base.VsOneSimilarityRequest):
    _symmetric = False
    _tablename = 'Deepsense'

    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(request, aid_list, config=None):
        depc = request.depc
        ibs = depc.controller
        passport_paths = ibs.depc_annot.get('DeepsensePassport', aid_list, 'image', config=config, read_extern=False, ensure=True)
        passports = list(map(vt.imread, passport_paths))
        return passports

    def render_single_result(request, cm, aid, **kwargs):
        # HACK FOR WEB VIEWER
        chips = request.get_fmatch_overlayed_chip([cm.qaid, aid], config=request.config)
        import vtool as vt
        out_img = vt.stack_image_list(chips)
        return out_img

    def postprocess_execute(request, parent_rowids, result_list):
        qaid_list, daid_list = list(zip(*parent_rowids))
        score_list = ut.take_column(result_list, 0)
        depc = request.depc
        config = request.config
        cm_list = list(get_match_results(depc, qaid_list, daid_list,
                                         score_list, config))
        return cm_list

    def execute(request, *args, **kwargs):
        kwargs['use_cache'] = False
        result_list = super(DeepsenseRequest, request).execute(*args, **kwargs)
        qaids = kwargs.pop('qaids', None)
        if qaids is not None:
            result_list = [
                result for result in result_list
                if result.qaid in qaids
            ]
        return result_list


@register_preproc_annot(
    tablename='Deepsense', parents=[ANNOTATION_TABLE, ANNOTATION_TABLE],
    colnames=['score'], coltypes=[float],
    configclass=DeepsenseConfig,
    requestclass=DeepsenseRequest,
    fname='deepsense',
    rm_extern_on_delete=True,
    chunksize=None)
def ibeis_plugin_deepsense(depc, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m ibeis_deepsense._plugin --exec-ibeis_plugin_deepsense
        python -m ibeis_deepsense._plugin --exec-ibeis_plugin_deepsense:0

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_deepsense._plugin import *
        >>> import ibeis
        >>> import itertools as it
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> depc = ibs.depc_annot
        >>> gid_list, aid_list = ibs._ibeis_plugin_deepsense_init_testdb()
        >>>  # For tests, make a (0, 0, 1, 1) bbox with the same name in the same image for matching
        >>> annot_uuid_list = ibs.get_annot_uuids(aid_list)
        >>> annot_name_list = ibs.get_annot_names(aid_list)
        >>> aid_list_ = ibs.add_annots(gid_list, [(0, 0, 1, 1)] * len(gid_list), name_list=annot_name_list)
        >>> qaid = aid_list[0]
        >>> qannot_name = annot_name_list[0]
        >>> qaid_list = [qaid]
        >>> daid_list = aid_list + aid_list_
        >>> root_rowids = tuple(zip(*it.product(qaid_list, daid_list)))
        >>> config = DeepsenseConfig()
        >>> # Call function via request
        >>> request = DeepsenseRequest.new(depc, qaid_list, daid_list)
        >>> result = request.execute()
        >>> am = result[0]
        >>> unique_nids = am.unique_nids
        >>> name_score_list = am.name_score_list
        >>> unique_name_text_list = ibs.get_name_texts(unique_nids)
        >>> name_score_list_ = ['%0.04f' % (score, ) for score in am.name_score_list]
        >>> name_score_dict = dict(zip(unique_name_text_list, name_score_list_))
        >>> print('Queried Deepsense algorithm for ground-truth ID = %s' % (qannot_name, ))
        >>> result = ut.repr3(name_score_dict)
        {
            '64edec9a-b998-4f96-a9d6-6dddcb8f8c0a': '0.8082',
            '825c5de0-d764-464c-91b6-9e507c5502fd': '0.0000',
            'bf017955-9ed9-4311-96c9-eed4556cdfdf': '0.0000',
            'e36c9f90-6065-4354-822d-c0fef25441ad': '0.0001',
        }
    """
    ibs = depc.controller

    qaids = list(set(qaid_list))
    daids = list(set(daid_list))

    assert len(qaids) == 1
    qaid = qaids[0]
    annot_uuid = ibs.get_annot_uuids(qaid)
    resp_json = ibs.ibeis_plugin_deepsense_identify(annot_uuid, use_depc=True, config=config)
    # update response_json to use flukebook names instead of deepsense

    dnames = ibs.get_annot_name_texts(daids)
    name_counter_dict = {}
    for daid, dname in zip(daids, dnames):
        if dname in [None, const.UNKNOWN]:
            continue
        if dname not in name_counter_dict:
            name_counter_dict[dname] = 0
        name_counter_dict[dname] += 1

    ids = resp_json['identification']
    name_score_dict = {}
    for rank, result in enumerate(ids):
        name = result['flukebook_id']
        name_score = result['probability']
        name_counter = name_counter_dict.get(name, 0)
        if name_counter <= 0:
            if name_score > 0.01:
                args = (name, rank, name_score, len(daids), )
                print('Suggested match name = %r (rank %d) with score = %0.04f is not in the daids (total %d)' % args)
            continue
        assert name_counter >= 1
        annot_score = name_score / name_counter

        assert name not in name_score_dict, 'Deepsense API response had multiple scores for name = %r' % (name, )
        name_score_dict[name] = annot_score

    dname_list = ibs.get_annot_name_texts(daid_list)
    for qaid, daid, dname in zip(qaid_list, daid_list, dname_list):
        value = name_score_dict.get(dname, 0)
        yield (value, )


# @register_ibs_method
# def deepsense_embed(ibs):
#     ut.embed()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_deepsense._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
