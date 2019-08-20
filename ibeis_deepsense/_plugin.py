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
from PIL import Image
from io import BytesIO


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


INDIVIDUAL_MAP_FPATH = 'https://cthulhu.dyn.wildme.io/public/random/deepsense.flukebook.v0.csv'
ID_MAP = None


def _ibeis_plugin_deepsense_check_container(url):
    ut.embed()


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
    annot_name_list = [str(int(splitext(split(uri)[1])[0])) for uri in uri_list]
    aid_list = ibs.use_images_as_annotations(gid_list)
    ibs.set_annot_names(aid_list, annot_name_list)
    return gid_list, aid_list


@register_ibs_method
def _ibeis_plugin_deepsense_rank(ibs, response_json, desired_name):
    ids = response_json['identification']
    for index, result in enumerate(ids):
        whale_id = result['whale_id']
        probability = result['probability']
        name = str(whale_id)
        if name == desired_name:
            return (index, probability)
    return (-1, -1)


# This method converts from the ibeis/Flukebook individual UUIDs to the Deepsense/
# NEAQ IDs used by the deepsense container.
@register_ibs_method
def deepsense_id_to_flukebook(ibs, deepsense_id):
    id_dict = ibs.ibeis_plugin_deepsense_ensure_id_map()
    if deepsense_id not in id_dict:
        # print warning bc we're missing a deepsense_id from our deepsense-flukebook map
        # print('[WARNING]: deepsense id %s is missing from the deepsense-flukebook ID map .csv' % deepsense_id)
        return str(deepsense_id)
    ans = id_dict[deepsense_id]
    return ans


@register_ibs_method
def ibeis_plugin_deepsense_ensure_backend(ibs, container_name='deepsense'):
    global BACKEND_URL
    # make sure that the container is online using docker_control functions
    if BACKEND_URL is None:
        BACKEND_URL = ibs.docker_ensure(container_name)
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
def ibeis_plugin_deepsense_identify(ibs, annot_uuid, use_depc=True, **kwargs):
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
        >>> assert score_list == score_list_cache
        >>> result = (rank_list, score_list)
        ([-1, 0, 1, 0, -1], ['-1.0000', '0.9205', '0.1283', '0.0386', '-1.0000'])
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
        response = ibs.ibeis_plugin_deepsense_identify_aid(aid, **kwargs)
    # ut.embed()
    response = ibs.update_response_with_flukebook_ids(response)
    return response


@register_ibs_method
def ibeis_plugin_deepsense_identify_aid(ibs, aid, **kwargs):
    url = ibs.ibeis_plugin_deepsense_ensure_backend(**kwargs)
    b64_image = ibs.get_b64_image(aid)
    data = {
        'image': b64_image,
        'configuration': {
            'top_n': 100,
            'threshold': 0.0,
        }
    }
    url = 'http://%s/api/classify' % (url)
    print('Sending identify to %s' % url)
    response = requests.post(url, json=data)
    assert response.status_code == 200
    return response.json()


@register_ibs_method
def get_b64_image(ibs, aid):
    image_path = ibs.get_annot_chip_fpath(aid)
    pil_image = Image.open(image_path)
    byte_buffer = BytesIO()
    pil_image.save(byte_buffer, format="JPEG")
    b64_image = base64.b64encode(byte_buffer.getvalue()).decode("utf-8")
    return b64_image


@register_ibs_method
def aid_from_annot_uuid(ibs, annot_uuid):
    annot_uuid_list = [annot_uuid]
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)
    annot_uuid_list = ensure_uuid_list(annot_uuid_list)
    # Ensure annotations
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    aid = aid_list[0]
    return aid


@register_ibs_method
def ibeis_plugin_deepsense_align_aid(ibs, aid, **kwargs):
    url = ibs.ibeis_plugin_deepsense_ensure_backend(**kwargs)
    b64_image = ibs.get_b64_image(aid)
    data = {
        'image': b64_image,
    }
    url = 'http://%s/api/alignment' % (url)
    print('Sending alignment to %s' % url)
    response = requests.post(url, json=data)
    assert response.status_code == 200
    return response.json()


@register_ibs_method
def ibeis_plugin_deepsense_keypoint_aid(ibs, aid, alignment_result, **kwargs):
    url = ibs.ibeis_plugin_deepsense_ensure_backend(**kwargs)
    b64_image = ibs.get_b64_image(aid)
    data = alignment_result.copy()
    data['image'] = b64_image
    url = 'http://%s/api/keypoints' % (url)
    print('Sending keypoints to %s' % url)
    response = requests.post(url, json=data)
    assert response.status_code == 200
    return response.json()


# TODO finish tests
@register_ibs_method
@register_api('/api/plugin/deepsense/align/', methods=['GET'])
def ibeis_plugin_deepsense_align(ibs, annot_uuid, use_depc=True, **kwargs):
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
        [{'localization': {'bbox1': {'x': 0, 'y': 82}, 'bbox2': {'x': 694, 'y': 201}}}, {'localization': {'bbox1': {'x': 235, 'y': 116}, 'bbox2': {'x': 388, 'y': 273}}}, {'localization': {'bbox1': {'x': 136, 'y': 215}, 'bbox2': {'x': 257, 'y': 332}}}, {'localization': {'bbox1': {'x': 278, 'y': 175}, 'bbox2': {'x': 453, 'y': 339}}}, {'localization': {'bbox1': {'x': 0, 'y': 134}, 'bbox2': {'x': 442, 'y': 207}}}]
    """
    aid = ibs.aid_from_annot_uuid(annot_uuid)

    if use_depc:
        response_list = ibs.depc_annot.get('DeepsenseAlignment', [aid], 'response')
        response = response_list[0]
    else:
        response = ibs.ibeis_plugin_deepsense_align_aid(aid, **kwargs)
    return response


# TODO finish tests
# TODO: depc version
@register_ibs_method
@register_api('/api/plugin/deepsense/keypoint/', methods=['GET'])
def ibeis_plugin_deepsense_keypoint(ibs, annot_uuid, use_depc=False, **kwargs):
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
        [{'keypoints': {'blowhead': {'x': 688, 'y': 82}, 'bonnet': {'x': 417, 'y': 152}, 'angle': -194.48308190914116}}, {'keypoints': {'blowhead': {'x': 316, 'y': 225}, 'bonnet': {'x': 319, 'y': 146}, 'angle': -87.82525588538995}}, {'keypoints': {'blowhead': {'x': 166, 'y': 246}, 'bonnet': {'x': 219, 'y': 274}, 'angle': -332.1524217402118}}, {'keypoints': {'blowhead': {'x': 388, 'y': 276}, 'bonnet': {'x': 329, 'y': 233}, 'angle': -143.91492695714788}}, {'keypoints': {'blowhead': {'x': 438, 'y': 134}, 'bonnet': {'x': 366, 'y': 134}, 'angle': -180.0}}]

    """
    aid = ibs.aid_from_annot_uuid(annot_uuid)

    if use_depc:
        # TODO: depc version
        response_list = ibs.depc_annot.get('DeepsenseKeypoint', [aid], 'response')
        response = response_list[0]
    else:
        alignment = ibs.ibeis_plugin_deepsense_align_aid(aid, **kwargs)
        response  = ibs.ibeis_plugin_deepsense_keypoint_aid(aid, alignment, **kwargs)
    # ut.embed()
    # response = ibs.update_response_with_flukebook_ids(response)
    return response


@register_ibs_method
def update_response_with_flukebook_ids(ibs, response):
    for score_dict in response['identification']:
        deepsense_id = score_dict['whale_id']
        flukebook_id = ibs.deepsense_id_to_flukebook(deepsense_id)
        score_dict['flukebook_id'] = flukebook_id
    return response


class DeepsenseIdentificationConfig(dt.Config):  # NOQA
    _param_info_list = []

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
        response = ibs.ibeis_plugin_deepsense_identify_aid(aid)
        yield (response, )


class DeepsenseAlignmentConfig(dt.Config):  # NOQA
    _param_info_list = []

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
        response = ibs.ibeis_plugin_deepsense_align_aid(aid)
        yield (response, )

class DeepsenseKeypointsConfig(dt.Config):  # NOQA
    _param_info_list = []


@register_preproc_annot(
    tablename='DeepsenseKeypoint', parents=['DeepsenseAlignment'],
    colnames=['response'], coltypes=[dict],
    configclass=DeepsenseKeypointsConfig,
    fname='deepsense',
    chunksize=128)
def ibeis_plugin_deepsense_align_deepsense_ids_depc(depc, alignment_rowids, config):
    # The doctest for ibeis_plugin_deepsense_identify_deepsense_ids also covers this func
    ibs = depc.controller
    alignments = depc.get_native('DeepsenseAlignment', alignment_rowids, 'response')
    aid_list = depc.get_ancestor_rowids('DeepsenseAlignment', alignment_rowids)
    for alignment, aid in zip(alignments, aid_list):
        response = ibs.ibeis_plugin_deepsense_keypoint_aid(aid, alignment)
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
        Deepsense()
    """
    def get_param_info_list(self):
        return []


class DeepsenseRequest(dt.base.VsOneSimilarityRequest):  # NOQA
    _symmetric = False
    _tablename = 'Deepsense'

    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(request, aid_list, config=None):
        depc = request.depc
        ibs = depc.controller
        chips = ibs.get_annot_chips(aid_list, config=request.config)
        return chips

    def render_single_result(request, cm, aid, **kwargs):
        # HACK FOR WEB VIEWER
        overlay = kwargs.get('draw_fmatches')
        chips = request.get_fmatch_overlayed_chip([cm.qaid, aid], overlay=overlay,
                                                  config=request.config)
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
        >>> # ENABLE_DOCTEST
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
        >>> qaid_list = aid_list[1:2]
        >>> daid_list = aid_list + aid_list_
        >>> root_rowids = tuple(zip(*it.product(qaid_list, daid_list)))
        >>> config = DeepsenseConfig()
        >>> # Call function via request
        >>> request = DeepsenseRequest.new(depc, qaid_list, daid_list)
        >>> result = request.execute()
        [(-1.0,), (-1.0,), (1.2908857570437249e-05,), (-1.0,), (-1.0,), (0.9205465316772461,), (-1.0,), (1.2908857570437249e-05,), (-1.0,)]
    """
    ibs = depc.controller

    qaids = list(set(qaid_list))
    daids = list(set(daid_list))

    assert len(qaids) == 1
    qaid = qaids[0]
    annot_uuid = ibs.get_annot_uuids(qaid)
    resp_json = ibs.ibeis_plugin_deepsense_identify(annot_uuid, use_depc=True)
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


@register_ibs_method
def deepsense_embed(ibs):
    ut.embed()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_deepsense._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
