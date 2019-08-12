from __future__ import absolute_import, division, print_function
import ibeis
from ibeis.control import controller_inject, docker_control
import utool as ut
import dtool as dt
import numpy as np
import vtool as vt
from PIL import Image
from io import BytesIO
import base64
import requests
from ibeis.constants import ANNOTATION_TABLE
from ibeis.web.apis_engine import ensure_uuid_list
import ibeis.constants as const


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
ROOT = const.ANNOTATION_TABLE


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
    annot_id_list = [int(splitext(split(uri)[1])[0]) for uri in uri_list]
    aid_list = ibs.use_images_as_annotations(gid_list)
    return aid_list, annot_id_list


@register_ibs_method
def ibeis_plugin_deepsense_ensure_backend(ibs, container_name='deepsense'):
    global BACKEND_URL
    # make sure that the container is online using docker_control functions
    if BACKEND_URL is None:
        BACKEND_URL = ibs.docker_ensure(container_name)
    return BACKEND_URL


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
        >>> from ibeis_deepsense._plugin import _ibeis_plugin_deepsense_rank  # NOQA
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> from os.path import abspath, exists, join, dirname, split, splitext
        >>> container_name = ut.get_argval('--container', default='deepsense')
        >>> print('Using container %s' % container_name)
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list, annot_id_list = deepsense._ibeis_plugin_deepsense_init_testdb(ibs)
        >>> uuid_list = ibs.get_annot_uuids(aid_list)
        >>> rank_list = []
        >>> score_list = []
        >>> for image_id, annot_uuid in zip(image_id_list, uuid_list):
        >>>     resp_json = ibs.ibeis_plugin_deepsense_identify(annot_uuid, use_depc=False, container_name=container_name)
        >>>     rank, score = _ibeis_plugin_deepsense_rank(resp_json, image_id)
        >>>     print('[instant] for whale id = %s, got rank %d with score %0.04f' % (image_id, rank, score, ))
        >>>     rank_list.append(rank)
        >>>     score_list.append('%0.04f' % score)
        >>> response_list = ibs.depc_annot.get('DeepsenseIdentification', aid_list, 'response')
        >>> rank_list_cache = []
        >>> score_list_cache = []
        >>> for image_id, resp_json in zip(image_id_list, response_list):
        >>>     rank, score = _ibeis_plugin_deepsense_rank(resp_json, image_id)
        >>>     print('[cache] for whale id = %s, got rank %d with score %0.04f' % (image_id, rank, score, ))
        >>>     rank_list_cache.append(rank)
        >>>     score_list_cache.append('%0.04f' % score)
        >>> assert rank_list == rank_list_cache
        >>> assert score_list == score_list_cache
        >>> result = (rank_list, score_list)
        ([-1, 0, 0, 3, -1], ['-1.0000', '0.9205', '0.1283', '0.0386', '-1.0000'])
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
    return response


@register_ibs_method
def ibeis_plugin_deepsense_identify_aid(ibs, aid, **kwargs):
    url = ibs.ibeis_plugin_deepsense_ensure_backend(**kwargs)

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
    url = 'http://%s/api/classify' % (url)
    print('Sending identify to %s' % url)
    response = requests.post(url, json=data)
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


class DeepsenseRequest(dt.base.VsOneSimilarityRequest):  # NOQA
    _symmetric = False

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


class DeepsenseConfig(dt.Config):  # NOQA
    """
    CommandLine:
        python -m ibeis_deepsense._plugin_depc --test-DeepsenseConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_deepsense._plugin_depc import *  # NOQA
        >>> config = DeepsenseConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        TODO
    """
    def get_param_info_list(self):
        return []


class DeepsenseRequest(DeepsenseRequest):  # NOQA
    _tablename = 'Deepsense'


@register_preproc_annot(
    tablename='Deepsense', parents=[ROOT, ROOT],
    colnames=['score'], coltypes=[float],
    configclass=DeepsenseConfig,
    requestclass=DeepsenseRequest,
    fname='deepsense',
    rm_extern_on_delete=True,
    chunksize=None)
def ibeis_plugin_deepsense(depc, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m ibeis_deepsense._plugin_depc --exec-ibeis_plugin_deepsense --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis_deepsense
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> from os.path import abspath, exists, join, dirname, split, splitext
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list, annot_id_list = deepsense._ibeis_plugin_deepsense_init_testdb(ibs)
        >>> root_rowids = tuple(zip(*it.product(aid_list, aid_list)))
        >>> qaid_list, daid_list = root_rowids
        >>> config = DeepsenseConfig()
        >>> # Call function via request
        >>> request = DeepsenseRequest.new(depc, aid_list, aid_list)
        >>> am_list1 = request.execute()
        >>> # Call function via depcache
        >>> prop_list = depc.get('Deepsense', root_rowids)
        >>> # Call function normally
        >>> score_list = list(ibeis_plugin_deepsense(depc, qaid_list, daid_list, config))
        >>> am_list2 = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        >>> assert score_list == prop_list, 'error in cache'
        >>> assert np.all(am_list1[0].score_list == am_list2[0].score_list)
        >>> ut.quit_if_noshow()
        >>> am = am_list2[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()
    """
    ibs = depc.controller
    ut.embed()

    qaids = list(set(qaid_list))
    daids = list(set(daid_list))

    assert len(qaids) == 1
    qaid = qaids[0]
    annot_uuid = ibs.get_annot_uuids(qaid, use_depc=True)
    resp_json = ibs.ibeis_plugin_deepsense_identify(annot_uuid, use_depc=False)

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
        whale_id = result['whale_id']
        name_score = result['probability']

        name_counter = name_counter_dict.get(whale_id, 0)
        if name_counter <= 0:
            args = (whale_id, rank, len(daids), )
            print('Suggested match whale_id = %r (rank %d) is not in the daids (total %d)' % args)
            continue
        assert name_counter >= 1
        annot_score = name_score / name_counter

        assert whale_id not in name_score_dict, 'Deepsense API response had multiple scores for whale_id = %r' % (whale_id, )
        name_counter_dict[whale_id] = annot_score

    dname_list = ibs.get_annot_name_texts(daid_list)
    for qaid, daid, dname in zip(qaid_list, daid_list, dname_list):
        value = name_counter_dict.get(dname, -1)
        yield value


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_deepsense._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
