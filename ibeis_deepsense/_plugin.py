from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject
from ibeis_deepsense import docker_control  # NOQA
import utool as ut
from PIL import Image
from io import BytesIO
import base64
import requests

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

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
def ibeis_plugin_deepsense_identify(ibs, annot_uuid):

    url = ibs.ibeis_plugin_deepsense_ensure_backend()

    annot_uuid_list = [annot_uuid]
    ibs.web_check_uuids(qannot_uuid_list=annot_uuid_list)
    annot_uuid_list = ibs.ensure_uuid_list(annot_uuid_list)
    # Ensure annotations
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    aid = aid_list[0]

    image_path = ibs.get_annot_chips_fpath(aid)
    pil_image = Image.open(image_path)
    byte_buffer = BytesIO()
    pil_image.save(byte_buffer, format="JPEG")
    b64_image = base64.b64encode(byte_buffer.getvalue()).decode("utf-8")

    data = {
        'image': b64_image,
    }
    response = requests.post('%s/api/alignment' % (url), json=data)
    return response


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_deepsense._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
