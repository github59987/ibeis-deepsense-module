from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject
import utool as ut
(print, rrr, profile) = ut.inject2(__name__)


try:
    import docker
    DOCKER_CLIENT = docker.from_env()
    assert DOCKER_CLIENT is not None
except:
    print('Local docker client is not available')
    DOCKER_CLIENT = None
    raise RuntimeError('Failed to connect to Docker for Deepsense Plugin')


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)


@register_ibs_method
def docker_image_list():
    tag_list = []
    for image in DOCKER_CLIENT.images.list():
        print(image)
        tag_list += image.tags
    tag_list = list(set(tag_list))
    return tag_list


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_deepsense.docker_control --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
