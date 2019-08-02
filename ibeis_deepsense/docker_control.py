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


DOCKER_CONFIG_REGISTRY = {}
DOCKER_IMAGE_PREFIX = 'wildme.azurecr.io'


@register_ibs_method
def docker_register_config(container_name, image_name, container_check_func=None):
    if container_name in DOCKER_CONFIG_REGISTRY:
        raise RuntimeError('Container name has already been added to the config registry')
    if DOCKER_IMAGE_PREFIX is not None and not image_name.startswith(DOCKER_IMAGE_PREFIX):
        raise RuntimeError('Cannot register an image name that does not have the prefix = %r' % (DOCKER_IMAGE_PREFIX, ))
    DOCKER_CONFIG_REGISTRY[container_name] = {
        'image': image_name,
    }


@register_ibs_method
def docker_image_list(ibs):
    tag_list = []
    for image in DOCKER_CLIENT.images.list():
        print(image)
        tag_list += image.tags
    tag_list = sorted(list(set(tag_list)))
    return tag_list


# TODO next--- download the image from remote server
def docker_pull_image(ibs, image_name):
    pass


def docker_login(ibs):
    # login to the ACR with credentials from "somewhere"
    pass


def docker_image_run(ibs, port=6000, volumes=None):
    pass


@register_ibs_method
def docker_container_status_dict(ibs):
    container_dict = {}
    for container in DOCKER_CLIENT.containers.list():
        if container.status not in container_dict:
            container_dict[container.status] = []
        container_dict[container.status].append(container.name)
    for status in container_dict:
        container_dict[status] = sorted(list(set(container_dict[status])))
    return container_dict


@register_ibs_method
def docker_container_status(ibs, name):
    container_dict = ibs.docker_container_status_dict()
    for status in container_dict:
        if name in container_dict[status]:
            return status
    return None


@register_ibs_method
def docker_container_url(ibs, name):
    if ibs.docker_container_status(name) != 'running':
        return None


@register_ibs_method
def docker_ensure(ibs, container_name):
    config = DOCKER_CONFIG_REGISTRY.get(container_name, None)

    message = 'The container name has not been registered' % (container_name, )
    assert config is not None, message

    # Check for container in running containers
    if ibs.docker_container_status(container_name) == 'running':
        # find out the port / url of the container
        return

    # If exists, return that container object

    # If not, check if the image has been downloaded from the config

    # If exists, start image into a container

    # If not, download model (may require login)

    # return container


@register_ibs_method
def docker_embed(ibs):
    ut.embed()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_deepsense.docker_control --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
