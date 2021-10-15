from visdom import Visdom
import sys
from utils import get_localhost
remove_vis = sys.argv[1]
vis = Visdom(get_localhost())
vis.delete_env('main')
assert remove_vis in vis.get_env_list()
if remove_vis in vis.get_env_list():
    print('Removing ' + remove_vis)
    vis.delete_env(remove_vis)
