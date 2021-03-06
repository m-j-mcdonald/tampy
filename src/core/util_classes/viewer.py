from IPython import embed as shell
from openrave_body import OpenRAVEBody
from openravepy import Environment
from core.internal_repr.parameter import Object
from core.util_classes.pr2 import PR2
from core.util_classes.can import Can
from core.util_classes.table import Table
from core.util_classes.box import Box
import numpy as np
import time, os, os.path as osp, shutil, scipy.misc, subprocess


class Viewer(object):
    """
    Defines viewers for visualizing execution.
    """
    def __init__(self, viewer = None):
        self.viewer = viewer

class GridWorldViewer(Viewer):
    def initialize_from_workspace(self, workspace):
        pass

class OpenRAVEViewer(Viewer):

    _viewer = None

    def __init__(self, env = None):
        assert OpenRAVEViewer._viewer == None
        if env == None:
            self.env = Environment()
        else:
            self.env = env
        self.env.SetViewer('qtcoin')
        self.name_to_rave_body = {}
        OpenRAVEViewer._viewer = self

    def clear(self):
        for b in self.name_to_rave_body.itervalues():
            b.delete()
        self.name_to_rave_body = {}

    @staticmethod
    def create_viewer():
        # if reset and OpenRAVEViewer._viewer != None:
        #     ## close the old viewer to avoid a threading error
        #     OpenRAVEViewer._viewer = None
        if OpenRAVEViewer._viewer == None:
            return OpenRAVEViewer()
        OpenRAVEViewer._viewer.clear()
        return OpenRAVEViewer._viewer

    def record_plan(self, plan, outf, res = (640, 480), cam=None):
        """
        creates a video of a plan and stores it in outf
        """
        obj_list = []
        horizon = plan.horizon
        v = self.env.GetViewer()
        if osp.exists('.video_images'):
            shutil.rmtree('.video_images')
        os.makedirs('.video_images')
        for p in plan.params.itervalues():
            if not p.is_symbol():
                obj_list.append(p)
        for t in range(horizon):
            self.draw(obj_list, t)
            time.sleep(0.1)
            if cam is None:
                cam = v.GetCameraTransform()
            im = v.GetCameraImage(res[0], res[1], cam,[640,640,320,240])
            scipy.misc.imsave('.video_images/frame_'+str('%05d'%t)+'.png', im)
        outfname = "{}.mp4".format(outf)
        if osp.exists(outfname):
            os.remove(outfname)
        arglist = ['avconv',
                   '-f', 'image2',
                   '-r', '10',
                   "-i", ".video_images/frame_%05d.png",
                   "-f", "mp4",
                   "-bf", "1",
                   "-r", "30",
                   "{}.mp4".format(outf)]
        subprocess.call(arglist)


    def initialize_from_workspace(self, workspace):
        pass

    def draw(self, objList, t, transparency = 0.7):
        """
        This function draws all the objects from the objList at timestep t

        objList : list of parameters of type Object
        t       : timestep of the trajectory
        """
        for obj in objList:
            self._draw_rave_body(obj, obj.name, t, transparency)

    def draw_traj(self, objList, t_range):
        """
        This function draws the trajectory of objects from the objList

        objList : list of parameters of type Object
        t_range : range of timesteps to draw
        """
        for t in t_range:
            for obj in objList:
                name = "{0}-{1}".format(obj.name, t)
                self._draw_rave_body(obj, name, t)

    def _draw_rave_body(self, obj, name, t, transparency = 0.7):
        rotation = np.array([[0],[0],[0]])
        assert isinstance(obj, Object)
        if name not in self.name_to_rave_body:
            self.name_to_rave_body[name] = OpenRAVEBody(self.env, name, obj.geom)
        if isinstance(obj.geom, PR2):
            self.name_to_rave_body[name].set_dof(obj.backHeight[:, t], obj.lArmPose[:, t], obj.lGripper[:, t], obj.rArmPose[:, t], obj.rGripper[:, t])
        if isinstance(obj.geom, Can) or isinstance(obj.geom, Box) or isinstance(obj.geom, Table):
            rotation = obj.rotation[:, t]
            assert not np.any(np.isnan(rotation))
        assert not np.any(np.isnan(obj.pose[:, t]))
        self.name_to_rave_body[name].set_pose(obj.pose[:, t], rotation)
        self.name_to_rave_body[name].set_transparency(transparency)

    def animate_plan(self, plan, delay=.1):
        obj_list = []
        horizon = plan.horizon
        for p in plan.params.itervalues():
            if not p.is_symbol():
                obj_list.append(p)
        for t in range(horizon):
            self.draw(obj_list, t)
            time.sleep(delay)

    def draw_plan(self, plan):
        horizon = plan.horizon
        self.draw_plan_range(plan, (0, horizon-1))

    def draw_plan_range(self, plan, (start, end)):
        obj_list = self._get_plan_obj_list(plan)
        self.draw_traj(obj_list, range(start, end+1))

    def _get_plan_obj_list(self, plan):
        obj_list = []
        for p in plan.params.itervalues():
            if not p.is_symbol():
                obj_list.append(p)
        return obj_list


    def draw_plan_ts(self, plan, t):
        obj_list = []
        horizon = plan.horizon
        for p in plan.params.itervalues():
            if not p.is_symbol():
                obj_list.append(p)
        self.draw(obj_list, t)

    def draw_cols(self, plan):
        horizon = plan.horizon
        for t in range(horizon):
            self.draw_cols_ts(plan, t)

    def draw_cols_ts(self, plan, t):
        preds = plan.get_active_preds(t)
        for p in preds:
            try:
                p.plot_cols(self.env, t)
            except AttributeError:
                ## some predicates won't define a collision
                continue
