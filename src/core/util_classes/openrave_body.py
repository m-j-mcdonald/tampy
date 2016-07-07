import numpy as np
from errors_exceptions import OpenRAVEException
from openravepy import quatFromAxisAngle, matrixFromPose, poseFromMatrix, \
axisAngleFromRotationMatrix, KinBody, GeometryType, RaveCreateRobot, \
RaveCreateKinBody, TriMesh
from core.util_classes.circle import Circle, BlueCircle, RedCircle, GreenCircle
from core.util_classes.obstacle import Obstacle


class OpenRAVEBody(object):
    def __init__(self, env, name, geom):
        self.name = name
        self._env = env
        self._geom = geom
        if isinstance(geom, Circle):
            self._add_circle(geom)
        elif isinstance(geom, Obstacle):
            self._add_obstacle()
        else:
            raise OpenRAVEException("Geometry not supported for %s for OpenRAVEBody"%geom)


    def _add_circle(self, geom):
        color = None
        if isinstance(geom, BlueCircle):
            color = [0, 0, 1]
        elif isinstance(geom, GreenCircle):
            color = [0, 1, 0]
        elif isinstance(geom, RedCircle):
            color = [1, 0, 0]
        self.env_body = self.create_cylinder(self._env, self.name, np.eye(4),
                [geom.radius, 2], color)
        self._env.AddRobot(self.env_body)

    def _add_obstacle(self):
        obstacles = np.matrix('-0.576036866359447, 0.918128654970760, 1;\
                        -0.806451612903226,-1.07017543859649, 1;\
                        1.01843317972350,-0.988304093567252, 1;\
                        0.640552995391705,0.906432748538011, 1;\
                        -0.576036866359447, 0.918128654970760, -1;\
                        -0.806451612903226,-1.07017543859649, -1;\
                        1.01843317972350,-0.988304093567252, -1;\
                        0.640552995391705,0.906432748538011, -1')

        body = RaveCreateKinBody(self._env, '')
        vertices = np.array(obstacles)
        indices = np.array([[0, 1, 2], [2, 3, 0], [4, 5, 6], [6, 7, 4], [0, 4, 5],
                            [0, 1, 5], [1, 2, 5], [5, 6, 2], [2, 3, 6], [6, 7, 3],
                            [0, 3, 7], [0, 4, 7]])
        body.InitFromTrimesh(trimesh=TriMesh(vertices, indices), draw=True)
        body.SetName(self.name)
        for link in body.GetLinks():
            for geom in link.GetGeometries():
                geom.SetDiffuseColor((.9, .9, .9))
        self.env_body = body
        self._env.AddKinBody(body)

    def set_pose(self, x):
        trans = None
        if isinstance(self._geom, Circle):
            trans = OpenRAVEBody.base_pose_2D_to_mat(x)
        elif isinstance(self._geom, Obstacle):
            trans = OpenRAVEBody.base_pose_2D_to_mat(x)
        self.env_body.SetTransform(trans)

    @staticmethod
    def create_cylinder(env, body_name, t, dims, color=[0, 1, 1]):
        infocylinder = KinBody.GeometryInfo()
        infocylinder._type = GeometryType.Cylinder
        infocylinder._vGeomData = dims
        infocylinder._bVisible = True
        infocylinder._vDiffuseColor = color
        # infocylinder._t[2, 3] = dims[1] / 2

        # cylinder = RaveCreateKinBody(env, '')
        cylinder = RaveCreateRobot(env, '')
        cylinder.InitFromGeometries([infocylinder])
        cylinder.SetName(body_name)
        cylinder.SetTransform(t)
        return cylinder

    @staticmethod
    def base_pose_2D_to_mat(pose):
        # x, y = pose
        assert len(pose) == 2
        x = pose[0]
        y = pose[1]
        rot = 0
        q = quatFromAxisAngle((0, 0, rot)).tolist()
        pos = [x, y, 0]
        # pos = np.vstack((x,y,np.zeros(1)))
        matrix = matrixFromPose(q + pos)
        return matrix

    @staticmethod
    def mat_to_base_pose_2D(mat):
        pose = poseFromMatrix(mat)
        x = pose[4]
        y = pose[5]
        return np.array([x,y])

    @staticmethod
    def base_pose_to_mat(pose):
        # x, y, rot = pose
        assert len(pose) == 3
        x = pose[0]
        y = pose[1]
        rot = pose[2]
        q = quatFromAxisAngle((0, 0, rot)).tolist()
        pos = [x, y, 0]
        # pos = np.vstack((x,y,np.zeros(1)))
        matrix = matrixFromPose(q + pos)
        return matrix

    @staticmethod
    def mat_to_base_pose(mat):
        pose = poseFromMatrix(mat)
        x = pose[4]
        y = pose[5]
        rot = axisAngleFromRotationMatrix(mat)[2]
        return np.array([x,y,rot])