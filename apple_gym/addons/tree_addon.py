from diy_gym.addons.addon import Addon, AddonFactory
import numpy as np
import logging
import json
import time
import pybullet as p
import pybullet_planning as pbp
from pathlib import Path
from colour import Color
from gym import spaces
from diy_gym.util.bullet import normal_force_between_bodies, quaternion_multiply
from apple_gym.util.color import gen_rand_color_between_two_of, c2rgba
import logging
logger = logging.getLogger(__file__)

    

def add_fruit(
    pos, orn, fruitStemVisualShapeId, fruitCollisionShapeId, fruitVisualShapeId
):
    link_Masses = [0.018]
    linkCollisionShapeIndices = [fruitCollisionShapeId]
    linkVisualShapeIndices = [fruitVisualShapeId]
    linkPositions = [[0.0, 0.0, -0.03]]
    linkOrientations = [[0, 0, 0, 1]]
    linkInertialFramePositions = [[0, 0, 0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    jointTypes = [p.JOINT_FIXED]
    axis = [[1, 0, 0]]
    baseOrientation = [0, 0, 0, 1]
    fruitId = p.createMultiBody(
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=fruitStemVisualShapeId,
        basePosition=pos,
        baseOrientation=orn,
        linkMasses=link_Masses,
        linkCollisionShapeIndices=linkCollisionShapeIndices,
        linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions,
        linkOrientations=linkOrientations,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkParentIndices=indices,
        linkJointTypes=jointTypes,
        linkJointAxis=axis,
    )

    for i in [-1] + list(range(p.getNumJoints(fruitId))):
        p.changeDynamics(fruitId, i, activationState=p.ACTIVATION_STATE_SLEEP)
    return fruitId


def add_twig(pos, orn, twigCollisionShapeId, twigVisualShapeId):
    link_Masses = [0.018]
    linkCollisionShapeIndices = [twigCollisionShapeId]
    linkVisualShapeIndices = [twigVisualShapeId]
    linkPositions = [[0, 0, 0]]
    linkOrientations = [[0, 0, 0, 1]]
    linkInertialFramePositions = [[0, 0, 0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    jointTypes = [p.JOINT_SPHERICAL]
    axis = [[1, 0, 0]]
    baseOrientation = [0, 0, 0, 1]
    twigId = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=pos,
        baseOrientation=orn,
        linkMasses=link_Masses,
        linkCollisionShapeIndices=linkCollisionShapeIndices,
        linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions,
        linkOrientations=linkOrientations,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkParentIndices=indices,
        linkJointTypes=jointTypes,
        linkJointAxis=axis,
    )

    for i in [-1] + list(range(p.getNumJoints(twigId))):
        p.changeDynamics(twigId, i, activationState=p.ACTIVATION_STATE_SLEEP)
    return twigId


class FruitTreeAddon(Addon):
    """
    Adds fruit to chosen frame on parent. Also observed force applied, grabbing reward, and detaching when gripped

    parent: the bush

    config:
    - target_model_filter: any models with this string in names, will be searched for matching frames
    - target_frame_filter: any frames with this string in the name, will have fruit attached
    - mesh: fruit mesh to use
    - source_model: gripper
    - source_frame (optional): effector frame
    - tolerance: grip tolerance
    """

    def __init__(self, parent, config):
        super().__init__(parent, config)
        self.picking_tolerance = config.get("picking_tolerance", 0.04)
        self.fruit_force_mult = config.get("fruit_force_mult", 10)
        self.force_reward_mult = config.get("force_reward_mult", 1e-3)
        self.vary_colors = config.get("vary_colors", 2)
        self.debug = config.get("debug", 0) # will print "pick"
        self.cheat = config.get("cheat", 0) # will observe privlidges state info

        self.source_model = parent.models[config.get("source_model", "robot")]
        source_frame = config.get("source_frame", "ee_joint")
        self.source_frame_id = self.source_model.get_frame_id(source_frame)
        assert self.source_frame_id >= 0, "need to set picker joint for distanc calc"

        # for _generate_tree_pose
        self.initial_pose = (
            config.get("xyz", [0.3, 0.4, -0.2]),
            pbp.quat_from_euler(config.get("rpy", [0, 0, 0])),
        )
        self.position_range = config.get("position_range", [0.5, 0.5, 0.2])
        self.rotation_range = config.get("rotation_range", [0.0, 0.0, 1.0])

        apple_gym_dir = Path(__file__).parent.parent.parent
        self.twig_dir = Path(config.get("twig_dir", apple_gym_dir/"./data/models/twigs"))
        self.fruit_mesh = Path(
            config.get("fruit_mesh", apple_gym_dir/"./data/models/apple/apple.obj")
        )
        self.tree_model_name = config.get("tree_model_name", "tree_Hornbeam.obj")
        self.model_folders = config.get("model_folders")
        self.max_fruit = config.get("max_fruit", 200)
        self.max_twigs = config.get("max_twigs", 1000)

        # make some of the assets absolute
        self.model_folders = [apple_gym_dir/m for m in self.model_folders]

        self.observation_space = spaces.Dict(
            {
                "force": spaces.Dict(
                    {
                        # "fruit": spaces.Box(-10000, 10000, shape=(1,), dtype="float32"),
                        # "tree": spaces.Box(-10000, 10000, shape=(1,), dtype="float32"),
                    }
                )
            }
        )

        # init
        self.fruitIds = []
        self.twigIds = []
        self.tree_id = None
        self.fruitCollisionShapeId = (
            self.treeCollisionShapeId
        ) = self.twigCollisionShapeId = None
        # self.reset()

    def reset(self):
        self.reset_tree()
        self.picks = 0.0
        if self.debug:
            p.removeAllUserDebugItems()

    def remove_tree(self):
        if self.tree_id is not None:
            p.removeBody(self.tree_id)
        for twigId in self.twigIds:
            p.removeBody(twigId)
        for fruitId in self.fruitIds:
            p.removeBody(fruitId)
        self.twigIds = []
        self.fruitIds = []

        # It doesn't let me?
        # if self.treeCollisionShapeId is not None:
        #     p.removeCollisionShape(self.treeCollisionShapeId)
        # if self.twigCollisionShapeId is not None:
        #     p.removeCollisionShape(self.twigCollisionShapeId)
        # if self.fruitCollisionShapeId is not None:
        #     p.removeCollisionShape(self.fruitCollisionShapeId)

    def reset_tree(self):
        t0 = time.time()
        self.remove_tree()
        t1 = time.time()
        tree_pos, tree_orn = self._generate_tree_pose()

        model_folder = np.random.choice(self.model_folders)
        tree_model = Path(model_folder, self.tree_model_name)
        assert tree_model.is_file(), f"tree_model {tree_model} should be a file"
        assert (
            self.fruit_mesh.is_file()
        ), f"fruit_mesh {self.fruit_mesh} should be a file"
        tree_data = json.load(Path(model_folder, "twigs.json").open())
        logger.debug(f"Loading model_folder={model_folder}")

        # Adjust the height based on the heightest leaf
        tree_twig_dict = {k: v for k, v in tree_data.items() if k[1] == '_'}
        locs_all = np.concatenate([row['loc'] for row in tree_twig_dict.values()])
        max_height = locs_all[:, 2].max()
        tree_scale = 3 / max_height
        twig_scale = tree_scale / 2.0

        leaf_colors = [
            Color('#618a3d'),  # leaf
            Color('#899a30'),  # Fresh onion
            Color('#4c6e31'),  # dark
            Color('#4f7d29'),  # olive green
            Color('#598a30'),  # Fresh onion
            Color('#3A5F0B'),
            Color('#618a3d'),
            Color('#6F9940'),
            Color('#087830'),
            Color('#059033'),
        ]
        rand_leaf_color = gen_rand_color_between_two_of(leaf_colors)
        leaf_color = c2rgba(leaf_colors[0])

        # Load tree
        bark_colors = [
            Color('#585045'), # burnt bark
            Color('#5d432c'), # Walnut brown
            Color('#69594f'),  # maple bark
            Color('#716454'), # dark brown
            Color('#725c42'), # Glidden Tall Tree Bark Brown
            Color('#755f46'), #  Deep Walnut
            Color('#857e5d'),  # olive bark
            Color('#8c6258'), # Pratt & Lambert Bark Brown
            Color('#9d6f46'),  # autumn bark
        ]
        rand_bark_color = gen_rand_color_between_two_of(bark_colors)
        bark_color = c2rgba(bark_colors[0])
        self.treeVisualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(tree_model.absolute()),
            rgbaColor=rand_bark_color() if self.vary_colors else bark_color,
            meshScale=[tree_scale, tree_scale, tree_scale],
        )
        self.treeCollisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=str(tree_model.absolute()),
            meshScale=[tree_scale, tree_scale, tree_scale],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        self.tree_id = p.createMultiBody(
            baseCollisionShapeIndex=self.treeCollisionShapeId,
            baseVisualShapeIndex=self.treeVisualShapeId,
            basePosition=tree_pos,
            baseOrientation=tree_orn,
        )
        t2 = time.time()

        # Fruit and stem assets

        # Random apple color
        apple_red_colors = [
            # https://www.researchgate.net/publication/224593129_A_new_method_for_fruits_recognition_system
            Color('#800e2a'),
            Color('#d46243'),
        ]
        apple_green_colors = [
            Color('#639120'),
            Color('#92bd3e'),
        ]
        rand_apple_color = gen_rand_color_between_two_of(apple_red_colors)
        apple_color = c2rgba(apple_red_colors[0])


        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)

        self.fruitVisualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(self.fruit_mesh.absolute()),
            rgbaColor=rand_apple_color() if self.vary_colors else apple_color,
            visualFramePosition=[0, 0, -0.024 / 2],
            meshScale=[2, 2, 2],
        )
        self.fruitCollisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.024,
            collisionFramePosition=[0, 0, -0.024 / 2],
        )
        self.fruitStemVisualShapeId = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.0015,
            length=0.06,
            visualFramePosition=[0.0, 0.0, -0.03],
            rgbaColor=[0.44, 0.36, 0.25, 1.0],
        )

        # Make twigs (a small branch with leaves)
        tree_twig_items=list(tree_twig_dict.items())
        np.random.shuffle(tree_twig_items)
        for j, (name, row) in enumerate(tree_twig_items):
            typ = row["type"].replace(" ", "_").lower()
            if typ in ["dead_twigs"]:
                # these are optional overriding twigs in grove
                continue
            name = f"{name}_{typ}"
            mesh = self.twig_dir / 'tree_SmallTwig.obj' #Path(row["mesh"]).name
            mesh_collision = self.twig_dir / mesh.name.replace(".obj", "_collision.obj")
            assert mesh.is_file()
            assert mesh_collision.is_file()

            # account for tree scale, pos, orn
            rots = [pbp.quat_from_euler(r) for r in row["rot"]]
            rots = [quaternion_multiply(tree_orn, q0) for q0 in rots]

            # adjust twig loc for tree base rotation and pos
            R_tree = pbp.quaternion_matrix(tree_orn)
            loc = np.array(row["loc"]) * tree_scale
            loc = [
                pbp.tform_point(pbp.pose_from_tform(R_tree), l) + tree_pos for l in loc
            ]



            # Twig assets
            meshScale = [twig_scale, twig_scale, twig_scale]
            self.twigVisualShapeId = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=str(mesh.absolute()),
                rgbaColor=rand_leaf_color() if self.vary_colors else leaf_color,
                meshScale=meshScale,
            )
            self.twigCollisionShapeId = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=str(mesh_collision.absolute()),
                meshScale=meshScale,
            )

            logger.debug(
                f'twig_type "{name}" instances={len(loc)}, mesh="{mesh}"m model_folder={model_folder}'
            )

            ii = np.arange(len(loc))
            np.random.shuffle(ii)
            for i in ii:
                orn = rots[i]
                pos = loc[i]

                # # if we superimpose 2 joints, they might collid
                # # so we will move out the twigs a little bit from the branch
                joint_padding = 0.01
                R0 = pbp.quaternion_matrix(orn)
                pos += pbp.tform_point(pbp.pose_from_tform(R0), [joint_padding, 0, 0.0])

                # We don't want to add apples to horizontal branches, as they will overlap
                if typ == "lateral_twigs":
                    # hard to tell for lateral twigs
                    # FIXME need to also export branch angle
                    rpy = pbp.euler_from_quat(orn)
                    twig_pitch = rpy[1]
                    horizontal_branch = True
                else:
                    # apical and upwards twigs
                    # -90 would be straight up branch
                    rpy = pbp.euler_from_quat(orn)
                    twig_pitch = rpy[1]
                    horizontal_branch = np.abs(twig_pitch) < np.deg2rad(45)


                if len(self.twigIds) < self.max_twigs:
                    if self.vary_colors>1:
                        self.twigVisualShapeId = p.createVisualShape(
                            shapeType=p.GEOM_MESH,
                            fileName=str(mesh.absolute()),
                            rgbaColor=rand_leaf_color(),
                            meshScale=meshScale,
                        )
                    twigId = add_twig(
                        pos, orn, self.twigCollisionShapeId, self.twigVisualShapeId
                    )
                    self.twigIds.append(twigId)

                if horizontal_branch:
                    if np.random.rand() < 0.9:
                        if len(self.fruitIds) < self.max_fruit:
                            if self.vary_colors>1:
                                self.fruitVisualShapeId = p.createVisualShape(
                                    shapeType=p.GEOM_MESH,
                                    fileName=str(self.fruit_mesh.absolute()),
                                    rgbaColor=rand_apple_color(),
                                    visualFramePosition=[0, 0, -0.024 / 2],
                                    meshScale=[2, 2, 2],
                                )
                            fruitId = add_fruit(
                                pos,
                                orn,
                                self.fruitStemVisualShapeId,
                                self.fruitCollisionShapeId,
                                self.fruitVisualShapeId,
                            )
                            self.fruitIds.append(fruitId)
                # if typ == 'lateral_twigs':
                #     p.addUserDebugText(
                #         f'{"h" if horizontal_branch else "v"} {np.rad2deg(twig_pitch):2.2f} {typ}',
                #         pos,
                #         lifeTime=40
                #     )

        # now make everything not collide with itself
        collisionFilterGroup = 0x1
        collisionFilterMask = 0x2
        bodyIds = self.twigIds + self.fruitIds + [self.tree_id]
        for bodyId in bodyIds:
            p.setCollisionFilterGroupMask(
                bodyId, -1, collisionFilterGroup, collisionFilterMask
            )
            for i in range(p.getNumJoints(bodyId)):
                p.setCollisionFilterGroupMask(
                    bodyId, i, collisionFilterGroup, collisionFilterMask
                )

        t3 = time.time()
        # logger.debug(
        #     f"reset times {t1-t0:2.2g}s load_tree:{t2-t1:2.2g} load twigs and fruit:{t3-t2:2.2g}s"
        # )

    def _dist(self, target_body=-1):
        source_xyz = (
            p.getLinkState(self.source_model.uid, self.source_frame_id)[4]
            if self.source_frame_id >= 0
            else p.getBasePositionAndOrientation(self.source_model.uid)[0]
        )

        # target_xyz, _ = p.getBasePositionAndOrientation(target_body)
        target_xyz = p.getLinkState(target_body, 0)[4]
        return np.linalg.norm(np.array(target_xyz) - source_xyz)

    def _forces(self):
        """Get force on tree and fruit."""
        # Loop through force in all links (tree and twigs)
        tree_uids = [self.tree_id] + self.twigIds
        force_trees = sum(
            [normal_force_between_bodies(self.source_model.uid, i) for i in tree_uids]
        )

        # Loop through force in all links
        force_fruit = sum(
            [
                normal_force_between_bodies(self.source_model.uid, i)
                for i in self.fruitIds
            ]
        )

        # Get force on stems
        return dict(
            trees=force_trees,
            fruit=force_fruit,
        )

    # def _cheat():
    #     """Return privlidged information on fruit, tree, twig positions

    #     The problem is there is a variable amount of these...
    #     """
    #     [self.tree_id] + self.twigIds
    #     self.fruitIds
    #     p.getBasePositionAndOrientation(self.source_model.uid)[0]

    def observe(self):
        # observe force on fruit or tree
        obs = {
            # "forces": self._forces(),
            "picks": self.picks
        }
        # if self.cheat:
        #     obs[''] = 1
        return obs

    def remove_fruit(self, fid, remove=True):
        if remove:
            p.removeBody(fid)
        else:
            # giving it a mass, makes it not fixed, so it falls down
            p.changeDynamics(fid, -1, mass=0.018)
            # TODO move to self.pickedFruit, delete on reset
        i = self.fruitIds.index(fid)
        del self.fruitIds[i]


    def reward(self):
        # reward for distance from gripper to closest fruit
        min_dist = np.inf
        grip = 0.0
        self.picks = 0
        for fid in list(self.fruitIds):
            dist = self._dist(fid)
            if dist < min_dist:
                min_dist = dist

            # if it's gripped inside effector remove it, as if it's been vaccumed down a tube
            if dist < self.picking_tolerance:
                # also check it's facing up
                pos, orn = pbp.get_link_pose(self.source_model.uid, self.source_frame_id)
                rpy = pbp.euler_from_quat(orn)
                pitch = np.rad2deg(rpy[1])
                if np.abs(pitch) < 90:
                    self.remove_fruit(fid, remove=True)
                    logger.info(f"Picked fruit {fid} dist={dist:2.2g} pitch={pitch:2.2f} deg")
                    grip += 200.0
                    self.picks += 1.0
                    if self.debug:
                        p.addUserDebugText(f'picked', pos, lifeTime=360)
                        # p.addUserDebugLine(
                        #     lineFromXYZ=pos, 
                        #     lineToXYZ=target_pos, lineColorRGB=[0.2, 0.9, 0.2], lineWidth=1)
                else:
                    logger.warning(f"ALMOST Picked fruit {fid} dist={dist:2.2g} pitch={p:2.2f} deg")

        forces = self._forces()
        return dict(
            min_fruit_dist_reward=-min_dist/10.,
            gripping_fruit_reward=grip,
            # force_tree_reward=-np.sqrt(forces["trees"])
            # * self.force_reward_mult,  # can be 0-10143, often 50
            # force_fruit_reward=-np.sqrt(forces["fruit"])
            # * self.force_reward_mult
            # * self.fruit_force_mult,  # often 100
        )

    def _generate_tree_pose(self):
        pos = (np.random.random(3) - 0.5) * self.position_range + self.initial_pose[0]
        quat = quaternion_multiply(
            self.initial_pose[1],
            p.getQuaternionFromEuler((np.random.random(3) - 0.5) * self.rotation_range),
        )
        return (pos, quat)

    def is_terminal(self):
        return len(self.fruitIds) == 0


AddonFactory.register_addon("fruit_tree", FruitTreeAddon)
