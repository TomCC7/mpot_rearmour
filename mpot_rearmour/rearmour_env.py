#!/usr/bin/env python3
import torch
from submodules.rearmour.environments.robots import robot as robot2
from submodules.rearmour.reachability.joint_reachable_set.process_jrs_trig import (
    process_batch_JRS_trig,
)
from submodules.rearmour.planning.rearmour_nlp_problem import (
    RearmourNlpProblem,
    OfflineRearmourFoConstraints,
)


class RearmourEnv:
    """Env to be used for mpot planning"""

    def __init__(
        self,
        robot: robot2.ZonoArmRobot,
        obs: tuple[torch.Tensor, torch.Tensor],
        name="Rearmour",
        dim: int = 3,
        zono_order: int = 2,
        max_combs: int = 200,
        include_end_effector: bool = False,
    ):
        # used by mpot task
        self.dim = dim
        self.limits = torch.ones([self.dim, 2])
        self.limits[:, 0] = -1

        # rearmour...
        self.robot = robot
        self.JRS_tensor = preload_batch_JRS_trig(dtype=self.dtype, device=self.device)
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.combs = self._generate_combinations_upto(max_combs)
        self.include_end_effector = include_end_effector
        # robots
        self.dof = robot.dof
        self.joint_axis = robot.joint_axis
        self.pos_lim = robot.np.pos_lim
        self.vel_lim = robot.np.vel_lim
        # self.eff_lim = np.array(eff_lim) # Unused for now
        self.continuous_joints = robot.np.continuous_joints
        self.pos_lim_mask = robot.np.pos_lim_mask
        self.g_ka = (
            np.ones((self.dof), dtype=self.np_dtype) * np.pi / 24
        )  # Hardcoded because it's preloaded...

    def build_occupancy_map(self, *args):
        raise NotImplementedError("rearmour env doesn't support occupancy map")

    def get_df_obj_list(self):
        pass

    def sdf(trajs, q_pos, q_vel, H_positions) -> float:
        # joint reachable set
        _, JRS_R = process_batch_JRS_trig(
            self.JRS_tensor,
            torch.as_tensor(qpos, dtype=self.dtype, device=self.device),
            torch.as_tensor(qvel, dtype=self.dtype, device=self.device),
            self.joint_axis,
        )

        # Create obs zonotopes
        obs_Z = torch.cat(
            (
                torch.as_tensor(obs[0], dtype=self.dtype, device=self.device).unsqueeze(
                    -2
                ),
                torch.diag_embed(
                    torch.as_tensor(obs[1], dtype=self.dtype, device=self.device)
                )
                / 2.0,
            ),
            dim=-2,
        )
        obs_zono = batchZonotope(obs_Z)

        # Compute FO
        FO_constraint, FO_times = self._prepare_FO_constraints(JRS_R, obs_zono)

    def _generate_combinations_upto(self, max_combs):
        return [
            torch.combinations(torch.arange(i, device=self.device), 2)
            for i in range(max_combs + 1)
        ]

    def _prepare_FO_constraints(
        self,
        JRS_R: batchMatPolyZonotope,
        obs_zono: batchZonotope,
    ):
        # constant
        n_obs = obs_zono.batch_shape[0]

        ### get the forward occupancy
        FO_gen_time = time.perf_counter()
        FO_links, _ = forward_occupancy(JRS_R, self.robot, self.zono_order)
        # let's assume we can ignore the base link and convert to a list of pz's
        # end effector link is a thing???? (n actuated_joints = 7, n links = 8)
        FO_links = list(FO_links.values())[1:]
        if not self.include_end_effector:
            FO_links = FO_links[:-1]
        # two more constants
        n_links = len(FO_links)
        n_frs_timesteps = FO_links[0].batch_shape[0]

        ### begin generating constraints
        constraint_gen_time = time.perf_counter()
        out_g_ka = self.g_ka
        out_FO_links = np.empty((n_links), dtype=object)

        # Buffer obstacles with FO independent generators
        dimension = 3
        obs_Z = obs_zono.Z
        max_num_generators_for_obs = obs_zono.Z.shape[-2] - 1
        buffered_obstacle_Zs = (
            obs_Z.unsqueeze(1)
            .repeat(1, n_links * n_frs_timesteps, 1, 1)
            .view(
                n_obs * n_links * n_frs_timesteps,
                max_num_generators_for_obs + 1,
                dimension,
            )
        )
        # import pdb; pdb.set_trace()
        num_FO_independent_generators = FO_links[0].Grest.shape[
            -2
        ]  # TODO: this will depende on the zono order chosen during forward occupancy
        FO_independent_generators = torch.zeros(
            n_frs_timesteps,
            n_links,
            num_FO_independent_generators,
            dimension,
            device=self.device,
        )
        for idx, FO_link_zono in enumerate(FO_links):
            FO_independent_generators[:, idx] = FO_links[idx].Grest
            out_FO_links[idx] = batchPolyZonotope(
                FO_link_zono.Z[..., : FO_link_zono.n_dep_gens + 1, :],
                FO_link_zono.n_dep_gens,
                FO_link_zono.expMat,
                FO_link_zono.id,
            ).cpu()
        FO_independent_generators = FO_independent_generators.view(
            n_links * n_frs_timesteps, num_FO_independent_generators, dimension
        )
        FO_independent_generators = FO_independent_generators.repeat(n_obs, 1, 1)
        buffered_obstacle_Zs = torch.cat(
            (buffered_obstacle_Zs, FO_independent_generators), dim=1
        )

        # 4. Reduce buffered obstacle zonotope
        if (
            self.num_desired_generators is not None
            and self.num_desired_generators
            < max_num_generators_for_obs + num_FO_independent_generators
        ):
            buffered_obstacle_generators = buffered_obstacle_Zs[:, 1:, :]
            order = self.num_desired_generators / 3
            buffered_obstacle_generators = batch_reduce_PCA(
                buffered_obstacle_generators, order=order
            )
            buffered_obstacle_Zs = torch.cat(
                (buffered_obstacle_Zs[:, 0:1, :], buffered_obstacle_generators), dim=1
            )

        # 5. Compute hyperplanes from buffered obstacles generators
        # TODO: this step might be able to be optimized
        hyperplanes_A, hyperplanes_b = batchZonotope(buffered_obstacle_Zs).polytope(
            self.combs
        )
        hyperplanes_b = hyperplanes_b.unsqueeze(-1)
        hyperplanes_A = hyperplanes_A.to(self.device)
        hyperplanes_b = hyperplanes_b.to(self.device)

        # 6. Compute vertices from buffered obstacles generators
        # self.v1, self.v2 = compute_edges_from_convexhulls(buffered_obstacle_Zs[:,0:1,:], buffered_obstacle_Zs[:,1:,:])
        # v1 = v1.to(self.device)
        # v2 = v2.to(self.device)
        torch.cuda.synchronize()
        vertices_time = time.perf_counter()
        v1, v2 = compute_edges_from_generators(
            buffered_obstacle_Zs[:, 0:1, :],
            buffered_obstacle_Zs[:, 1:, :],
            hyperplanes_A,
            hyperplanes_b,
        )
        torch.cuda.synchronize()
        ### REARMOUR SETUP FINISHES!

        # out_n_obs_in_frs = int(obs_in_reach_idx.sum())
        final_time = time.perf_counter()
        out_times = {
            "FO_gen": constraint_gen_time - FO_gen_time,
            "constraint_gen": vertices_time - constraint_gen_time,
            "vertices": final_time - vertices_time,
        }
        Rearmour_constraint = OfflineRearmourFoConstraints(dtype=self.dtype)
        Rearmour_constraint.set_params(
            out_FO_links,
            hyperplanes_A,
            hyperplanes_b,
            out_g_ka,
            n_obs,
            self.dof,
            v1,
            v2,
            self.distance_net,
        )
        return Rearmour_constraint, out_times


class REARMOUR_3D_planner:
    def __init__(
        self,
        robot: ZonoArmRobot,
        zono_order: int = 2,  # this appears to have been 40 before but it was ignored for 2
        max_combs: int = 200,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu"),
        include_end_effector: bool = False,
        num_desired_generators=None,
    ):
        self.dtype, self.device = dtype, device
        self.np_dtype = torch.empty(0, dtype=dtype).numpy().dtype

        self.robot = robot
        self.PI = torch.tensor(torch.pi, dtype=self.dtype, device=self.device)
        self.JRS_tensor = preload_batch_JRS_trig(dtype=self.dtype, device=self.device)

        self.zono_order = zono_order
        self.max_combs = max_combs
        self.combs = self._generate_combinations_upto(max_combs)
        self.include_end_effector = include_end_effector

        self.num_desired_generators = num_desired_generators
        self.distance_net = DistanceGradientNet()
        self.distance_buffer = 0.0

        self._setup_robot(robot)

        # Prepare the nlp
        self.g_ka = (
            np.ones((self.dof), dtype=self.np_dtype) * np.pi / 24
        )  # Hardcoded because it's preloaded...
        self.nlp_problem_obj = RearmourNlpProblem(
            self.dof,
            self.g_ka,
            self.pos_lim,
            self.vel_lim,
            self.continuous_joints,
            self.pos_lim_mask,
            self.dtype,
            T_PLAN,
            T_FULL,
        )

        # self.wrap_env(env)
        # self.n_timesteps = 100
        # self.joint_speed_limit = torch.vstack((torch.pi*torch.ones(n_links),-torch.pi*torch.ones(n_links)))

    def _setup_robot(self, robot: ZonoArmRobot):
        self.dof = robot.dof
        self.joint_axis = robot.joint_axis
        self.pos_lim = robot.np.pos_lim
        self.vel_lim = robot.np.vel_lim
        # self.eff_lim = np.array(eff_lim) # Unused for now
        self.continuous_joints = robot.np.continuous_joints
        self.pos_lim_mask = robot.np.pos_lim_mask
        pass

    def _generate_combinations_upto(self, max_combs):
        return [
            torch.combinations(torch.arange(i, device=self.device), 2)
            for i in range(max_combs + 1)
        ]

    def _prepare_FO_constraints(
        self,
        JRS_R: batchMatPolyZonotope,
        obs_zono: batchZonotope,
    ):
        # constant
        n_obs = obs_zono.batch_shape[0]

        ### get the forward occupancy
        FO_gen_time = time.perf_counter()
        FO_links, _ = forward_occupancy(JRS_R, self.robot, self.zono_order)
        # let's assume we can ignore the base link and convert to a list of pz's
        # end effector link is a thing???? (n actuated_joints = 7, n links = 8)
        FO_links = list(FO_links.values())[1:]
        if not self.include_end_effector:
            FO_links = FO_links[:-1]
        # two more constants
        n_links = len(FO_links)
        n_frs_timesteps = FO_links[0].batch_shape[0]

        ### begin generating constraints
        constraint_gen_time = time.perf_counter()
        out_g_ka = self.g_ka
        out_FO_links = np.empty((n_links), dtype=object)

        # Buffer obstacles with FO independent generators
        dimension = 3
        obs_Z = obs_zono.Z
        max_num_generators_for_obs = obs_zono.Z.shape[-2] - 1
        buffered_obstacle_Zs = (
            obs_Z.unsqueeze(1)
            .repeat(1, n_links * n_frs_timesteps, 1, 1)
            .view(
                n_obs * n_links * n_frs_timesteps,
                max_num_generators_for_obs + 1,
                dimension,
            )
        )
        # import pdb; pdb.set_trace()
        num_FO_independent_generators = FO_links[0].Grest.shape[
            -2
        ]  # TODO: this will depende on the zono order chosen during forward occupancy
        FO_independent_generators = torch.zeros(
            n_frs_timesteps,
            n_links,
            num_FO_independent_generators,
            dimension,
            device=self.device,
        )
        for idx, FO_link_zono in enumerate(FO_links):
            FO_independent_generators[:, idx] = FO_links[idx].Grest
            out_FO_links[idx] = batchPolyZonotope(
                FO_link_zono.Z[..., : FO_link_zono.n_dep_gens + 1, :],
                FO_link_zono.n_dep_gens,
                FO_link_zono.expMat,
                FO_link_zono.id,
            ).cpu()
        FO_independent_generators = FO_independent_generators.view(
            n_links * n_frs_timesteps, num_FO_independent_generators, dimension
        )
        FO_independent_generators = FO_independent_generators.repeat(n_obs, 1, 1)
        buffered_obstacle_Zs = torch.cat(
            (buffered_obstacle_Zs, FO_independent_generators), dim=1
        )

        # 4. Reduce buffered obstacle zonotope
        if (
            self.num_desired_generators is not None
            and self.num_desired_generators
            < max_num_generators_for_obs + num_FO_independent_generators
        ):
            buffered_obstacle_generators = buffered_obstacle_Zs[:, 1:, :]
            order = self.num_desired_generators / 3
            buffered_obstacle_generators = batch_reduce_PCA(
                buffered_obstacle_generators, order=order
            )
            buffered_obstacle_Zs = torch.cat(
                (buffered_obstacle_Zs[:, 0:1, :], buffered_obstacle_generators), dim=1
            )

        # 5. Compute hyperplanes from buffered obstacles generators
        # TODO: this step might be able to be optimized
        hyperplanes_A, hyperplanes_b = batchZonotope(buffered_obstacle_Zs).polytope(
            self.combs
        )
        hyperplanes_b = hyperplanes_b.unsqueeze(-1)
        hyperplanes_A = hyperplanes_A.to(self.device)
        hyperplanes_b = hyperplanes_b.to(self.device)

        # 6. Compute vertices from buffered obstacles generators
        # self.v1, self.v2 = compute_edges_from_convexhulls(buffered_obstacle_Zs[:,0:1,:], buffered_obstacle_Zs[:,1:,:])
        # v1 = v1.to(self.device)
        # v2 = v2.to(self.device)
        torch.cuda.synchronize()
        vertices_time = time.perf_counter()
        v1, v2 = compute_edges_from_generators(
            buffered_obstacle_Zs[:, 0:1, :],
            buffered_obstacle_Zs[:, 1:, :],
            hyperplanes_A,
            hyperplanes_b,
        )
        torch.cuda.synchronize()
        ### REARMOUR SETUP FINISHES!

        # out_n_obs_in_frs = int(obs_in_reach_idx.sum())
        final_time = time.perf_counter()
        out_times = {
            "FO_gen": constraint_gen_time - FO_gen_time,
            "constraint_gen": vertices_time - constraint_gen_time,
            "vertices": final_time - vertices_time,
        }
        Rearmour_constraint = OfflineRearmourFoConstraints(dtype=self.dtype)
        Rearmour_constraint.set_params(
            out_FO_links,
            hyperplanes_A,
            hyperplanes_b,
            out_g_ka,
            n_obs,
            self.dof,
            v1,
            v2,
            self.distance_net,
        )
        return Rearmour_constraint, out_times

    def trajopt(self, qpos, qvel, qgoal, ka_0, FO_constraint):
        # Moved to another file
        self.nlp_problem_obj.reset(qpos, qvel, qgoal, FO_constraint)
        n_constraints = self.nlp_problem_obj.M
        n_obs_constraint = FO_constraint.M

        nlp = cyipopt.Problem(
            n=self.dof,
            m=n_constraints,
            problem_obj=self.nlp_problem_obj,
            lb=[-1] * self.dof,
            ub=[1] * self.dof,
            cl=[-1e20] * (n_constraints - n_obs_constraint)
            + [self.distance_buffer] * n_obs_constraint,
            cu=[-1e-6] * (n_constraints - n_obs_constraint) + [1e20] * n_obs_constraint,
        )

        # nlp.add_option('hessian_approximation', 'exact')
        nlp.add_option("sb", "yes")  # Silent Banner
        nlp.add_option("print_level", 0)
        nlp.add_option("tol", 1e-3)

        if ka_0 is None:
            ka_0 = np.zeros(self.dof, dtype=np.float32)
        k_opt, self.info = nlp.solve(ka_0)
        self.final_cost = self.info["obj_val"] if self.info["status"] == 0 else None
        return (
            FO_constraint.g_ka * k_opt,
            self.info["status"],
            self.nlp_problem_obj.constraint_times,
        )

    def plan(self, qpos, qvel, qgoal, obs, ka_0=None):
        # prepare the JRS
        JRS_process_time = time.perf_counter()
        _, JRS_R = process_batch_JRS_trig(
            self.JRS_tensor,
            torch.as_tensor(qpos, dtype=self.dtype, device=self.device),
            torch.as_tensor(qvel, dtype=self.dtype, device=self.device),
            self.joint_axis,
        )
        JRS_process_time = time.perf_counter() - JRS_process_time

        # Create obs zonotopes
        obs_Z = torch.cat(
            (
                torch.as_tensor(obs[0], dtype=self.dtype, device=self.device).unsqueeze(
                    -2
                ),
                torch.diag_embed(
                    torch.as_tensor(obs[1], dtype=self.dtype, device=self.device)
                )
                / 2.0,
            ),
            dim=-2,
        )
        obs_zono = batchZonotope(obs_Z)

        # Compute FO
        FO_constraint, FO_times = self._prepare_FO_constraints(JRS_R, obs_zono)

        # preproc_time, FO_gen_time, constraint_time = self.prepare_constraints2(env.qpos,env.qvel,env.obs_zonos)

        trajopt_time = time.perf_counter()
        k_opt, flag, constraint_times = self.trajopt(
            qpos, qvel, qgoal, ka_0, FO_constraint
        )
        trajopt_time = time.perf_counter() - trajopt_time

        stats = {
            "cost": self.final_cost,
            "nlp": trajopt_time,
            "constraint_gen": FO_times["constraint_gen"],
            "FO_gen": FO_times["FO_gen"],
            "JRS_process_time": JRS_process_time,
            "constraint_times": constraint_times,
            "vertices_time": FO_times["vertices"],
        }
        return k_opt, flag, stats
