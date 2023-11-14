#!/usr/bin/env python3
import torch
from matplotlib import pyplot as plt
import numpy as np

from mpot.ot.problem import Epsilon
from mpot.ot.sinkhorn import Sinkhorn
from mpot.planner import MPOT
from mpot.costs import CostGPHolonomic, CostField, CostComposite
from mpot_rearmour.rearmour_env import RearmourEnv

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from torch_robotics.torch_utils.torch_timer import TimerCUDA


class MPOTRearmourTask:
    def __init__(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        obs: torch.Tensor,
        exp_name: str,
        RobType: RobotBase,
        seed: int = 0,
        tensor_args=None,
    ):
        """Initialize the task for MPOT

        Args:
            start_state (torch.Tensor): start state of the robot shape (dof, )
            end_states (torch.Tensor): end states of the robot, shape (dof, )
        """
        self.start_state: torch.Tensor = start_state
        self.goal_state: torch.Tensor = goal_state
        self.env: RearmourEnv = RearmourEnv(obs)
        self.robot: RobotBase = RobType(tensor_args=tensor_args)
        self.exp_name: str = exp_name
        self.tensor_args = tensor_args
        self.seed = seed

        assert len(start_state.shape) == 1
        assert len(goal_state.shape) == 1

        self.task = PlanningTask(
            env=self.env,
            robot=self.robot,
            ws_limits=torch.tensor(
                [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args
            ),
            obstacle_cutoff_margin=0.03,
            tensor_args=tensor_args,
        )

    def run(self):
        """Run the task"""
        robot = self.robot
        task = self.task
        env = self.env
        start_state = self.start_state
        goal_state = self.goal_state
        tensor_args = self.tensor_args
        seed = self.seed

        # -------------------------------- Params ---------------------------------
        start_state = torch.concatenate((start_state, torch.zeros_like(start_state)))
        goal_state = torch.concatenate((goal_state, torch.zeros_like(goal_state)))
        multi_goal_states = goal_state.unsqueeze(0)

        # Construct planner
        duration = 5  # sec
        traj_len = 64
        dt = duration / traj_len
        num_particles_per_goal = 10  # number of plans per goal  # NOTE: if memory is not enough, reduce this number

        # NOTE: these parameters are tuned for this environment
        step_radius = 0.03
        probe_radius = 0.08  # probe radius >= step radius

        # NOTE: changing polytope may require tuning again
        # NOTE: cube in this case could lead to memory insufficiency, depending how many plans are optimized
        polytope = "cube"  # 'simplex' | 'orthoplex' | 'cube';

        epsilon = 0.02
        ent_epsilon = Epsilon(1e-2)
        num_probe = 3  # number of probes points for each polytope vertices
        # panda joint limits
        q_max = torch.tensor(
            [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], **tensor_args
        )
        q_min = torch.tensor(
            [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
            **tensor_args,
        )
        pos_limits = torch.stack([q_min, q_max], dim=1)
        vel_limits = [-5, 5]
        w_coll = 1e-1  # for tuning the obstacle cost
        w_smooth = (
            1e-7  # for tuning the GP cost: error = w_smooth * || Phi x(t) - x(1+1) ||^2
        )
        sigma_gp = 0.01  # for tuning the GP cost: Q_c = sigma_gp^2 * I
        sigma_gp_init = (
            0.5  # for controlling the initial GP variance: Q0_c = sigma_gp_init^2 * I
        )
        max_inner_iters = 100  # max inner iterations for Sinkhorn-Knopp
        max_outer_iters = 100  # max outer iterations for MPOT

        # --------------------------------- Cost function ---------------------------------

        cost_func_list = []
        weights_cost_l = []
        for collision_field in task.get_collision_fields():
            cost_func_list.append(
                CostField(
                    robot,
                    traj_len,
                    field=collision_field,
                    sigma_coll=1.0,
                    tensor_args=tensor_args,
                )
            )
            weights_cost_l.append(w_coll)
        cost_gp = CostGPHolonomic(
            robot,
            traj_len,
            dt,
            sigma_gp,
            [0, 1],
            weight=w_smooth,
            tensor_args=tensor_args,
        )
        cost_func_list.append(cost_gp)
        weights_cost_l.append(w_smooth)
        cost = CostComposite(
            robot,
            traj_len,
            cost_func_list,
            weights_cost_l=weights_cost_l,
            tensor_args=tensor_args,
        )

        # --------------------------------- MPOT Init ---------------------------------

        linear_ot_solver = Sinkhorn(
            threshold=1e-3,
            inner_iterations=1,
            max_iterations=max_inner_iters,
        )
        ss_params = dict(
            epsilon=epsilon,
            ent_epsilon=ent_epsilon,
            step_radius=step_radius,
            probe_radius=probe_radius,
            num_probe=num_probe,
            min_iterations=5,
            max_iterations=max_outer_iters,
            threshold=2e-3,
            store_history=True,
            tensor_args=tensor_args,
        )

        mpot_params = dict(
            objective_fn=cost,
            linear_ot_solver=linear_ot_solver,
            ss_params=ss_params,
            dim=7,
            traj_len=traj_len,
            num_particles_per_goal=num_particles_per_goal,
            dt=dt,
            start_state=start_state,
            multi_goal_states=multi_goal_states,
            pos_limits=pos_limits,
            vel_limits=vel_limits,
            polytope=polytope,
            fixed_goal=True,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=sigma_gp_init,
            seed=seed,
            tensor_args=tensor_args,
        )
        planner = MPOT(**mpot_params)

        # Optimize
        with TimerCUDA() as t:
            trajs, optim_state, opt_iters = planner.optimize()
        sinkhorn_iters = optim_state.linear_convergence[:opt_iters]
        print(
            f"Optimization finished at {opt_iters}! Optimization time: {t.elapsed:.3f} sec"
        )
        print(
            f"Average Sinkhorn Iterations: {sinkhorn_iters.mean():.2f}, min: {sinkhorn_iters.min():.2f}, max: {sinkhorn_iters.max():.2f}"
        )

        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

        # -------------------------------- Visualize ---------------------------------
        planner_visualizer = PlanningVisualizer(task=task, planner=planner)

        traj_history = optim_state.X_history[:opt_iters]
        traj_history = traj_history.view(opt_iters, -1, traj_len, 14)  # 7 + 7
        pos_trajs_iters = robot.get_position(traj_history)
        trajs = trajs.flatten(0, 1)
        trajs_coll, trajs_free = task.get_trajs_collision_and_free(trajs)
        if trajs_coll is None:
            trajs_coll = []
        if trajs_free is None:
            trajs_free = []

        print(f"free/collision: {len(trajs_free)}/{len(trajs_coll)}")

        planner_visualizer.animate_opt_iters_joint_space_state(
            trajs=traj_history,
            pos_start_state=start_state,
            pos_goal_state=goal_state,
            vel_start_state=torch.zeros_like(start_state),
            vel_goal_state=torch.zeros_like(goal_state),
            video_filepath=f"results/{self.exp_name}-joint-space-opt-iters.mp4",
            n_frames=max((2, opt_iters // 2)),
            anim_time=5,
        )

        if len(trajs_free) != 0:
            planner_visualizer.animate_robot_trajectories(
                trajs=trajs_free,
                start_state=start_state,
                goal_state=goal_state,
                plot_trajs=False,
                draw_links_spheres=False,
                video_filepath=f"results/{self.exp_name}-robot-traj.mp4",
                # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
                n_frames=trajs_free.shape[-2],
                anim_time=duration,
            )

        if len(trajs_free) == 0 and len(trajs_coll) != 0:
            planner_visualizer.animate_robot_trajectories(
                trajs=trajs_coll,
                start_state=start_state,
                goal_state=goal_state,
                plot_trajs=False,
                draw_links_spheres=False,
                video_filepath=f"results/{self.exp_name}-robot-traj_coll.mp4",
                # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
                n_frames=trajs_coll.shape[-2],
                anim_time=duration,
            )

        plt.show()
