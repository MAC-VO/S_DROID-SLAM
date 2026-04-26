import torch
import lietorch
import numpy as np

from lietorch import SE3
from .factor_graph import FactorGraph

from .cuda_timer import CudaTimer


ENABLE_TIMING = False

class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(
            video, net.update, max_factors=48, upsample=args.upsample
        )

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 20
        self.iters1 = 3
        self.iters2 = 2

        self.keyframe_removal_index = 3

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

        self.depth_window = 3

        self.motion_damping = 0.0
        if hasattr(args, "motion_damping"):
            self.motion_damping = args.motion_damping

        # vo_only: strict two-frame BA on the (L, L-1) pair only
        self.vo_only = bool(getattr(args, "vo_only", False))

    def _init_next_state(self):
        # set pose / depth for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]

        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - 3 : self.t1 - 1], 0.5
        )

        # damped linear velocity model
        if self.motion_damping >= 0:
            poses = SE3(self.video.poses)
            vel = (poses[self.t1 - 1] * poses[self.t1 - 2].inv()).log()
            damped_vel = self.motion_damping * vel
            next_pose = SE3.exp(damped_vel) * poses[self.t1 - 1]
            self.video.poses[self.t1] = next_pose.data

    def _add_pairwise_factors(self, latest: int) -> None:
        """Add the bidirectional (latest, latest-1) factor pair (+ stereo self-edge)."""
        ii_list = [latest, latest - 1]
        jj_list = [latest - 1, latest]
        # Stereo self-edge gives the known-baseline depth constraint at `latest`.
        if self.video.stereo:
            ii_list.append(latest)
            jj_list.append(latest)
        device = self.graph.device
        ii = torch.as_tensor(ii_list, dtype=torch.long, device=device)
        jj = torch.as_tensor(jj_list, dtype=torch.long, device=device)
        self.graph.add_factors(ii, jj)

    def _update_vo_only(self):
        """Strict two-frame BA between L and L-1 (vo_only mode)."""
        self.count += 1
        self.t1 += 1
        L = self.t1 - 1

        # Drop everything from the previous pair so the linear system only sees L,L-1.
        if self.graph.ii.numel() > 0:
            mask = torch.ones_like(self.graph.ii, dtype=torch.bool)
            self.graph.rm_factors(mask, store=False)

        self._add_pairwise_factors(L)

        # Sensor-disparity injection at the new keyframe (matches the default branch).
        self.video.disps[L] = torch.where(
            self.video.disps_sens[L] > 0,
            self.video.disps_sens[L],
            self.video.disps[L],
        )

        # T_{L-1} is the gauge (t0=L), only T_L and disps at {L-1, L} are updated.
        for _ in range(self.iters1 + self.iters2):
            self.graph.update(t0=L, t1=L + 1, use_inactive=False)

        # Seed pose/disparity for the next slot (mirrors the default path).
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - self.depth_window - 1 : self.t1 - 1], 0.7
        )

        self.video.dirty[self.graph.ii.min() : self.t1] = True

    def _update(self):
        """add edges, perform update"""

        if self.vo_only:
            self._update_vo_only()
            return

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(
            self.t1 - 5,
            max(self.t1 - self.frontend_window, 0),
            rad=self.frontend_radius,
            nms=self.frontend_nms,
            thresh=self.frontend_thresh,
            beta=self.beta,
            remove=True,
        )

        self.video.disps[self.t1 - 1] = torch.where(
            self.video.disps_sens[self.t1 - 1] > 0,
            self.video.disps_sens[self.t1 - 1],
            self.video.disps[self.t1 - 1],
        )

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
        d = self.video.distance(
            [self.t1 - 4], [self.t1 - 2], beta=self.beta, bidirectional=True
        )

        if d.item() < 2 * self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 3)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)


        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - self.depth_window - 1 : self.t1 - 1], 0.7
        )

        # update visualization
        self.video.dirty[self.graph.ii.min() : self.t1] = True

    def _initialize_vo_only(self):
        """Bootstrap warmup keyframes via sequential strict pairwise BA."""
        self.t0 = 0
        self.t1 = self.video.counter.value

        for i in range(1, self.t1):
            # Constant-pose seed: Frame i starts where i-1 currently sits.
            self.video.poses[i] = self.video.poses[i - 1].clone()

            if self.graph.ii.numel() > 0:
                mask = torch.ones_like(self.graph.ii, dtype=torch.bool)
                self.graph.rm_factors(mask, store=False)

            self._add_pairwise_factors(i)

            self.video.disps[i] = torch.where(
                self.video.disps_sens[i] > 0,
                self.video.disps_sens[i],
                self.video.disps[i],
            )

            for _ in range(self.iters1 + self.iters2):
                self.graph.update(t0=i, t1=i + 1, use_inactive=False)

    def _initialize(self):
        """initialize the SLAM system"""

        if self.vo_only:
            self._initialize_vo_only()
            self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
            self.video.disps[self.t1] = self.video.disps[self.t1 - 4 : self.t1].mean()

            self.is_initialized = True
            self.last_pose = self.video.poses[self.t1 - 1].clone()
            self.last_disp = self.video.disps[self.t1 - 1].clone()
            self.last_time = self.video.tstamp[self.t1 - 1].clone()

            with self.video.get_lock():
                self.video.ready.value = 1
                self.video.dirty[: self.t1] = True
            return

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(
            0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False
        )

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1 - 4 : self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1 - 1].clone()
        self.last_disp = self.video.disps[self.t1 - 1].clone()
        self.last_time = self.video.tstamp[self.t1 - 1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[: self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

    def __call__(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self._initialize()
            self._init_next_state()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self._update()
            self._init_next_state()
