import numpy as np
import torch
from gs_model import BasicPointCloud, GaussianModel
from gs2d.arguments import OptimizationParams
from smplinit.utils.poser import Skeleton
from argparse import ArgumentParser


class GaussianUser:
    class Config:
        radius: float = 4
        controlnet: bool = False
        smplx_path: str = "/home/jingyiwu/Dream2DGS/smplx_models"
        # pts_num: int = 100000

        num_pts: 100000
        sh_degree: 3
        position_lr_init: 0.00016
        position_lr_final: 0.0000016
        position_lr_delay_mult: 0.01
        position_lr_max_steps: 10000
        feature_lr: 0.0025
        opacity_lr: 0.05
        scaling_lr: 0.005
        rotation_lr: 0.001
        percent_dense: 0.01
        density_start_iter: 500
        density_end_iter: 10000
        densification_interval: 100
        opacity_reset_interval: 3000
        densify_grad_threshold: 0.0002
        densify_min_opacity: 0.05
        densify_extent: 4
        densify_max_screen_size: 1

        gender: str = 'neutral'


        apose: bool = True
        bg_white: bool = False

    cfg: Config
    def configure(self) -> None:
        self.gs_model = GaussianModel(sh_degree=3)
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.skel = Skeleton(apose=self.cfg.apose)

        self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.gender)
        self.skel.scale(-10)


        self.cameras_extent = 4.0

        self.parser = ArgumentParser(description="Training script parameters")


    def pcb(self):
        # Since this data set has no colmap data, we start with random points

        # coords,rgb,scale = self.shape()
        # bound= self.radius*scale
        # all_coords,all_rgb = self.add_points(coords,rgb)
        # pcd = BasicPointCloud(points=all_coords *bound, colors=all_rgb, normals=np.zeros((self.num_pts, 3)))

        #  使用SMPLX骨架模型中的点生成点云，从self.skel中采样N个smplx模型的点（10,000个）
        points = self.skel.sample_smplx_points(N=self.cfg.num_pts)
        colors = np.ones_like(points) * 0.5
        pcd = BasicPointCloud(points, colors, None)
        return pcd

    def configure_optimizers(self):
        opt = OptimizationParams(self.parser)

        point_cloud = self.pcb()
        self.gs_model.create_from_pcd(point_cloud, self.cameras_extent)

        self.gs_model.training_setup(opt)

        ret = {
            "optimizer": self.gs_model.optimizer,
        }

        return ret








