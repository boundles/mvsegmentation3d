import torch

from mvseg3d.datasets import build_dataloader
from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.models import VFE
from mvseg3d.models.backbones import PointTransformer
from mvseg3d.utils.config import cfg
from mvseg3d.utils.data_utils import load_data_to_gpu
from mvseg3d.utils.logging import get_root_logger

if __name__ == '__main__':
    logger = get_root_logger(name="test_point_transformer")

    # load data
    train_dataset = WaymoDataset(cfg, '/nfs/dataset-dtai-common/waymo_open_dataset_v_1_3_2', 'validation')
    logger.info('Loaded %d train samples' % len(train_dataset))

    train_set, train_loader, train_sampler = build_dataloader(
        dataset=train_dataset,
        batch_size=2,
        dist=True,
        num_workers=4,
        seed=0,
        training=True)

    dim_point = train_dataset.dim_point
    vfe = VFE(dim_point, reduce='mean')

    drop_info = {
        0: {'max_tokens': 30, 'drop_range': (0, 30)},
        1: {'max_tokens': 60, 'drop_range': (30, 60)},
        2: {'max_tokens': 100, 'drop_range': (60, 100000)}
    }
    window_shape = (20, 20, 20)
    model = PointTransformer(dim_point, 64, train_dataset.grid_size, train_dataset.voxel_size,
                             train_dataset.point_cloud_range, drop_info, window_shape).cuda()
    model.train()
    for step, data_dict in enumerate(train_loader):
        load_data_to_gpu(data_dict)

        points = data_dict['points'][:, 1:]
        point_voxel_ids = data_dict['point_voxel_ids']
        data_dict['voxel_features'] = vfe(points, point_voxel_ids)

        result = model(data_dict)
        print(result)
