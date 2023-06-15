from pyskl.datasets.pose_dataset import PoseDataset
import torchvision
ann_file = '/inspur/hdfs6/chenjunfen/xzh/dataset/hmdb51_hrnet.pkl'
dataset = PoseDataset(ann_file=ann_file, pipeline=[], split='train1')
print(dataset[0].keys())

torchvision.transforms