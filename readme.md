#Tiny Mapping
Minimal implementation of point cloud mapping.

## Dependency
- Open3D

## Data
A point cloud sequence is required, and there should be overlap between each point cloud.

## Run

#### 1. Modify the data paths in `mapping.py` :

```python
data_list = [
    '/path/to/your/point_cloud_1.pcd',
    '/path/to/your/point_cloud_2.pcd',
    # ...
    '/path/to/your/point_cloud_n.pcd',
]
```

#### 2.Run
```shell script
python mapping.py
```