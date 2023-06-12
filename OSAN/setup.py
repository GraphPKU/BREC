from setuptools import find_packages, setup

install_requires = [
    'ml_collections',
    'numba',
    'tqdm',
    'ortools',
    'sacred',
    'PyYAML',
    'tensorboard',
    'setuptools==59.5.0',
    'ogb==1.3.3',

    'torch@https://download.pytorch.org/whl/cu102/torch-1.10.1%2Bcu102-cp38-cp38-linux_x86_64.whl',
    'torch-scatter@https://data.pyg.org/whl/torch-1.10.0%2Bcu102/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl',
    'torch-sparse@https://data.pyg.org/whl/torch-1.10.0%2Bcu102/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl',

    # 'torch@https://download.pytorch.org/whl/cpu/torch-1.10.1%2Bcpu-cp38-cp38-linux_x86_64.whl',
    # 'torch-scatter@https://data.pyg.org/whl/torch-1.10.0%2Bcpu/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl',
    # 'torch-sparse@https://data.pyg.org/whl/torch-1.10.0%2Bcpu/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl',

    # 'torch@https://download.pytorch.org/whl/cu113/torch-1.10.1%2Bcu113-cp38-cp38-linux_x86_64.whl',
    # 'torch-scatter@https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl',
    # 'torch-sparse@https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl',

]

setup(name='diffsub',
      version='0.1.0',
      description='Differentiable subgraph sampling with I-MLE',
      packages=find_packages('.'),
      install_requires=install_requires,
      zip_safe=False)
