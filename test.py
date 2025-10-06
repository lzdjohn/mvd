#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

# 极简脚本：直接打印文件中的数组或数据集
# 用法: python3 test.py [path/to/file]

DEFAULT = '/home/zhendong/MVDoppler-Pose/Data/dataset/2022Jul13-1744/20220713174516/output_3D/keypoint3D_adjusted.npz'
DEFAULT2 = '/home/zhendong/MVDoppler-Pose/Data/dataset/2022Jul13-1744/20220713174516/output_3D/keypoints.npy'

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT)
print('path:', path)


print(np.load(DEFAULT2).shape)
print(11111111111111111111111111)
try:
	import numpy as np
	data = np.load(str(path), allow_pickle=True)
	if hasattr(data, 'files'):
		for k in data.files:
			print('=== KEY:', k, '===')
			print(data[k].shape)
	else:
		print('aassssssssssaaa')
	data.close()
	sys.exit(0)
except Exception as e:
	try:
		import h5py
		with h5py.File(str(path), 'r') as hf:
			for k in hf.keys():
				print('=== DATASET:', k, '===')
				print(hf[k][...].shape)
		sys.exit(0)
	except Exception as e2:
		print('numpy error:', e)
		print('h5py error:', e2)
		sys.exit(1)
