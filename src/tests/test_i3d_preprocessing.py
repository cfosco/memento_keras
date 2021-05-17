'''
Tests for the preprocessing file, operating over i3d network.
i3d is DeepMind's Inception-v1 Inflated 3D CNN for action recognition.
The model is introduced in:
  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.
'''

__author__ = "Camilo Fosco"
__email__ = "cfosco@nvidia.com"


def test_resize(prev_frames, frames):
    print('Shape before resizing:', prev_frames.shape)
    print('Shape after resizing:', frames.shape)

def test_rescale(prev_frames, frames):
    print('max and min before rescaling:', np.max(prev_frames), np.min(prev_frames))
    print('max and min after rescaling:', np.max(frames), np.min(frames))

def test_crop(prev_frames, frames):
    print('Shape before cropping:', prev_frames.shape)
    print('Shape after cropping:', frames.shape)
    plot_frames(prev_frames, frames)

def test_npy_file_similarity(file1, file2, names=['ours','reference'], is_optical_flow=False, idxs=[0]):
    frames1 = np.load(file1)[0,:10,...]
    frames2 = np.load(file2)[0,:10,...]

    print('Shape of', names[0], frames1.shape, ' - Shape of', names[1], frames2.shape)
    print('Sum of differences:', np.sum(frames1-frames2))
    print('Max of video',names,':', np.max(frames1), np.max(frames2))
    print('Min of video',names,':', np.min(frames1), np.min(frames2))
    print('Avg of video',names,':', np.mean(frames1), np.mean(frames2))
    print('median of video',names,':', np.median(frames1), np.median(frames2))

    plot_frames(frames1, frames2, is_optical_flow=is_optical_flow, names=names, idxs=idxs)
