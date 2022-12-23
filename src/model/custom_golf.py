dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='josw631@gmail.com',
        title='golf',
        container='',
        year='2022',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(
            name='head',
            id=0,
            color=[0,0,255],
            type='upper',
            swap=''),
        1:
        dict(
            name='neck',
            id=1,
            color=[0,0,255],
            type='upper',
            swap=''),
        2:
        dict(
            name='left_shoulder',
            id=2,
            color=[0,0,255],
            type='upper',
            swap='right_shoulder'),
        3:
        dict(
            name='right_shoulder',
            id=3,
            color=[0,0,255],
            type='upper',
            swap='left_shoulder'),
        4:
        dict(
            name='left_elbow',
            id=4,
            color=[0,0,255],
            type='upper',
            swap='right_elbow'),
        5:
        dict(
            name='right_elbow',
            id=5,
            color=[0,0,255],
            type='upper',
            swap='left_elbow'),
        6:
        dict(
            name='left_wrist',
            id=6,
            color=[0,0,255],
            type='upper',
            swap='right_wrist'),
        7:
        dict(
            name='right_wrist',
            id=7,
            color=[0,0,255],
            type='upper',
            swap='left_wrist'),
        8:
        dict(
            name='left_hip',
            id=8,
            color=[0,0,255],
            type='lower',
            swap='right_hip'),
        9:
        dict(
            name='right_hip',
            id=9,
            color=[0,0,255],
            type='lower',
            swap='left_hip'),
        10:
        dict(
            name='left_knee',
            id=10,
            color=[0,0,255],
            type='lower',
            swap='right_knee'),
        11:
        dict(
            name='right_knee',
            id=11,
            color=[0,0,255],
            type='lower',
            swap='left_knee'),
        12:
        dict(
            name='left_ankle',
            id=12,
            color=[0,0,255],
            type='lower',
            swap='right_ankle'),
        13:
        dict(
            name='right_ankle',
            id=13,
            color=[0,0,255],
            type='lower',
            swap='left_ankle'),
        14:
        dict(name='hand', id=14, color=[0,0,255], type='upper', swap=''),
        15:
        dict(name='club', id=15, color=[0,0,255], type='', swap=''),
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[0, 51, 0]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[0, 51, 0]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[0, 51, 0]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[0, 51, 0]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[0, 255, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[255, 128, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('head', 'neck'), id=12, color=[204, 153, 51]),
        13:
        dict(link=('right_shoulder', 'neck'), id=13, color=[204, 153, 51]),
        14:
        dict(link=('left_shoulder', 'neck'), id=14, color=[204, 153, 51]),
        15:
        dict(link=('right_wrist', 'hand'), id=15, color=[0, 255, 0]),
        16:
        dict(link=('left_wrist', 'hand'), id=16, color=[0, 255, 0]),
        17:
        dict(link=('club', 'hand'), id=17, color=[0, 102, 255]),
        },
    joint_weights=[0.2, 0.5, 0.2, 0.2, 0.2, 1.3, 1.5, 0.2, 1.3, 1.5, 0.2, 0.2, 0.5, 0.2, 1.0, 1.0],
    sigmas=[
        0.079, 0.079, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089, 0.072, 0.062
    ])