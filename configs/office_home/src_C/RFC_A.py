_base_ = [
    '../RFC_base.py'
]

# data
src, tgt = 'C', 'A'
info_path = f'./data/office_home_infos/{tgt}_list.txt'
data = dict(
    warmup=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    label=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    unlabel=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    test=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    test_label_or_unlabel=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    select=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
)

load = f'./checkpoints/office_home/src_{src}/train_src_{src}/best_val.pth'
