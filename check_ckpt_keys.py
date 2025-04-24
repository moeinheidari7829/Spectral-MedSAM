import torch
ckpt = torch.load('/home/moein/Desktop/BMEG591_Project/SAMed/output/Thyroid_512_pretrain_vit_b_epo100_bs2_lr0.0001/epoch_99.pth', map_location='cpu')
print('Checkpoint type:', type(ckpt))
if isinstance(ckpt, dict):
    print('Top-level keys:', ckpt.keys())
    if 'state_dict' in ckpt:
        print('state_dict keys:', ckpt['state_dict'].keys())
    else:
        print('Direct keys:', ckpt.keys())
else:
    print('Not a dict, cannot inspect keys')
