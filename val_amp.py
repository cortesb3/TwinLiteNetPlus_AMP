import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import yaml
from argparse import ArgumentParser
from pathlib import Path

from model.model import TwinLiteNetPlus
from utils import val, netParams
from loss import TotalLoss
from AMP import AMPDataset


class AMPTestDataset(AMPDataset):
    '''
    AMP Test Dataset class that extends AMPDataset for test/train evaluation
    '''
    def __init__(self, hyp, split='val'):
        '''
        :param hyp: hyperparameters dictionary
        :param split: dataset split to evaluate on ('train', 'val', or 'test')
        '''
        # Use valid=True to disable augmentation for evaluation.
        # We override root/names below based on split.
        super().__init__(hyp, valid=True)

        split = split.lower()
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid split '{split}'. Choose from: train, val, test")

        self.root = hyp['dataset_path'] + f'/images/{split}'
            
        # Update names list
        import os
        if os.path.exists(self.root):
            self.names = [f for f in os.listdir(self.root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(self.names)} images in {self.root}")
        else:
            raise FileNotFoundError(f"Directory not found: {self.root}")


def validation_amp(args):
    """
    Perform model validation on the AMP dataset and return accuracy metrics.
    :param args: Parsed command-line arguments.
    """
    
    # Initialize model
    model = TwinLiteNetPlus(args)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.cuda()
        cudnn.benchmark = True
    
    # Load hyperparameters from YAML file
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    
    # Create test data loader
    print(f"Loading dataset from: {hyp['dataset_path']}")
    split = 'train' if args.test_on_train else args.split

    testLoader = torch.utils.data.DataLoader(
        AMPTestDataset(hyp, split=split),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Print model parameter count
    print(f'Total network parameters: {netParams(model)}')
    
    # Load pretrained weights
    if not args.weight or not Path(args.weight).exists():
        raise FileNotFoundError(f"Weight file not found: {args.weight}")

    print(f"Loading weights from: {args.weight}")
    checkpoint = torch.load(args.weight, map_location='cuda' if cuda_available else 'cpu')
    
    # Check if it's a saved checkpoint dictionary (from train.py)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Use EMA weights if they exist, otherwise fallback to standard weights
        if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            model.load_state_dict(checkpoint['ema_state_dict'])
            print("Loaded EMA weights from checkpoint")
        else:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded standard weights from checkpoint")
    else:
        # It's a raw weight file (like large.pth)
        model.load_state_dict(checkpoint)
        print("Loaded raw weights")
    
    model.eval()
    
    # Perform validation
    print("Evaluating model on AMP dataset...")
    da_segment_results, ll_segment_results = val(testLoader, model, args.half, args=args)
    
    # Baseline-style output (aligned with val.py)
    print(f"Driving Area Segment: mIOU({da_segment_results[2]:.3f})")
    if args.report_dummy_lane:
        print(f"Lane Line Segment (dummy target): Acc({ll_segment_results[0]:.3f}) IOU({ll_segment_results[1]:.3f})")
    
    return da_segment_results[2], ll_segment_results[0], ll_segment_results[1]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default="pretrained/large.pth", help='Path to model weights')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for validation')
    parser.add_argument('--config', type=str, choices=["nano", "small", "medium", "large"], default="large", help='Model configuration')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='Path to hyperparameters YAML file')
    parser.add_argument('--half', action='store_true', help='Use half precision for inference')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='val', help='Dataset split to evaluate')
    parser.add_argument('--test_on_train', action='store_true', help='Test on train set instead of val set (add flag to enable)')
    parser.add_argument('--report_dummy_lane', action='store_true', help='Also print lane metrics against dummy zero labels')
    
    # Parse arguments and run validation
    args = parser.parse_args()
    da_miou, ll_acc, ll_iou = validation_amp(args)