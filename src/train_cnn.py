"""M1: CNN baseline tren tung frame.  python src/train_cnn.py"""
from fight_data import FrameDataset
from models import build_cnn
from train_utils import run_training, train_parser


def main():
    args = train_parser("data/frames", "results/m1", batch_size=32).parse_args()
    model = build_cnn(args.arch, pretrained=not args.no_pretrained)
    run_training(args, model, FrameDataset, {"arch": args.arch})


if __name__ == "__main__":
    main()
