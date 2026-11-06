"""M3: CNN + LSTM tren chuoi frame da crop nguoi (chay crop_persons.py truoc).  python src/train_cnn_lstm.py"""
from fight_data import ClipDataset
from models import CNNLSTM
from train_utils import run_training, train_parser


def main():
    parser = train_parser("data/crops", "results/m3", batch_size=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    model = CNNLSTM(args.arch, args.hidden_size, not args.no_pretrained, args.freeze_backbone)
    run_training(args, model, ClipDataset, {"arch": args.arch, "hidden_size": args.hidden_size})


if __name__ == "__main__":
    main()
