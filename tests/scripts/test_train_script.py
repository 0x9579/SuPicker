def test_train_parser_accepts_val_interval(monkeypatch):
    from scripts import train

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train-images", "train_imgs",
            "--train-star", "train.star",
            "--val-interval", "5",
        ],
    )

    args = train.parse_args()

    assert args.val_interval == 5
