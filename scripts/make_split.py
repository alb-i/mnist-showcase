"""Write the fixed stratified validation split (IDs) and the cross-validation folds configured in
configs/split.yaml."""

from digits.data import (
    REPO_DIR,
    load_split,
    make_cv_folds,
    make_val_split,
    sample_ids,
    split_config,
)


def main() -> None:
    cfg = split_config()
    _, labels = load_split("train")
    assert len(labels) == cfg["n_train"] + cfg["n_val"]
    val_idx = make_val_split(labels, cfg["n_val"], cfg["seed"])
    ids = sample_ids("train", len(labels))[val_idx]
    path = REPO_DIR / cfg["val_ids_file"]
    path.write_text("\n".join(ids) + "\n")
    print(f"wrote {len(ids)} validation IDs to {path}")
    folds = make_cv_folds(labels, cfg["cv_folds"], cfg["cv_seed"])
    path = REPO_DIR / cfg["cv_folds_file"]
    ids = sample_ids("train", len(labels))
    path.write_text("".join(f"{i} {f}\n" for i, f in zip(ids, folds, strict=True)))
    print(f"wrote {cfg['cv_folds']} folds of {len(ids)} training IDs to {path}")


if __name__ == "__main__":
    main()
