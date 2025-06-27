import yaml

def load_config(main_config_path):
    with open(main_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # transforms도 별도 로딩해서 merge
    with open(cfg["TRANSFORMS_PATH"], "r") as f:
        transform_cfg = yaml.safe_load(f)
    
    cfg["transforms"] = transform_cfg["transforms"]
    return cfg