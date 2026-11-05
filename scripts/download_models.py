"""下载配置中固定 revision 的模型和共同预算 tokenizer，服务器可提前缓存。"""

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bert_mlm.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seen = set()
    for settings in [config["model"], config["gate"], config["retrieval"]]:
        if Path(settings["name"]).is_dir():
            continue
        key = (settings["name"], settings["revision"])
        if key in seen:
            continue
        seen.add(key)
        location = snapshot_download(settings["name"], revision=settings["revision"],
                                     ignore_patterns=["*.h5", "*.msgpack", "*.ot", "onnx/*", "openvino/*", "*.bin"])
        print(f"{key[0]} @ {key[1]}: {location}")
    for settings in config["input"]["comparison_tokenizers"]:
        if Path(settings["name"]).is_dir():
            continue
        key = (settings["name"], settings["revision"])
        if key not in seen:
            location = snapshot_download(settings["name"], revision=settings["revision"],
                                         allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json", "vocab.*", "merges.txt"])
            print(f"Tokenizer {key[0]} @ {key[1]}: {location}")


if __name__ == "__main__":
    main()
