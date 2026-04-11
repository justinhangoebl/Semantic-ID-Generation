import pickle
import torch


path = "./dataset/KGGLM - pre - ml-1m 2 week run_pretrain_paths_(5, 5).pickle"

with open(path, "rb") as f:
    data = pickle.load(f)


# for x in data.split("\n"):
#     if "I91 " in x:
#         print(x)


with open("./dataset/ml1m-for-KGGLM-finetune", "rb") as f:
    obj = pickle.load(f)

print("container type:", type(obj))
print("num entries:", len(obj))

for idx, (loader, payload) in enumerate(obj):
    print(f"\n== entry {idx} ==")
    print("loader type:", type(loader))
    print("payload dtype:", payload.dtype, "shape:", payload.shape)

    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        print("dataset type:", type(dataset))
        if hasattr(dataset, "__len__"):
            try:
                print("dataset len:", len(dataset))
            except Exception as exc:
                print("dataset len: <error>", repr(exc))
        try:
            sample = dataset[0]
            print("dataset[0] type:", type(sample))
            print("dataset[0] preview:", sample)
        except Exception as exc:
            print("dataset[0]: <error>", repr(exc))

    try:
        it = iter(loader)
        batch = next(it)
        print("batch type:", type(batch))
        print("batch preview:", batch)
    except Exception as exc:
        print("batch: <error>", repr(exc))