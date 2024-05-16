from tqdm import tqdm

for i in range(5):
    dict = {"a": 123, "b": 456}
    for i in tqdm(range(50), total=50, desc="WSX", ncols=100, postfix=dict, mininterval=0.3):
        pass
