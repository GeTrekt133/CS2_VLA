import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from collections import defaultdict
# from Dataset import CSRoundDataset
from DatasetIntent import CSRoundDataset
from tqdm import tqdm
import random


def debug_collate(batch):
    for k in batch[0].keys():
        shapes = [b[k].shape for b in batch if torch.is_tensor(b[k])]
        if len(set(shapes)) > 1:
            print(f"[COLLATE ERROR] {k}: {shapes}")
    return torch.utils.data.default_collate(batch)

class ExactMatchBucketSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        print("[INFO] Collecting sequence lengths...")
        self.buckets = defaultdict(list)

        for idx in tqdm(range(len(dataset))):
            scene_len, radar_len = dataset[idx]
            key = (scene_len, radar_len)
            self.buckets[key].append(idx)

        # фильтруем маленькие бакеты
        self.bucket_keys = [k for k, v in self.buckets.items() if len(v) >= batch_size]
        print(f"[INFO] Using {len(self.bucket_keys)} buckets: {self.bucket_keys}")

    def __iter__(self):
        all_batches = []
        for key in self.bucket_keys:
            indices = self.buckets[key]
            if self.shuffle:
                random.shuffle(indices)
            batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
            if len(batches[-1]) != self.batch_size:
                batches = batches[:-1]
            # if len(batches) > 0:
            #     print(key)
            #     break
            all_batches.extend(batches)
        if self.shuffle:
            random.shuffle(all_batches)
        return iter(all_batches)
    
    def __len__(self):
        return sum(len(self.buckets[k]) // self.batch_size for k in self.bucket_keys)


if __name__ == "__main__":
    # dataset = CSRoundDataset(r"C:\Users\misas\CS2_NN\train_dataset.json", sampler=True)
    dataset = CSRoundDataset(r"/mnt/ml/msirotkin/shock2/train_dataset.json", sampler=True)
    sampler = ExactMatchBucketSampler(dataset, batch_size=4, shuffle=True)
    # dataset = CSRoundDataset(r"C:\Users\misas\CS2_NN\train_dataset.json")
    dataset = CSRoundDataset(r"/mnt/ml/msirotkin/shock2/train_dataset.json")
    loader = DataLoader(dataset, 
                        batch_sampler=sampler,
                        collate_fn=debug_collate,
                        num_workers=4)

    for i, sample in tqdm(enumerate(loader)):
        # print(i, sample["game_id"], sample["round_id"])
        # print(sample["scene_seq"].shape, sample["radar_seq"].shape, sample["state_vec"].shape, sample["actions_mouse"].shape, sample["actions_keys"].shape, sample["labels_mouse"].shape, sample["labels_keys"].shape)
        print(sample["scene_seq"].shape, sample["radar_seq"].shape, sample["intent"].shape, sample["actions_mouse"].shape, sample["actions_keys"].shape, sample["target_mouse"].shape, sample["T"].shape)
        # print(sample["intent"])
        # print(sample["target_mouse"])
        # print(sample["actions_keys"], sample["actions_keys"].shape)
        # print(sample["actions_mouse"], sample["actions_mouse"].shape)
        if i == 200:
            break
