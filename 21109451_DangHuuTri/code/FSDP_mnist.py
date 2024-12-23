import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
import torch.multiprocessing as mp


# Distributed setup
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# Model definition
class LLaMAModel(nn.Module):
    def __init__(self, model_name="meta-llama/LLaMA-3B"):
        super(LLaMAModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


# Training function
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    for batch in train_loader:
        input_ids = batch["input_ids"].to(rank)
        attention_mask = batch["attention_mask"].to(rank)
        labels = batch["labels"].to(rank)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item() * input_ids.size(0)
        ddp_loss[1] += input_ids.size(0)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f'Train Epoch: {epoch} \tLoss: {ddp_loss[0] / ddp_loss[1]:.6f}')


# Evaluation function
def test(model, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels = batch["labels"].to(rank)

            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum", ignore_index=-100)
            ddp_loss[0] += loss.item()
            ddp_loss[1] += (logits.argmax(dim=-1) == labels).sum().item()
            ddp_loss[2] += labels.numel()

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f'Test set: Average loss: {ddp_loss[0] / ddp_loss[2]:.4f}')


# Main training script
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-3B")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = dataset["validation"].map(tokenize_function, batched=True, remove_columns=["text"])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler)

    # Set FSDP policy
    fsdp_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e7)

    # Setup model and optimizer
    torch.cuda.set_device(rank)
    model = LLaMAModel().to(rank)
    model = FSDP(model, auto_wrap_policy=fsdp_policy)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=train_sampler)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    if args.save_model and rank == 0:
        torch.save(model.state_dict(), "llama3B_model.pt")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP with LLaMA-3B")
    parser.add_argument("--batch-size", type=int, default=4, metavar="N", help="input batch size for training")
    parser.add_argument("--test-batch-size", type=int, default=4, metavar="N", help="input batch size for testing")
    parser.add_argument("--epochs", type=int, default=3, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=3e-5, metavar="LR", help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma")
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(world_size, args), nprocs=world_size, join=True)
