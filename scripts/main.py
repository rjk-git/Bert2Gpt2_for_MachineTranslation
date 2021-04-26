import types
import argparse
import logging
from functools import partial
import json

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertGenerationConfig,
    BertGenerationEncoder,
    BertTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from utils import TSVDataset, collect_fn, build_inputs_with_special_tokens
import sacrebleu
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.ngpu > 1:
        torch.cuda.manual_seed_all(args.seed)


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt


def get_optimizer_and_schedule(args, model: EncoderDecoderModel):
    # 预训练参数和初始化参数使用不同的学习率
    if args.ngpu > 1:
        model = model.module
    init_params_id = []
    for layer in model.decoder.transformer.h:
        init_params_id.extend(list(map(id, layer.crossattention.parameters())))
        init_params_id.extend(list(map(id, layer.ln_cross_attn.parameters())))
    pretrained_params = filter(
        lambda p: id(p) not in init_params_id, model.parameters()
    )
    initialized_params = filter(lambda p: id(p) in init_params_id, model.parameters())
    params_setting = [
        {"params": initialized_params},
        {"params": pretrained_params, "lr": args.finetune_lr},
    ]
    optimizer = optim.Adam(params_setting, lr=args.lr)
    schedule = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=args.num_training_steps,
        num_warmup_steps=args.num_warmup_steps,
    )
    return optimizer, schedule


def get_model(args):
    if args.model_path:
        model = EncoderDecoderModel.from_pretrained(args.model_path)
        src_tokenizer = BertTokenizer.from_pretrained(
            os.path.join(args.model_path, "src_tokenizer")
        )
        tgt_tokenizer = GPT2Tokenizer.from_pretrained(
            os.path.join(args.model_path, "tgt_tokenizer")
        )
        tgt_tokenizer.build_inputs_with_special_tokens = types.MethodType(
            build_inputs_with_special_tokens, tgt_tokenizer
        )
        if local_rank == 0 or local_rank == -1:
            print("model and tokenizer load from save success")
    else:
        src_tokenizer = BertTokenizer.from_pretrained(args.src_pretrain_dataset_name)
        tgt_tokenizer = GPT2Tokenizer.from_pretrained(args.tgt_pretrain_dataset_name)
        tgt_tokenizer.add_special_tokens(
            {"bos_token": "[BOS]", "eos_token": "[EOS]", "pad_token": "[PAD]"}
        )
        tgt_tokenizer.build_inputs_with_special_tokens = types.MethodType(
            build_inputs_with_special_tokens, tgt_tokenizer
        )
        encoder = BertGenerationEncoder.from_pretrained(args.src_pretrain_dataset_name)
        decoder = GPT2LMHeadModel.from_pretrained(
            args.tgt_pretrain_dataset_name, add_cross_attention=True, is_decoder=True
        )
        decoder.resize_token_embeddings(len(tgt_tokenizer))
        decoder.config.bos_token_id = tgt_tokenizer.bos_token_id
        decoder.config.eos_token_id = tgt_tokenizer.eos_token_id
        decoder.config.vocab_size = len(tgt_tokenizer)
        decoder.config.add_cross_attention = True
        decoder.config.is_decoder = True
        model_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config
        )
        model = EncoderDecoderModel(
            encoder=encoder, decoder=decoder, config=model_config
        )
    if local_rank != -1:
        model = model.to(device)
    if args.ngpu > 1:
        print("{}/{} GPU start".format(local_rank, torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    optimizer, scheduler = get_optimizer_and_schedule(args, model)
    return model, src_tokenizer, tgt_tokenizer, optimizer, scheduler


def save_model(
    args,
    model,
    optimizer,
    src_tokenizer: BertTokenizer,
    tgt_tokenizer: GPT2Tokenizer,
    nstep,
    nepoch,
    bleu,
    loss,
):
    # 记录整体训练评价结果
    train_metric_log_file = os.path.join(args.output_dir, "training_metric.tsv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(train_metric_log_file):
        with open(train_metric_log_file, "a", encoding="utf-8") as fa:
            fa.write("{}\t{}\t{}\t{}\n".format(nepoch, nstep, loss, bleu))
    else:
        with open(train_metric_log_file, "w", encoding="utf-8") as fw:
            fw.write("epoch\tstep\tloss\tbleu\n")
            fw.write("{}\t{}\t{}\t{}\n".format(nepoch, nstep, loss, bleu))

    # 保存模型
    model_save_path = os.path.join(
        args.output_dir, "epoch{}_step{}/".format(nepoch, nstep)
    )
    os.makedirs(model_save_path)
    model.save_pretrained(model_save_path)
    if local_rank == 0 or local_rank == -1:
        print(
            "epoch:{} step:{} loss:{} bleu:{} model save complete.".format(
                nepoch, nstep, round(loss, 4), round(bleu, 4)
            )
        )
    if args.save_optimizer:
        torch.save(optimizer, os.path.join(model_save_path, "optimizer.pt"))

    # 保存tokenizer
    src_tokenizer.save_pretrained(os.path.join(model_save_path, "src_tokenizer"))
    tgt_tokenizer.save_pretrained(os.path.join(model_save_path, "tgt_tokenizer"))


def main(args):
    if local_rank == 0 or local_rank == -1:
        print(vars(args))
    model, src_tokenizer, tgt_tokenizer, optimizer, scheduler = get_model(args)
    if args.ispredict:
        while True:
            input_str = input("src: ")
            output_str = predict(
                input_str,
                model,
                src_tokenizer,
                tgt_tokenizer,
                args.max_src_len,
                args.max_tgt_len,
            )
            print(output_str)
    else:
        if args.eval_data_path:
            train_dataset = TSVDataset(data_path=args.train_data_path)
            eval_dataset = TSVDataset(data_path=args.eval_data_path)
            if local_rank == 0 or local_rank == -1:
                print(
                    "load train_dataset:{} and eval_dataset:{}".format(
                        len(train_dataset), len(eval_dataset)
                    )
                )
        else:
            dataset = TSVDataset(data_path=args.train_data_path)
            train_size = int(args.train_dataset_ratio * len(dataset))
            eval_size = len(dataset) - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset, [train_size, eval_size], generator=torch.Generator()
            )
            if local_rank == 0 or local_rank == -1:
                print(
                    "load dataset:{} split into train_dataset{} and eval_dataset:{}".format(
                        len(dataset), train_size, eval_size
                    )
                )
        collect_fn_ = partial(
            collect_fn,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=collect_fn_,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            sampler=DistributedSampler(train_dataset),
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=collect_fn_,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
        )
        train(
            args=args,
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
        )


def train(
    args,
    model: EncoderDecoderModel,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    src_tokenizer,
    tgt_tokenizer,
):
    eval_bleu = -1
    for epoch in range(args.nepoch):
        step = 0
        total_batch = train_dataloader.__len__()
        for data in train_dataloader:
            (
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                labels,
            ) = data
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                decoder_attention_mask=decoder_attention_mask.to(device),
                labels=labels.to(device),
                return_dict=True,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            model.zero_grad()
            if local_rank == 0 or local_rank == -1:
                if args.ngpu > 1:
                    reduced_loss = reduce_tensor(loss).cpu().item()
                else:
                    reduced_loss = loss.cpu().item()
                lr, finetune_lr = scheduler.get_lr()[0], scheduler.get_lr()[1]
                print(
                    "\rstep:{}/{}, bleu:{} loss:{}, lr:{}, ft.lr:{}".format(
                        step,
                        total_batch,
                        round(eval_bleu, 4),
                        round(reduced_loss, 4),
                        round(lr, 6),
                        round(finetune_lr, 6),
                    ),
                    end="",
                )
                writer.add_scalar("loss", reduced_loss, int(step * (1 + epoch)))
                writer.add_scalar("lr", lr, int(step * (1 + epoch)))
                writer.add_scalar("finetune_lr", finetune_lr, int(step * (1 + epoch)))
                if step % args.save_step == 0 or step % total_batch == 0:
                    eval_bleu = eval(
                        model, eval_dataloader, tgt_tokenizer, args.max_src_len
                    )
                    writer.add_scalar("bleu", eval_bleu, int(step * (1 + epoch)))
                    model_to_save = model.module if hasattr(model, "module") else model
                    save_model(
                        args,
                        model_to_save,
                        optimizer,
                        src_tokenizer,
                        tgt_tokenizer,
                        step,
                        epoch,
                        eval_bleu,
                        reduced_loss,
                    )


def eval(
    model: EncoderDecoderModel,
    eval_dataloader,
    tgt_tokenizer,
    max_src_len,
    num_beams=6,
):
    hyp, ref = [], []
    with torch.no_grad():
        for data in tqdm(eval_dataloader):
            input_ids, attention_mask, decoder_input_ids, _, _ = data
            if hasattr(model, "module"):
                generate = model.module.generate
            else:
                generate = model.generate
            outputs = generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=max_src_len,
                num_beams=num_beams,
                bos_token_id=tgt_tokenizer.bos_token_id,
                eos_token_id=tgt_tokenizer.eos_token_id,
            )
            hypoth = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refer = tgt_tokenizer.batch_decode(
                decoder_input_ids, skip_special_tokens=True
            )
            hyp.extend(hypoth)
            ref.extend(refer)
    bleu = sacrebleu.corpus_bleu(hyp, [ref])
    return bleu.score


def predict(
    input_str,
    model: EncoderDecoderModel,
    src_tokenizer,
    tgt_tokenizer,
    max_src_len,
    max_tgt_len,
    num_beam=6,
):
    inputs = src_tokenizer(
        [input_str],
        padding="max_length",
        truncation=True,
        max_length=max_src_len,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    if hasattr(model, "module"):
        generate = model.module.generate
    else:
        generate = model.generate
    outputs = generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_tgt_len,
        num_beams=num_beam,
        bos_token_id=tgt_tokenizer.bos_token_id,
        eos_token_id=tgt_tokenizer.eos_token_id,
    )
    output_str = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset_name", default="origin", type=str)
    parse.add_argument("--src_pretrain_dataset_name", default=None, type=str)
    parse.add_argument("--tgt_pretrain_dataset_name", default=None, type=str)
    parse.add_argument("--train_data_path", default=None, type=str)
    parse.add_argument("--eval_data_path", default=None, type=str)
    parse.add_argument("--log_path", default=None, type=str)
    parse.add_argument("--run_path", default=None, type=str)
    parse.add_argument("--output_dir", default="../checkpoints/", type=str)
    parse.add_argument("--optimizer", default="adam", type=str)
    parse.add_argument("--lr", default=1e-7, type=float)
    parse.add_argument("--finetune_lr", default=1e-5, type=float)
    parse.add_argument("--ngpu", default=1, type=int)
    parse.add_argument("--seed", default=17, type=int)
    parse.add_argument("--max_src_len", default=128, type=int)
    parse.add_argument("--max_tgt_len", default=128, type=int)
    parse.add_argument("--save_step", default=100, type=int)
    parse.add_argument("--num_training_steps", default=100, type=int)
    parse.add_argument("--num_warmup_steps", default=100, type=int)
    parse.add_argument("--nepoch", default=1, type=int)
    parse.add_argument("--num_workers", default=16, type=int)
    parse.add_argument("--train_batch_size", default=32, type=int)
    parse.add_argument("--eval_batch_size", default=32, type=int)
    parse.add_argument("--drop_last", default=False, action="store_true")
    parse.add_argument("--ispredict", action="store_true", default=False)
    parse.add_argument("--save_optimizer", action="store_true", default=False)
    parse.add_argument("--train_dataset_ratio", default=0.999, type=float)
    parse.add_argument("--model_path", default=None, type=str)
    parse.add_argument("--local_rank", default=0, type=int)
    args = parse.parse_args()
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = args.local_rank
        device = torch.device("cpu")
    if local_rank == 0 or local_rank == -1:
        sw_log_path = os.path.join(args.run_path, args.dataset_name)
        if not os.path.exists(sw_log_path):
            os.makedirs(sw_log_path)
        writer = SummaryWriter(sw_log_path)
    set_seed(args)
    main(args)
